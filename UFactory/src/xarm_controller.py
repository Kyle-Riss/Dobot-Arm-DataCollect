#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UFactory xArm Controller
Drop-in replacement for DobotE6Controller.
Exposes the same public interface plus .feed and .dashboard adapters
so robot_server.py works without structural changes.
"""

import time
import numpy as np
from typing import Optional, List

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    raise ImportError("xarm-python-sdk not installed: pip install xarm-python-sdk")

# ── xArm axis/direction jog mapping (matches Dobot MoveJog axis strings) ──────
_JOG_JOINT_MAP = {
    "J1+": (0,  1), "J1-": (0, -1),
    "J2+": (1,  1), "J2-": (1, -1),
    "J3+": (2,  1), "J3-": (2, -1),
    "J4+": (3,  1), "J4-": (3, -1),
    "J5+": (4,  1), "J5-": (4, -1),
    "J6+": (5,  1), "J6-": (5, -1),
}
_JOG_CART_MAP = {
    "X+": (0,  1), "X-": (0, -1),
    "Y+": (1,  1), "Y-": (1, -1),
    "Z+": (2,  1), "Z-": (2, -1),
    "Rx+": (3,  1), "Rx-": (3, -1),
    "Ry+": (4,  1), "Ry-": (4, -1),
    "Rz+": (5,  1), "Rz-": (5, -1),
}

# xArm state codes
_STATE_SPORT  = 1   # moving
_STATE_READY  = 2   # standby / idle
_STATE_PAUSE  = 3   # paused
_STATE_STOP   = 4   # stopped
_STATE_ERROR  = 9   # error (unofficial, varies)


class _FeedRecord:
    """Emulates a single-row numpy structured array returned by DobotApiFeedBack.feedBackData().
    Supports: feed['Field'][0], feed['Field'][0].tolist(), len(feed), feed.dtype.names
    """

    _NAMES = ("QActual", "ToolVectorActual", "RobotMode", "RunningStatus")

    def __init__(self, joints: List[float], pose: List[float], robot_mode: int, running: int):
        self._d = {
            "QActual":          [np.array(joints, dtype=np.float64)],
            "ToolVectorActual": [np.array(pose,   dtype=np.float64)],
            "RobotMode":        [np.uint64(robot_mode)],
            "RunningStatus":    [np.uint8(running)],
        }

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self._d[key]

    @property
    def dtype(self):
        names = self._NAMES
        class _DType:
            pass
        _DType.names = names
        return _DType()


class XArmFeedback:
    """Adapter that mirrors DobotApiFeedBack.feedBackData() using xArm polling."""

    def __init__(self, arm: XArmAPI):
        self._arm = arm

    def feedBackData(self) -> Optional[_FeedRecord]:
        try:
            code_j, joints = self._arm.get_servo_angle(is_radian=False)
            code_p, pose   = self._arm.get_position(is_radian=False)
            code_s, state  = self._arm.get_state()
            if code_j != 0 or code_p != 0:
                return None
            joints = list(joints)[:6] if joints else [0.0] * 6
            pose   = list(pose)[:6]   if pose   else [0.0] * 6

            # Map xArm state → Dobot robot_mode integer
            #   Dobot: 5=ENABLE(idle), 7=RUNNING, 9=ERROR
            #   xArm:  1=sport(moving), 2=standby, 3=pause, 4=stop, 9=error
            if state in (None, 0, 2, _STATE_READY):
                robot_mode = 5      # idle
            elif state == _STATE_SPORT:
                robot_mode = 7      # running
            elif state in (_STATE_PAUSE, _STATE_STOP):
                robot_mode = 10     # paused/stopped
            else:
                robot_mode = 9      # error

            running = 1 if state == _STATE_SPORT else 0
            return _FeedRecord(joints, pose, robot_mode, running)
        except Exception as e:
            print(f"[XArmFeedback] feedBackData error: {e}")
            return None

    def close(self):
        pass


class XArmDashboard:
    """Adapter that mirrors the DobotApiDashboard methods used by robot_server.py."""

    # Base speeds — scaled by speed_factor %
    _JOINT_BASE_DEG_S = 80.0    # deg/s at 100%
    _CART_TRANS_BASE  = 200.0   # mm/s  at 100%
    _CART_ROT_BASE    = 60.0    # deg/s at 100%

    def __init__(self, arm: XArmAPI):
        self._arm = arm
        self._speed_factor = 10     # percent (1-100); SpeedFactor() updates this
        self._jog_type: Optional[str] = None  # 'joint' | 'cart'

    # ── Enable / Disable ─────────────────────────────────────────────────────

    def EnableRobot(self, *args, **kwargs):
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        return "0,{},EnableRobot();"

    def DisableRobot(self):
        self._arm.motion_enable(False)
        return "0,{},DisableRobot();"

    def ClearError(self):
        self._arm.clean_error()
        self._arm.clean_warn()
        return "0,{},ClearError();"

    def ResumeRobot(self):
        self._arm.set_state(0)
        return "0,{},ResumeRobot();"

    # ── Speed ─────────────────────────────────────────────────────────────────

    def SpeedFactor(self, speed: int):
        self._speed_factor = max(1, min(100, int(speed)))
        return "0,{},SpeedFactor();"

    # ── Jog (velocity control) ────────────────────────────────────────────────

    def MoveJog(self, axis: str, coordtype: int = 0, user: int = 0, tool: int = 0):
        """
        axis: "J1+"..."J6+/-"  → joint velocity control (mode 4)
              "X+"..."Rz+/-"   → cartesian velocity control (mode 5)
              ""               → stop, back to position mode (mode 0)
        """
        pct = self._speed_factor / 100.0

        if axis == "":
            self._jog_stop()
            return "0,{},MoveJog();"

        if axis in _JOG_JOINT_MAP:
            jid, direction = _JOG_JOINT_MAP[axis]
            speeds = [0.0] * 7
            speeds[jid] = direction * self._JOINT_BASE_DEG_S * pct
            self._arm.set_mode(4)
            self._arm.set_state(0)
            self._arm.vc_set_joint_velocity(speeds, is_radian=False)
            self._jog_type = 'joint'
            return "0,{},MoveJog();"

        if axis in _JOG_CART_MAP:
            dof, direction = _JOG_CART_MAP[axis]
            speeds = [0.0] * 6
            if dof < 3:
                speeds[dof] = direction * self._CART_TRANS_BASE * pct
            else:
                speeds[dof] = direction * self._CART_ROT_BASE * pct
            self._arm.set_mode(5)
            self._arm.set_state(0)
            self._arm.vc_set_cartesian_velocity(speeds, is_radian=False)
            self._jog_type = 'cart'
            return "0,{},MoveJog();"

        print(f"[XArmDashboard] Unknown jog axis: {axis}")
        return "-1,{},MoveJog();"

    def _jog_stop(self):
        try:
            if self._jog_type == 'joint':
                self._arm.vc_set_joint_velocity([0.0] * 7, is_radian=False)
            else:
                self._arm.vc_set_cartesian_velocity([0.0] * 6, is_radian=False)
        except Exception:
            pass
        try:
            self._arm.set_mode(0)
            self._arm.set_state(0)
        except Exception:
            pass
        self._jog_type = None

    # ── Motion (MovJ / MovL) ──────────────────────────────────────────────────

    def MovJ(self, x, y, z, rx, ry, rz, coordinate_mode=0, v=30, a=50, **kwargs):
        if coordinate_mode == 1:
            code = self._arm.set_servo_angle(
                angle=[x, y, z, rx, ry, rz],
                speed=v, mvacc=a,
                is_radian=False, wait=False
            )
        else:
            code = self._arm.set_position(
                x, y, z, rx, ry, rz,
                speed=v, mvacc=a,
                is_radian=False, wait=False
            )
        return f"0,{{{code}}},MovJ();"

    def MovL(self, x, y, z, rx, ry, rz, coordinate_mode=0, v=50, a=500, **kwargs):
        code = self._arm.set_position(
            x, y, z, rx, ry, rz,
            speed=v, mvacc=a,
            radius=-1,              # -1 = straight line (no blending)
            is_radian=False, wait=False
        )
        return f"0,{{{code}}},MovL();"

    def RobotMode(self):
        _, state = self._arm.get_state()
        if state == _STATE_SPORT:
            mode = 7
        elif state in (_STATE_READY, 0):
            mode = 5
        elif state in (_STATE_PAUSE, _STATE_STOP):
            mode = 10
        else:
            mode = 9
        return f"0,{{{mode}}},RobotMode();"

    def GetAngle(self):
        _, joints = self._arm.get_servo_angle(is_radian=False)
        return f"0,{{{','.join(f'{v:.4f}' for v in joints[:6])}}},GetAngle();"

    def GetPose(self, **kwargs):
        _, pose = self._arm.get_position(is_radian=False)
        return f"0,{{{','.join(f'{v:.4f}' for v in pose[:6])}}},GetPose();"

    def ToolDOInstant(self, index: int, status: int):
        """Tool digital output — index 1→ionum 0, index 2→ionum 1."""
        ionum = index - 1
        self._arm.set_tgpio_digital(ionum, status)
        return f"0,{{}},ToolDOInstant({index},{status});"

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# XArmController — public interface matches DobotE6Controller
# ═══════════════════════════════════════════════════════════════════════════════

class XArmController:
    """UFactory xArm controller with the same public API as DobotE6Controller."""

    def __init__(self, ip: str = "192.168.1.225", **kwargs):
        self.ip = ip
        self._arm: Optional[XArmAPI] = None

        # Adapters exposed as .feed / .dashboard (accessed directly in robot_server.py)
        self.feed: Optional[XArmFeedback] = None
        self.dashboard: Optional[XArmDashboard] = None

        self.connected = False
        self.current_pose   = [0.0] * 6
        self.current_joints = [0.0] * 6
        self.last_move_response = ""

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self._arm = XArmAPI(self.ip, is_radian=False)
            self._arm.connect()
            if not self._arm.connected:
                print(f"✗ xArm connection failed: {self.ip}")
                return False

            self.feed      = XArmFeedback(self._arm)
            self.dashboard = XArmDashboard(self._arm)

            self._arm.clean_error()
            self._arm.clean_warn()
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            time.sleep(0.5)

            self._apply_safety_config()

            self.connected = True
            print(f"✓ xArm connected @ {self.ip}")
            code_j, joints = self._arm.get_servo_angle(is_radian=False)
            code_p, pose   = self._arm.get_position(is_radian=False)
            if code_j == 0:
                print(f"  joints: {[round(v,2) for v in joints[:6]]}")
            if code_p == 0:
                print(f"  pose  : {[round(v,2) for v in pose[:6]]}")
            return True
        except Exception as e:
            print(f"✗ xArm connect error: {e}")
            self.disconnect()
            return False

    # xArm6 공식 joint limits (from xarm_ros2/xarm_description xacro, degrees)
    _JOINT_LIMITS_DEG = [
        [-360.0, 360.0],  # J1
        [-118.0, 120.0],  # J2  ← 비대칭, 펌웨어 하드웨어 한계
        [-225.0,  11.0],  # J3  ← 비대칭, 가장 좁음
        [-360.0, 360.0],  # J4
        [ -97.0, 180.0],  # J5  ← 비대칭
        [-360.0, 360.0],  # J6
    ]

    def _apply_safety_config(self):
        """연결 시마다 안전 설정 확인 및 재적용."""
        try:
            # TCP 작업공간 경계 + 속도 제한
            if not self._arm.is_reduced_mode:
                self._arm.set_reduced_tcp_boundary([750, 100, 550, -550, 650, -100])
                self._arm.set_reduced_max_joint_speed(100, is_radian=False)
                self._arm.set_reduced_max_tcp_speed(300)
                # xacro 기반 joint 소프트웨어 한계 적용
                self._arm.set_reduced_joint_range(self._JOINT_LIMITS_DEG, is_radian=False)
                self._arm.set_reduced_mode(True)
                print("  [safety] reduced mode ON  (X:100~750, Y:-550~550, Z:-100~650 mm)")
                print(f"  [safety] joint limits: J2[-118~120] J3[-225~11] J5[-97~180] deg")
            else:
                print(f"  [safety] reduced mode already ON  boundary={self._arm.reduced_tcp_boundary}")
                # joint limits 항상 재확인
                cur = self._arm.reduced_joint_limits
                if cur and any(abs(c[0]) > 300 for c in cur[:6]):
                    self._arm.set_reduced_joint_range(self._JOINT_LIMITS_DEG, is_radian=False)
                    print("  [safety] joint limits re-applied")
            if not self._arm.is_collision_rebound:
                self._arm.set_collision_rebound(True)
                print("  [safety] collision rebound ON")
        except Exception as e:
            print(f"  [safety] config warning: {e}")

    def disconnect(self):
        if self._arm:
            try:
                self._arm.disconnect()
            except Exception:
                pass
            self._arm = None
        self.feed      = None
        self.dashboard = None
        self.connected = False
        print("xArm disconnected")

    # ── Enable / Error recovery ───────────────────────────────────────────────

    def enable_robot(self, sleep_after: float = 0.5):
        if self._arm:
            self._arm.motion_enable(True)
            self._arm.set_mode(0)
            self._arm.set_state(0)
            if sleep_after > 0:
                time.sleep(sleep_after)

    def disable_robot(self):
        if self._arm:
            self._arm.motion_enable(False)

    def resume_robot(self):
        if self._arm:
            try:
                self._arm.set_state(0)
                print("[resume_robot] state reset to 0")
            except Exception as e:
                print(f"[resume_robot] failed: {e}")

    def clear_error(self):
        if self._arm:
            self._arm.clean_error()
            self._arm.clean_warn()
            self._arm.set_state(0)
            time.sleep(0.3)

    def _get_robot_mode(self) -> int:
        if not self._arm:
            return -1
        try:
            _, state = self._arm.get_state()
            if state == _STATE_SPORT:
                return 7
            if state in (_STATE_READY, 0, None):
                return 5
            if state in (_STATE_PAUSE, _STATE_STOP):
                return 10
            return 9
        except Exception:
            return -1

    def _recover_if_needed(self):
        mode = self._get_robot_mode()
        if mode == 10:
            self.resume_robot()
        elif mode in (9, 11):
            self.clear_error()
            self.enable_robot(sleep_after=0.1)

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_j(self, x: float, y: float, z: float,
               rx: float, ry: float, rz: float,
               velocity: float = 30.0, accel: float = 50.0,
               coordinate_mode: int = 0,
               use_waypoint: bool = False) -> bool:
        """
        coordinate_mode=0 : TCP pose (x,y,z mm, rx,ry,rz deg)
        coordinate_mode=1 : joint angles (deg)
        """
        if not self._arm:
            print("✗ xArm not connected")
            return False

        self._recover_if_needed()

        try:
            if coordinate_mode == 1:
                code = self._arm.set_servo_angle(
                    angle=[x, y, z, rx, ry, rz],
                    speed=float(velocity), mvacc=float(accel),
                    is_radian=False, wait=False
                )
                label = f"Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)"
            else:
                code = self._arm.set_position(
                    x, y, z, rx, ry, rz,
                    speed=float(velocity), mvacc=float(accel),
                    is_radian=False, wait=False
                )
                label = f"Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})"

            if code == 0:
                print(f"MovJ → {label}")
                self.last_move_response = ""
                return True
            else:
                print(f"✗ MovJ failed (code={code}) → {label}")
                self.last_move_response = str(code)
                if code in (1, 2, 9):
                    self.clear_error()
                return False
        except Exception as e:
            print(f"✗ move_j exception: {e}")
            return False

    def move_l(self, x: float, y: float, z: float,
               rx: float, ry: float, rz: float,
               velocity: float = 50.0, coordinate_mode: int = 0) -> bool:
        if not self._arm:
            print("✗ xArm not connected")
            return False
        try:
            code = self._arm.set_position(
                x, y, z, rx, ry, rz,
                speed=float(velocity), mvacc=500.0,
                radius=-1, is_radian=False, wait=False
            )
            if code == 0:
                print(f"MovL → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                return True
            print(f"✗ MovL failed (code={code})")
            return False
        except Exception as e:
            print(f"✗ move_l exception: {e}")
            return False

    # ── State feedback ────────────────────────────────────────────────────────

    def get_current_joints_from_feedback(self) -> Optional[List[float]]:
        if not self._arm:
            return None
        try:
            code, joints = self._arm.get_servo_angle(is_radian=False)
            if code == 0 and joints:
                return list(joints)[:6]
        except Exception as e:
            print(f"✗ get joints error: {e}")
        return None

    def get_current_pose_from_feedback(self) -> Optional[List[float]]:
        if not self._arm:
            return None
        try:
            code, pose = self._arm.get_position(is_radian=False)
            if code == 0 and pose:
                self.current_pose = list(pose)[:6]
                return self.current_pose
        except Exception as e:
            print(f"✗ get pose error: {e}")
        return None

    def get_pose(self) -> Optional[List[float]]:
        return self.get_current_pose_from_feedback()

    # ── Motion complete ───────────────────────────────────────────────────────

    def wait_for_motion_complete(self, timeout: float = 30.0) -> bool:
        if not self._arm:
            return False

        start = time.time()
        seen_moving = False
        idle_start  = None

        while time.time() - start < timeout:
            try:
                _, state = self._arm.get_state()
                is_moving = (state == _STATE_SPORT)
                if is_moving:
                    seen_moving = True
                    idle_start  = None
                else:
                    if seen_moving:
                        print("✓ Motion complete")
                        return True
                    if idle_start is None:
                        idle_start = time.time()
                    elif time.time() - idle_start >= 0.3:
                        print("✓ Motion complete")
                        return True
            except Exception:
                pass
            time.sleep(0.05)

        print("✗ Motion timeout")
        return False

    # ── Digital output (suction gripper) ─────────────────────────────────────

    def set_digital_output(self, index: int, value: bool) -> bool:
        """
        index: 1-based (matches Dobot ToolDOInstant).
        Maps to xArm tool digital output (ionum = index-1).
        """
        if not self._arm:
            return False
        try:
            ionum  = index - 1
            status = 1 if value else 0
            code   = self._arm.set_tgpio_digital(ionum, status)
            state  = "ON" if value else "OFF"
            if code == 0:
                print(f"DO{index} (ionum={ionum}) = {state}")
                return True
            print(f"✗ set_digital_output failed (code={code})")
            return False
        except Exception as e:
            print(f"✗ set_digital_output exception: {e}")
            return False

    # ── IK pre-check (workspace estimate) ────────────────────────────────────

    def check_ik_solution(self, x: float, y: float, z: float,
                          rx: float, ry: float, rz: float) -> tuple:
        # reduced_tcp_boundary: [xmax=750, xmin=100, ymax=550, ymin=-550, zmax=650, zmin=-100]
        if not (100 <= x <= 750):
            return False, f"X={x:.0f}mm out of range [100, 750]"
        if not (-550 <= y <= 550):
            return False, f"Y={y:.0f}mm out of range [-550, 550]"
        if not (-100 <= z <= 650):
            return False, f"Z={z:.0f}mm out of range [-100, 650]"
        return True, "In workspace"

    def home(self) -> bool:
        return self.move_j(0, 0, 0, 0, 0, 0, coordinate_mode=1)


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    robot = XArmController(ip="192.168.1.225")
    try:
        if robot.connect():
            print("\n=== xArm connected ===")
            joints = robot.get_current_joints_from_feedback()
            pose   = robot.get_current_pose_from_feedback()
            print(f"joints : {joints}")
            print(f"pose   : {pose}")
            feed = robot.feed.feedBackData()
            if feed:
                print(f"feed joints     : {feed['QActual'][0].tolist()}")
                print(f"feed pose       : {feed['ToolVectorActual'][0].tolist()}")
                print(f"feed robot_mode : {int(feed['RobotMode'][0])}")
                print(f"feed running    : {int(feed['RunningStatus'][0])}")
        else:
            print("Connection failed")
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
