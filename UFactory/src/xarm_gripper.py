#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UFactory xArm Gripper G2 controller.
Range: 0 mm (fully closed) ~ 85 mm (fully open).
"""

import time
from typing import Optional

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    raise ImportError("xarm-python-sdk not installed: pip install xarm-python-sdk")

GRIPPER_OPEN_MM   = 85    # fully open
GRIPPER_CLOSE_MM  = 0     # fully closed
GRIPPER_GRIP_THR  = 20    # below this = is_gripping True
GRIPPER_SPEED_DEF = 100   # r/min
GRIPPER_FORCE_DEF = 50    # %


class XArmGripper:
    """xArm Gripper G2 — position control (mm)."""

    def __init__(self, arm: XArmAPI,
                 close_pos: int = GRIPPER_CLOSE_MM,
                 open_pos:  int = GRIPPER_OPEN_MM,
                 speed: int = GRIPPER_SPEED_DEF,
                 force: int = GRIPPER_FORCE_DEF):
        self._arm       = arm
        self.close_pos  = close_pos
        self.open_pos   = open_pos
        self.speed      = speed
        self.force      = force
        self._last_pos: Optional[float] = None
        self.is_gripping = False

        self._init_gripper()

    def _init_gripper(self):
        try:
            self._arm.set_gripper_enable(True)
            time.sleep(0.2)
            self._arm.set_gripper_mode(0)   # position mode
            time.sleep(0.2)
            self._last_pos = self.get_position()
            print(f"[XArmGripper] initialized, pos={self._last_pos:.1f}mm")
        except Exception as e:
            print(f"[XArmGripper] init error: {e}")

    # ── Position ──────────────────────────────────────────────────────────────

    def get_position(self) -> Optional[float]:
        """Returns current position in mm, or None on failure."""
        try:
            code, pos = self._arm.get_gripper_g2_position()
            if code == 0 and pos is not None:
                self._last_pos = float(pos)
                self.is_gripping = self._last_pos < GRIPPER_GRIP_THR
                return self._last_pos
        except Exception as e:
            print(f"[XArmGripper] get_position error: {e}")
        return self._last_pos

    def _ensure_position_mode(self):
        """Jog 후 mode 4/5 상태이면 position mode(0)로 복귀."""
        try:
            if self._arm.mode not in (0, None):
                self._arm.set_mode(0)
                self._arm.set_state(0)
                time.sleep(0.1)
        except Exception:
            pass

    def set_position(self, pos_mm: int, wait: bool = False,
                     speed: int = None, force: int = None) -> bool:
        """Move gripper to pos_mm (0~85). Returns True on success."""
        pos_mm = max(GRIPPER_CLOSE_MM, min(GRIPPER_OPEN_MM, int(pos_mm)))
        spd = speed if speed is not None else self.speed
        frc = force if force is not None else self.force
        self._ensure_position_mode()
        try:
            code = self._arm.set_gripper_g2_position(
                pos_mm, speed=spd, force=frc, wait=wait
            )
            if code == 0:
                self._last_pos = float(pos_mm)
                self.is_gripping = pos_mm < GRIPPER_GRIP_THR
                return True
            print(f"[XArmGripper] set_position failed (code={code})")
            return False
        except Exception as e:
            print(f"[XArmGripper] set_position error: {e}")
            return False

    # ── Grip / Release ────────────────────────────────────────────────────────

    def grip(self, wait_time: float = 0.5) -> bool:
        print(f"[Gripper] → CLOSE ({self.close_pos}mm)")
        ok = self.set_position(self.close_pos)
        if ok:
            self.is_gripping = True
            time.sleep(wait_time)
        return ok

    def release(self, wait_time: float = 0.3) -> bool:
        print(f"[Gripper] → OPEN ({self.open_pos}mm)")
        ok = self.set_position(self.open_pos)
        if ok:
            self.is_gripping = False
            time.sleep(wait_time)
        return ok

    def emergency_release(self):
        self.set_position(self.open_pos)
        self.is_gripping = False
        print("[Gripper] EMERGENCY OPEN")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    arm = XArmAPI("192.168.1.225", is_radian=False)
    arm.connect()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)

    g = XArmGripper(arm)
    print(f"position: {g.get_position()} mm")

    print("closing...")
    g.grip(wait_time=1.0)
    print(f"position: {g.get_position()} mm")

    print("opening...")
    g.release(wait_time=1.0)
    print(f"position: {g.get_position()} mm")

    arm.disconnect()
