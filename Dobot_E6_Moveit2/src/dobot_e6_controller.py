#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dobot E6 Robot Controller
TCP/IP communication with Dobot E6 robot using dobot_api.py
"""

import sys
import os
import time
import re
import numpy as np
from typing import Optional, List

# Windows 콘솔 UTF-8 인코딩 설정 (안전한 방법 - 이미 설정되어 있으면 건너뛰기)
if sys.platform == 'win32':
    try:
        import io
        # stdout이 이미 래핑되어 있지 않은 경우에만 설정
        if not isinstance(sys.stdout, io.TextIOWrapper) or (hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != 'utf-8'):
            if hasattr(sys.stdout, 'buffer') and not sys.stdout.buffer.closed:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'buffer') and not sys.stderr.buffer.closed:
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # 실패해도 계속 진행

# Add parent directory to path for importing dobot_api
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from dobot_api import DobotApiDashboard, DobotApiFeedBack
except ImportError:
    print("Error: dobot_api.py not found. Make sure it's in the project root.")
    sys.exit(1)


class DobotE6Controller:
    """Controller for Dobot E6 robot via TCP/IP using dobot_api"""
    
    def __init__(self, ip: str = "192.168.5.1", dashboard_port: int = 29999, feedback_port: int = 30004):
        """
        Initialize Dobot E6 controller
        
        Args:
            ip: Robot IP address (default: 192.168.5.1)
            dashboard_port: Dashboard server port (default 29999)
            feedback_port: Feedback server port (default 30004)
        """
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.feedback_port = feedback_port
        
        self.dashboard: Optional[DobotApiDashboard] = None
        self.feed: Optional[DobotApiFeedBack] = None
        
        self.connected = False
        
        # Current robot state
        self.current_pose = [0, 0, 0, 0, 0, 0]  # x, y, z, rx, ry, rz
        self.current_joints = [0, 0, 0, 0, 0, 0]
        self.last_move_response = ""  # 실패 시 로봇 응답 문자열 (GUI 로그용)
        
    def connect(self) -> bool:
        """
        Connect to robot dashboard and feedback servers
        
        Returns:
            True if connection successful
        """
        try:
            # Connect to dashboard server
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            print(f"✓ Connected to Dashboard server at {self.ip}:{self.dashboard_port}")
            
            # Connect to feedback server
            self.feed = DobotApiFeedBack(self.ip, self.feedback_port)
            print(f"✓ Connected to Feedback server at {self.ip}:{self.feedback_port}")
            
            # Enable robot
            self.enable_robot()
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.dashboard:
            try:
                self.dashboard.close()
            except:
                pass
            self.dashboard = None
            
        if self.feed:
            try:
                self.feed.close()
            except:
                pass
            self.feed = None
            
        self.connected = False
        print("Disconnected from robot")
    
    def enable_robot(self, sleep_after: float = 0.5):
        """Enable robot motors. sleep_after: 대기(초). 이동 직전 호출 시 0.1 정도로 짧게."""
        if self.dashboard:
            result = self.dashboard.EnableRobot()
            print(f"Enable robot: {result}")
            if sleep_after > 0:
                time.sleep(sleep_after)
    
    def disable_robot(self):
        """Disable robot motors"""
        if self.dashboard:
            result = self.dashboard.DisableRobot()
            print(f"Disable robot: {result}")
    
    def clear_error(self):
        """Clear robot errors"""
        if self.dashboard:
            result = self.dashboard.ClearError()
            print(f"Clear error: {result}")
            time.sleep(0.5)  # Wait a bit after clearing error
    
    def move_j_relative(self, dx: float, dy: float, dz: float, drx: float = 0, dry: float = 0, drz: float = 0,
                        velocity: int = 30) -> bool:
        """
        상대 이동 (베이스/유저 좌표계). IK 절대 좌표 대신 오프셋만 보내서 꺾인 자세에서도 동작하기 쉬움.
        RelMovJUser(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz)
        """
        if not self.dashboard:
            print("✗ Dashboard not connected")
            return False
        self.clear_error()
        self.enable_robot(sleep_after=0.1)
        result = self.dashboard.RelMovJUser(dx, dy, dz, drx, dry, drz, v=velocity)
        if result is None or result == "":
            print(f"RelMovJUser ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) [accepted]")
            return True
        result_str = str(result).strip()
        self.last_move_response = result_str
        if "Control Mode Is Not Tcp" in result_str or "Not Tcp" in result_str:
            print(f"✗ RelMovJUser failed: Robot not in TCP mode")
            return False
        err = re.search(r'\{([^\}]*)\}', result_str)
        if err:
            code = err.group(1).strip()
            if code == "" or code == "[]" or code == "0":
                self.last_move_response = ""
                print(f"RelMovJUser ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) [accepted]")
                return True
            try:
                if int(code) != 0:
                    print(f"✗ RelMovJUser failed: error {code}")
                    self.clear_error()
                    return False
            except ValueError:
                pass
        if result_str.startswith("0,"):
            self.last_move_response = ""
            print(f"RelMovJUser ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) [accepted]")
            return True
        if result_str.startswith("-"):
            print(f"✗ RelMovJUser failed: {result_str}")
            self.clear_error()
            return False
        print(f"RelMovJUser ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) [accepted]")
        return True
    
    def move_to_safe_position(self) -> bool:
        """
        Move robot to a safe intermediate position to avoid collisions
        
        Returns:
            True if successful
        """
        print("Moving to safe intermediate position...")
        
        # Get current position
        current_pose = self.get_current_pose_from_feedback()
        if not current_pose:
            # If can't get current position, try joint mode home
            return self.move_j(0, 0, 0, 0, 0, 0, coordinate_mode=1, use_waypoint=False)
        
        x, y, z, rx, ry, rz = current_pose
        
        # Create safe intermediate position (higher Z, closer to origin)
        safe_x = x * 0.5  # Move halfway to origin
        safe_y = y * 0.5
        safe_z = max(z + 50, 200)  # At least 200mm high
        
        # Ensure safe position is within workspace (상한선 완화 적용)
        safe_radius = np.sqrt(safe_x**2 + safe_y**2)
        max_safe_radius = self._get_max_radius_for_z(safe_z)
        if safe_radius > max_safe_radius:
            # Normalize to safe radius
            scale = max_safe_radius / safe_radius
            safe_x *= scale
            safe_y *= scale
        
        print(f"Safe position: ({safe_x:.1f}, {safe_y:.1f}, {safe_z:.1f})")
        
        # Try to move to safe position
        success = self.move_j(safe_x, safe_y, safe_z, rx, ry, rz, 
                             velocity=30.0, coordinate_mode=0, use_waypoint=False)
        
        if success:
            self.wait_for_motion_complete()
            return True
        else:
            # Fallback to joint mode
            print("Falling back to joint mode home...")
            return self.move_j(0, 0, 0, 0, 0, 0, coordinate_mode=1, use_waypoint=False)
    
    def check_ik_solution(self, x: float, y: float, z: float, rx: float, ry: float, rz: float) -> tuple[bool, str]:
        """
        Check if IK solution exists for given pose (pre-check before movement)
        
        Args:
            x, y, z: Position in mm
            rx, ry, rz: Orientation in degrees
            
        Returns:
            (success, message) tuple
        """
        if not self.dashboard:
            return False, "Dashboard not connected"
        
        # Try a test MovJ command (very slow, just to check IK)
        # Note: This will actually move the robot, so use with caution
        # Better approach: Check workspace limits first
        
        radius = np.sqrt(x**2 + y**2)
        max_radius = self._get_max_radius_for_z(z)
        
        if radius > max_radius:
            return False, f"Position outside estimated workspace (radius: {radius:.1f}mm > {max_radius:.1f}mm for Z={z:.1f}mm)"
        
        # Check if position is in reasonable range
        if z < -100 or z > 600:
            return False, f"Z coordinate out of range: {z:.1f}mm"
        
        if abs(rx) > 180 or abs(ry) > 180 or abs(rz) > 180:
            return False, f"Orientation angles out of range: RX={rx:.1f}°, RY={ry:.1f}°, RZ={rz:.1f}°"
        
        return True, "Position appears reachable (estimated)"
    
    def move_j(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, 
               velocity: float = 30.0, coordinate_mode: int = 0, use_waypoint: bool = False) -> bool:
        """
        Move to position in joint space (point-to-point)
        
        Args:
            x, y, z: Position in mm (if coordinate_mode=0) or Joint angle in degrees (if coordinate_mode=1)
            rx, ry, rz: Orientation in degrees (if coordinate_mode=0) or Joint angle in degrees (if coordinate_mode=1)
            velocity: Movement velocity percentage (0-100)
            coordinate_mode: 0 for pose (x,y,z,rx,ry,rz), 1 for joint angles
            use_waypoint: If True, use intermediate waypoint if direct movement fails
            
        Returns:
            True if command sent successfully, False if failed
        """
        if not self.dashboard:
            print("✗ Dashboard not connected")
            return False
        
        # 이동 시도 전 에러 클리어 + 로봇 활성화(가능한 범위인데 안 움직일 때 대비)
        self.clear_error()
        self.enable_robot(sleep_after=0.1)
        
        # (이전 단순 동작으로 복구) move_j에서는 추가 속도 조정/경고 없이 그대로 명령만 보냄
        radius = None
        if coordinate_mode == 0:
            radius = (x**2 + y**2)**0.5
        
        # 조인트 모드: MovJ(joint, 1) 사용. RunTo는 일부 컨트롤러에서 미동작할 수 있음.
        # VelJ로 관절 속도 설정 후 MovJ 전송 (연속 재생에 유리)
        if coordinate_mode == 1:
            try:
                self.dashboard.VelJ(int(velocity))
            except Exception:
                pass
            result = self.dashboard.MovJ(x, y, z, rx, ry, rz, 1, v=int(velocity), a=50)
        else:
            result = self.dashboard.MovJ(x, y, z, rx, ry, rz, coordinate_mode, v=int(velocity), a=50)
        
        # Check for errors in result
        # Note: MovJ may return None, empty string, or error code string
        # Response format: "0,{error_code},Command(...)" or "-error_code,..."
        if result is None or result == "":
            self.last_move_response = ""
            if coordinate_mode == 0:
                print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            else:
                print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
            return True
        
        # Check if result contains error code
        result_str = str(result).strip()
        self.last_move_response = result_str
        
        # Debug: Print raw response for troubleshooting (MovJ or RunTo)
        print(f"[DEBUG] Move response: {result_str}")
        
        # Check for "Control Mode Is Not Tcp" error - this means robot is not in TCP mode
        if "Control Mode Is Not Tcp" in result_str or "Not Tcp" in result_str:
            print(f"✗ Movement failed: Robot is not in TCP mode")
            print(f"  Response: {result_str}")
            print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            print(f"  ⚠️  Solution: Robot needs to be in TCP control mode")
            print(f"     This usually means the robot controller is in a different mode")
            print(f"     (e.g., joint mode, manual mode, or teaching mode)")
            print(f"     Please switch the robot to TCP mode using the controller")
            return False
        
        # Check for error code in format "0,{error_code},..." (e.g., "0,{8047},MovJ(...)")
        # Also handle empty error code: "0,{},Command(...)" or "0,{[]},Command(...)" which means success
        # Pattern: {숫자} 또는 {} 또는 {[]}
        error_match = re.search(r'\{([^\}]*)\}', result_str)
        if error_match:
            error_code_str = error_match.group(1).strip()
            
            # Empty / 0 / [] = 성공 ("0,{},MovJ" 또는 "0,{0},MovJ" 등)
            # 컨트롤러가 수락했으면 성공으로 처리
            if error_code_str.strip() == "" or error_code_str == "[]" or error_code_str.strip() == "0":
                self.last_move_response = ""
                if coordinate_mode == 0:
                    print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}) [accepted]")
                else:
                    print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°) [accepted]")
                return True
            
            # Try to parse as integer error code
            try:
                error_code = int(error_code_str)
                if error_code != 0:
                    print(f"✗ Movement failed with error code: {error_code}")
                    print(f"  Response: {result_str}")
                    print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                    
                    # Common error codes:
                    # 8047: IK solution not found (Algorithm error)
                    # 8048: Collision detection
                    if error_code == 8047 or error_code == 2:
                        print("  Error: Inverse kinematics solution not found (Algorithm error)")
                        if radius is not None:
                            print(f"  Problem: Low Z ({z:.1f}mm) + Large radius ({radius:.1f}mm) combination")
                        else:
                            print(f"  Problem: IK solution not found for target position")
                        self.clear_error()
                        
                        # Try with simple waypoint strategy if enabled (minimal Z adjustment)
                        if use_waypoint and coordinate_mode == 0:
                            print("  Attempting simple waypoint strategy...")
                            current_pose = self.get_current_pose_from_feedback()
                            if current_pose:
                                # Simple strategy: small intermediate waypoint (minimal Z change)
                                # Only add 20-30mm to Z if current Z is very low
                                if z < 50:
                                    waypoint_z = z + 30  # Only add 30mm for very low Z
                                else:
                                    waypoint_z = max(current_pose[2], z) + 20  # Small increment
                                
                                # Try intermediate waypoint at same XY, slightly higher Z
                                waypoint1_result = self.dashboard.MovJ(
                                    x, y, waypoint_z, rx, ry, rz, 0, v=int(velocity)
                                )
                                
                                # Check waypoint result
                                waypoint1_success = False
                                if waypoint1_result is None or waypoint1_result == "":
                                    waypoint1_success = True
                                else:
                                    waypoint1_str = str(waypoint1_result).strip()
                                    waypoint_error_match = re.search(r'\{([^\}]*)\}', waypoint1_str)
                                    if waypoint_error_match:
                                        waypoint_error_code_str = waypoint_error_match.group(1).strip()
                                        if waypoint_error_code_str == "" or waypoint_error_code_str == "[]":
                                            waypoint1_success = True
                                        else:
                                            try:
                                                waypoint_error_code = int(waypoint_error_code_str)
                                                if waypoint_error_code == 0:
                                                    waypoint1_success = True
                                            except ValueError:
                                                pass
                                    elif not waypoint1_str.startswith("-"):
                                        waypoint1_success = True
                                
                                if waypoint1_success:
                                    self.wait_for_motion_complete()
                                    print(f"  ✓ Reached intermediate Z ({waypoint_z:.1f}mm)")
                                    
                                    # Now try target Z
                                    final_result = self.dashboard.MovJ(x, y, z, rx, ry, rz, 0, v=int(velocity))
                                    
                                    # Check final result
                                    final_success = False
                                    if final_result is None or final_result == "":
                                        final_success = True
                                    else:
                                        final_str = str(final_result).strip()
                                        final_error_match = re.search(r'\{([^\}]*)\}', final_str)
                                        if final_error_match:
                                            final_error_code_str = final_error_match.group(1).strip()
                                            if final_error_code_str == "" or final_error_code_str == "[]":
                                                final_success = True
                                            else:
                                                try:
                                                    final_error_code = int(final_error_code_str)
                                                    if final_error_code == 0:
                                                        final_success = True
                                                except ValueError:
                                                    pass
                                        elif not final_str.startswith("-"):
                                            final_success = True
                                    
                                    if final_success:
                                        print(f"✓ Reached target via waypoint")
                                        return True
                        
                        print("  Solution:")
                        print("    1. Check if position is actually reachable")
                        print("    2. Move robot to a different position first")
                        print("    3. Try adjusting orientation angles (especially Rx)")
                        print("    4. Try increasing Z height slightly")
                        print("    5. Use joint mode for manual control")
                        return False
                elif error_code == 8048 or error_code == 1:
                    print("  ⚠️  Error: Robot collision or joint limit exceeded")
                    print("  ⚠️  COLLISION DETECTED - Robot movement stopped for safety")
                    self.clear_error()
                    print("  Solution:")
                    print("    1. Check robot position - may be in collision state")
                    print("    2. Manually move robot to safe position")
                    print("    3. Verify target position is reachable")
                    print("    4. Check joint limits and workspace boundaries")
                    return False
                elif 100 <= error_code <= 299:
                    # 경고 코드: 컨트롤러가 명령 수락 후 동작하는 경우 (예: 182, 183, 184...) — 성공으로 처리
                    if coordinate_mode == 0:
                        print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}) [accepted, warning {error_code}]")
                    else:
                        print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°) [accepted, warning {error_code}]")
                    return True
                else:
                    print(f"  Unknown error code: {error_code}")
                    self.clear_error()
                    return False
            except ValueError:
                # Not a numeric error code, might be array or other format
                # If it's not empty and not a number, log it but don't fail
                if error_code_str and error_code_str != "[]":
                    print(f"[WARNING] Unexpected error code format: '{error_code_str}' in response: {result_str}")
                # Continue to check other formats
        
        if result_str.startswith("-") or "error" in result_str.lower():
            # Parse error code
            try:
                if result_str.startswith("-"):
                    error_code = int(result_str.split(',')[0])
                    if error_code < 0:
                        print(f"✗ Movement failed with error code: {error_code}")
                        print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                        
                        # Clear error first
                        if error_code == -2:
                            print("  Error: Inverse kinematics solution not found")
                            if radius is not None:
                                print(f"  Problem: Low Z ({z:.1f}mm) + Large radius ({radius:.1f}mm) combination")
                            else:
                                print(f"  Problem: IK solution not found for target position")
                            self.clear_error()
                            
                            # Try with simple waypoint strategy if enabled (minimal Z adjustment)
                            if use_waypoint and coordinate_mode == 0:
                                print("  Attempting simple waypoint strategy...")
                                current_pose = self.get_current_pose_from_feedback()
                                if current_pose:
                                    # Simple strategy: small intermediate waypoint (minimal Z change)
                                    # Only add 20-30mm to Z if current Z is very low
                                    if z < 50:
                                        waypoint_z = z + 30  # Only add 30mm for very low Z
                                    else:
                                        waypoint_z = max(current_pose[2], z) + 20  # Small increment
                                    
                                    # Try intermediate waypoint at same XY, slightly higher Z
                                    waypoint1_success = self.dashboard.MovJ(
                                        x, y, waypoint_z, rx, ry, rz, 0, v=int(velocity)
                                    )
                                    
                                    if waypoint1_success and waypoint1_success != "" and not str(waypoint1_success).strip().startswith("-"):
                                        self.wait_for_motion_complete()
                                        print(f"  ✓ Reached intermediate Z ({waypoint_z:.1f}mm)")
                                        
                                        # Now try target Z
                                        final_result = self.dashboard.MovJ(x, y, z, rx, ry, rz, 0, v=int(velocity))
                                        if final_result and final_result != "" and not str(final_result).strip().startswith("-"):
                                            print(f"✓ Reached target via waypoint")
                                            return True
                            
                            print("  Solution:")
                            print("    1. Check if position is actually reachable")
                            print("    2. Move robot to a different position first")
                            print("    3. Try adjusting orientation angles")
                            print("    4. Use joint mode for manual control")
                        elif error_code == -1:
                            print("  Error: Robot collision or joint limit exceeded")
                            self.clear_error()
                            print("  Solution: Check robot position and joint limits")
                        
                        return False
            except Exception as e:
                print(f"✗ Error parsing result: {e}")
                print(f"  Result string: {result_str}")
                return False
        
        # Check if response indicates success (format: "0,{},Command(...)" or "0,{0},Command(...)")
        if result_str.startswith("0,"):
            if ",{}," in result_str or ",{[]}," in result_str or (",{}" in result_str and "MovJ" in result_str):
                if coordinate_mode == 0:
                    print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}) [accepted]")
                else:
                    print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°) [accepted]")
                return True
            if ",{0}," in result_str or result_str.startswith("0,{0},"):
                if coordinate_mode == 0:
                    print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}) [accepted]")
                else:
                    print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°) [accepted]")
                return True
            else:
                # Unknown format starting with "0," - check for common error messages
                if "Control Mode Is Not Tcp" in result_str or "Not Tcp" in result_str:
                    print(f"✗ Movement failed: Robot is not in TCP mode")
                    print(f"  Response: {result_str}")
                    print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                    print(f"  ⚠️  Solution: Robot needs to be in TCP control mode")
                    print(f"     Please switch the robot to TCP mode using the controller")
                    return False
                
                # Unknown format starting with "0," - might contain error code in different format
                print(f"[WARNING] Unknown response format: {result_str}")
                print(f"[WARNING] Please check robot movement manually")
                if coordinate_mode == 0:
                    print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                else:
                    print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
                return True
        
        # Check for "Control Mode Is Not Tcp" in any response format
        if "Control Mode Is Not Tcp" in result_str or "Not Tcp" in result_str:
            print(f"✗ Movement failed: Robot is not in TCP mode")
            print(f"  Response: {result_str}")
            print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            print(f"  ⚠️  Solution: Robot needs to be in TCP control mode")
            print(f"     Please switch the robot to TCP mode using the controller")
            return False
        
        # No error detected - success (fallback)
        if coordinate_mode == 0:
            print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
        else:
            print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
        return True
    
    def _get_max_radius_for_z(self, z_height: float) -> float:
        """
        Z 높이에 따른 최대 반경 계산 (상한선 완화 - 캘리브레이션 데이터 기반)
        
        캘리브레이션 데이터 분석 결과:
        - Z=74-79mm에서 반경 350-450mm 도달 가능 확인됨
        - 따라서 낮은 Z에서도 더 큰 반경 허용
        
        Dobot E6의 실제 작업 공간 (캘리브레이션 데이터 반영):
        - 낮은 Z (0-100mm): 중간 반경 (~500mm, 캘리브레이션 데이터 기반)
        - 중간 Z (100-300mm): 최대 반경 (~550mm)
        - 높은 Z (300-500mm): 중간 반경 (~500mm)
        - 매우 높은 Z (500-600mm): 작은 반경 (~450mm)
        
        Args:
            z_height: Z 좌표 (mm)
            
        Returns:
            해당 Z 높이에서의 최대 반경 (mm)
        """
        if z_height < 100:
            # 캘리브레이션 데이터: Z=74-79mm에서 반경 350-450mm 도달 가능
            # 따라서 낮은 Z에서도 500mm까지 허용
            return 500.0  # 낮은 높이: 중간 반경 (캘리브레이션 데이터 기반)
        elif z_height < 300:
            # 선형 보간: 100mm에서 500mm, 300mm에서 550mm
            return 500.0 + (z_height - 100) * (550.0 - 500.0) / (300.0 - 100.0)
        elif z_height < 500:
            # 선형 보간: 300mm에서 550mm, 500mm에서 500mm
            return 550.0 - (z_height - 300) * (550.0 - 500.0) / (500.0 - 300.0)
        else:
            return 450.0  # 매우 높은 높이: 작은 반경
        
        # Check for errors in result
        if result:
            # Check if result contains error code
            result_str = str(result)
            if "error" in result_str.lower() or result_str.startswith("-"):
                # Parse error code
                try:
                    if result_str.startswith("-"):
                        error_code = int(result_str.split(',')[0])
                        if error_code < 0:
                            print(f"✗ Movement failed with error code: {error_code}")
                            print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                            
                            # Clear error first
                            if error_code == -2:
                                print("  Error: Inverse kinematics solution not found")
                                print(f"  Problem: Low Z ({z:.1f}mm) + Large radius ({radius:.1f}mm) combination")
                                self.clear_error()
                                
                                # Try with improved waypoint strategy if enabled
                                if use_waypoint and coordinate_mode == 0:
                                    print("  Attempting improved waypoint strategy for IK failure...")
                                    current_pose = self.get_current_pose_from_feedback()
                                    if current_pose:
                                        # Strategy 1: Higher Z first, then move to target XY, then lower Z
                                        # This is especially effective for low Z + large radius positions
                                        print("  Strategy 1: Raise Z first, then move to target XY...")
                                        waypoint_z = max(current_pose[2], z, 200) + 100  # At least 200mm, add 100mm
                                        
                                        # First waypoint: higher Z at current XY
                                        waypoint1_success = self.dashboard.MovJ(
                                            current_pose[0], current_pose[1], waypoint_z, rx, ry, rz, 0, v=int(velocity)
                                        )
                                        
                                        if waypoint1_success and not str(waypoint1_success).startswith("-"):
                                            self.wait_for_motion_complete()
                                            print(f"  ✓ Reached higher Z ({waypoint_z:.1f}mm)")
                                            
                                            # Second waypoint: move to target XY at high Z
                                            waypoint2_success = self.dashboard.MovJ(
                                                x, y, waypoint_z, rx, ry, rz, 0, v=int(velocity)
                                            )
                                            
                                            if waypoint2_success and not str(waypoint2_success).startswith("-"):
                                                self.wait_for_motion_complete()
                                                print(f"  ✓ Reached target XY at high Z")
                                                
                                                # Final: lower to target Z (if possible)
                                                final_result = self.dashboard.MovJ(x, y, z, rx, ry, rz, 0, v=int(velocity))
                                                if final_result and not str(final_result).startswith("-"):
                                                    print(f"✓ Reached target via waypoint strategy")
                                                    return True
                                                else:
                                                    print(f"  ⚠️  Could not lower to target Z={z:.1f}mm, but reached XY at Z={waypoint_z:.1f}mm")
                                                    print(f"  This is a partial success - position is reachable at higher Z")
                                                    return True  # Partial success
                                        
                                        # Strategy 2: Try different orientations if Strategy 1 fails
                                        print("  Strategy 2: Trying different orientations...")
                                        alt_orientations = [
                                            (180, 0, 0),   # Standard
                                            (0, 0, 0),     # Upright
                                            (180, 90, 0),  # Rotated
                                            (90, 0, 0),    # Sideways
                                        ]
                                        
                                        for alt_rx, alt_ry, alt_rz in alt_orientations:
                                            if (alt_rx, alt_ry, alt_rz) == (rx, ry, rz):
                                                continue
                                            
                                            print(f"  Trying orientation: RX={alt_rx}°, RY={alt_ry}°, RZ={alt_rz}°")
                                            alt_result = self.dashboard.MovJ(x, y, z, alt_rx, alt_ry, alt_rz, 0, v=int(velocity))
                                            if alt_result and not str(alt_result).startswith("-"):
                                                print(f"✓ Reached target with alternative orientation")
                                                return True
                                
                                print("  Solution:")
                                print("    1. Increase Z height (recommended: 150-250mm)")
                                print("    2. Move robot to a different position first")
                                print("    3. Try adjusting orientation angles")
                                print("    4. Use joint mode for manual control")
                            elif error_code == -1:
                                print("  Error: Robot collision or joint limit exceeded")
                                self.clear_error()
                                print("  Solution: Check robot position and joint limits")
                            
                            return False
                except Exception as e:
                    print(f"✗ Error parsing result: {e}")
                    return False
            
            if coordinate_mode == 0:
                print(f"MovJ → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            else:
                print(f"MovJ → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
            return True
        return False
    
    def move_l(self, x: float, y: float, z: float, rx: float, ry: float, rz: float,
               velocity: float = 50.0, coordinate_mode: int = 0) -> bool:
        """
        Move to position in linear path
        
        Args:
            x, y, z: Position in mm (if coordinate_mode=0) or Joint angle in degrees (if coordinate_mode=1)
            rx, ry, rz: Orientation in degrees (if coordinate_mode=0) or Joint angle in degrees (if coordinate_mode=1)
            velocity: Movement velocity percentage (0-100)
            coordinate_mode: 0 for pose (x,y,z,rx,ry,rz), 1 for joint angles
            
        Returns:
            True if command sent successfully
        """
        if not self.dashboard:
            print("✗ Dashboard not connected")
            return False
        
        # Validate position
        if coordinate_mode == 0:
            radius = (x**2 + y**2)**0.5
            
            # Z 높이에 따른 최대 반경 계산
            max_radius = self._get_max_radius_for_z(z)
            
            if radius > max_radius:
                print(f"⚠️  Warning: Target position may be unreachable")
                print(f"   Radius: {radius:.1f} mm (max for Z={z:.1f}mm: {max_radius:.1f} mm)")
            
            if z < 0 or z > 600:
                print(f"⚠️  Warning: Z coordinate out of safe range: {z:.1f} mm")
            
        result = self.dashboard.MovL(x, y, z, rx, ry, rz, coordinate_mode, v=int(velocity))
        
        if result:
            # Check for errors
            result_str = str(result)
            if result_str.startswith("-"):
                try:
                    error_code = int(result_str.split(',')[0])
                    if error_code < 0:
                        print(f"✗ Linear movement failed with error code: {error_code}")
                        print(f"  Target: ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
                        if error_code == -2:
                            print("  Error: Inverse kinematics solution not found")
                            self.clear_error()
                        elif error_code == -1:
                            print("  Error: Robot collision or joint limit")
                            self.clear_error()
                        return False
                except:
                    pass
            
            if coordinate_mode == 0:
                print(f"MovL → Pose ({x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f})")
            else:
                print(f"MovL → Joint ({x:.1f}°, {y:.1f}°, {z:.1f}°, {rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
            return True
        return False
    
    def set_digital_output(self, index: int, value: bool) -> bool:
        """
        Set digital output (immediate command)

        Args:
            index: DO index (1-16)
            value: True for ON, False for OFF

        Returns:
            True if command sent successfully
        """
        if not self.dashboard:
            print("✗ Dashboard not connected")
            return False

        status = 1 if value else 0
        result = self.dashboard.ToolDOInstant(index, status)

        if result:
            state = "ON" if value else "OFF"
            print(f"DO{index} = {state}")
            return True
        return False

    def get_tool_digital_input(self, index: int):
        """
        Read tool digital input (e.g. end-effector DI 1/2 for vacuum switch).
        E6 팔 끝단 DI 2채널 — PNP 출력 센서(3-wire PNP) 연결 시 0/1 읽기.

        Args:
            index: Tool DI index (1 or 2)

        Returns:
            True if ON(1), False if OFF(0), None if read failed (sensor not connected / parse error)
        """
        if not self.dashboard:
            return None
        try:
            result = self.dashboard.ToolDI(index)
            if result is None or result == "":
                return None
            # 응답 예: "0,{1},ToolDI(1)" 또는 "0,{0},ToolDI(1)" — 중괄호 안 값이 0이면 OFF, 1이면 ON
            m = re.search(r"\{(\d+)\}", str(result))
            if m:
                return int(m.group(1)) == 1
            return None
        except Exception:
            return None
    
    def get_pose(self) -> Optional[List[float]]:
        """
        Get current robot pose
        
        Returns:
            [x, y, z, rx, ry, rz] or None if failed
        """
        if not self.dashboard:
            print("✗ Dashboard not connected")
            return None
            
        result = self.dashboard.GetPose()
        
        if result:
            try:
                # Parse response: typically returns "{x,y,z,rx,ry,rz}"
                import re
                match = re.search(r'\{([^}]+)\}', result)
                if match:
                    values = [float(v.strip()) for v in match.group(1).split(',')]
                    self.current_pose = values
                    return values
            except Exception as e:
                print(f"✗ Failed to parse pose: {e}")
        
        return None
    
    def get_current_pose_from_feedback(self) -> Optional[List[float]]:
        """
        Get current robot pose from feedback data
        
        Returns:
            [x, y, z, rx, ry, rz] or None if failed
        """
        if not self.feed:
            return None
            
        try:
            data = self.feed.feedBackData()
            if data is not None and len(data) > 0:
                tcp_pose = data['ToolVectorActual'][0].tolist()
                self.current_pose = tcp_pose
                return tcp_pose
        except Exception as e:
            print(f"✗ Failed to get pose from feedback: {e}")
        
        return None
    
    def wait_for_motion_complete(self, timeout: float = 30.0) -> bool:
        """
        Wait for robot motion to complete
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if motion completed, False if timeout
        """
        if not self.feed:
            print("✗ Feedback not connected")
            return False
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                data = self.feed.feedBackData()
                if data is not None and len(data) > 0:
                    # RunningStatus: 0 = idle/ready, 1 = running
                    running_status = data['RunningStatus'][0]
                    if running_status == 0:
                        print("✓ Motion complete")
                        return True
            except Exception as e:
                pass
            
            time.sleep(0.1)
        
        print("✗ Motion timeout")
        return False
    
    def home(self) -> bool:
        """
        Move robot to home position using joint mode (safer, avoids IK issues)
        
        Returns:
            True if successful
        """
        print("Moving to home position (joint mode)...")
        # Use joint mode to avoid IK issues - all joints at 0 degrees
        return self.move_j(0, 0, 0, 0, 0, 0, coordinate_mode=1)


# Example usage
if __name__ == "__main__":
    # Create controller
    robot = DobotE6Controller(ip="192.168.5.1")
    
    try:
        # Connect to robot
        if robot.connect():
            print("\n=== Robot connected ===\n")
            
            # Get current pose
            pose = robot.get_pose()
            if pose:
                print(f"Current pose: {pose}")
            
            # Or get from feedback
            pose_fb = robot.get_current_pose_from_feedback()
            if pose_fb:
                print(f"Current pose (from feedback): {pose_fb}")
            
            # Move to home
            robot.home()
            robot.wait_for_motion_complete()
            
            # Example movement
            robot.move_j(300, 100, 200, 180, 0, 0)
            robot.wait_for_motion_complete()
            
            print("\n=== Test complete ===\n")
        else:
            print("Failed to connect to robot")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot.disconnect()
