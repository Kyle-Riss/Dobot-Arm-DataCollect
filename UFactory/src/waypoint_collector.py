#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waypoint-based trajectory collector for π0.5 training data.

Episode flow:
  home (joint, fixed) → wp0 → wp1 → ... → wpN → home
  pick/place waypoints get Cartesian noise each episode.
"""

import json
import os
import random
import time
import threading
from typing import Optional, List, Dict, Any

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), "waypoints.json")

# Safety boundary (must match reduced_tcp_boundary in xarm_controller.py)
_SAFE_X = (100, 750)
_SAFE_Y = (-550, 550)
_SAFE_Z = (-100, 650)


def _clamp_to_safe(x, y, z):
    x = max(_SAFE_X[0], min(_SAFE_X[1], x))
    y = max(_SAFE_Y[0], min(_SAFE_Y[1], y))
    z = max(_SAFE_Z[0], min(_SAFE_Z[1], z))
    return x, y, z


class WaypointCollector:
    """Execute pick-and-place episodes with per-episode Cartesian noise."""

    def __init__(self, robot, gripper):
        self.robot   = robot
        self.gripper = gripper

        self.home_joints: Optional[List[float]] = None
        self.home_speed:  float = 25.0
        self.noise_xy_mm: float = 10.0
        self.noise_z_mm:  float = 3.0
        self.waypoints:   List[Dict[str, Any]] = []

        self._stop_requested = False
        self._running        = False
        self._log_cb         = None   # optional log callback

        self.load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        data = {
            "home_joints":   self.home_joints,
            "home_speed":    self.home_speed,
            "noise_xy_mm":   self.noise_xy_mm,
            "noise_z_mm":    self.noise_z_mm,
            "waypoints":     self.waypoints,
        }
        with open(WAYPOINTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        if not os.path.exists(WAYPOINTS_FILE):
            return
        try:
            with open(WAYPOINTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self.home_joints  = data.get("home_joints")
            self.home_speed   = data.get("home_speed", 25.0)
            self.noise_xy_mm  = data.get("noise_xy_mm", 10.0)
            self.noise_z_mm   = data.get("noise_z_mm", 3.0)
            self.waypoints    = data.get("waypoints", [])
        except Exception as e:
            self._log(f"[WaypointCollector] load error: {e}")

    # ── Capture helpers ───────────────────────────────────────────────────────

    def capture_home(self) -> bool:
        """현재 joint 각도를 home으로 저장."""
        joints = self.robot.get_current_joints_from_feedback()
        if joints is None:
            self._log("capture_home: 관절 읽기 실패")
            return False
        self.home_joints = list(joints)
        self.save()
        self._log(f"Home 저장: {[round(v,2) for v in self.home_joints]}")
        return True

    def capture_waypoint(self, name: str, noise: bool = False,
                         gripper: Optional[str] = None,
                         speed: float = 40.0) -> bool:
        """현재 Cartesian 위치를 waypoint로 저장/업데이트."""
        pose = self.robot.get_current_pose_from_feedback()
        if pose is None:
            self._log(f"capture_waypoint({name}): pose 읽기 실패")
            return False
        wp = {
            "name":    name,
            "type":    "cartesian",
            "values":  [round(v, 4) for v in pose[:6]],
            "speed":   speed,
            "noise":   noise,
            "gripper": gripper,   # None | "grip" | "release"
        }
        # 이름 중복이면 덮어쓰기
        for i, existing in enumerate(self.waypoints):
            if existing["name"] == name:
                self.waypoints[i] = wp
                self.save()
                self._log(f"Waypoint 갱신: {name} → {[round(v,1) for v in pose[:3]]}")
                return True
        self.waypoints.append(wp)
        self.save()
        self._log(f"Waypoint 추가: {name} → {[round(v,1) for v in pose[:3]]}")
        return True

    def delete_waypoint(self, name: str) -> bool:
        before = len(self.waypoints)
        self.waypoints = [w for w in self.waypoints if w["name"] != name]
        if len(self.waypoints) < before:
            self.save()
            self._log(f"Waypoint 삭제: {name}")
            return True
        return False

    def reorder(self, names: List[str]):
        """names 순서대로 waypoints 재정렬."""
        wp_map = {w["name"]: w for w in self.waypoints}
        self.waypoints = [wp_map[n] for n in names if n in wp_map]
        self.save()

    # ── Noise ────────────────────────────────────────────────────────────────

    def _apply_noise(self, values: List[float]) -> List[float]:
        x, y, z, rx, ry, rz = values
        for _ in range(20):
            nx = x + random.uniform(-self.noise_xy_mm, self.noise_xy_mm)
            ny = y + random.uniform(-self.noise_xy_mm, self.noise_xy_mm)
            nz = z + random.uniform(-self.noise_z_mm,  self.noise_z_mm)
            cx, cy, cz = _clamp_to_safe(nx, ny, nz)
            if abs(cx - nx) < 1 and abs(cy - ny) < 1 and abs(cz - nz) < 1:
                return [cx, cy, cz, rx, ry, rz]
        # 안전 범위 밖이면 clamped 값 반환
        return list(_clamp_to_safe(x, y, z)) + [rx, ry, rz]

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _goto_home(self) -> bool:
        if not self.home_joints:
            self._log("Home 미설정 — 먼저 Home 캡처 필요")
            return False
        j = self.home_joints
        self._log(f"→ Home (joint)")
        ok = self.robot.move_j(j[0], j[1], j[2], j[3], j[4], j[5],
                               velocity=self.home_speed, coordinate_mode=1)
        if ok:
            self.robot.wait_for_motion_complete()
        return ok

    def _goto_waypoint(self, wp: Dict, use_noise: bool = True) -> bool:
        values = list(wp["values"])
        if use_noise and wp.get("noise"):
            values = self._apply_noise(values)

        x, y, z, rx, ry, rz = values
        speed = float(wp.get("speed", 40.0))
        self._log(f"→ {wp['name']}  ({x:.1f}, {y:.1f}, {z:.1f})")
        ok = self.robot.move_j(x, y, z, rx, ry, rz,
                               velocity=speed, coordinate_mode=0)
        if ok:
            self.robot.wait_for_motion_complete()

        # 그리퍼 동작
        gact = wp.get("gripper")
        if gact == "grip" and self.gripper:
            self.gripper.grip(wait_time=0.5)
        elif gact == "release" and self.gripper:
            self.gripper.release(wait_time=0.3)

        return ok

    # ── Episode execution ─────────────────────────────────────────────────────

    def run_episode(self, rec_start_cb=None, rec_stop_cb=None) -> bool:
        """
        단일 에피소드 실행.
        rec_start_cb: 홈 도달 후 레코딩 시작 콜백
        rec_stop_cb(success): 완료/실패 후 레코딩 종료 콜백
        """
        if not self.home_joints:
            self._log("Home 미설정")
            return False
        if not self.waypoints:
            self._log("Waypoint 없음")
            return False

        # 1. 홈으로 이동
        if not self._goto_home():
            self._log("Home 이동 실패")
            if rec_stop_cb:
                rec_stop_cb(False)
            return False

        # 2. 레코딩 시작
        if rec_start_cb:
            rec_start_cb()

        # 3. 각 waypoint 이동
        success = True
        for wp in self.waypoints:
            if self._stop_requested:
                self._log("STOP 요청됨")
                success = False
                break
            if not self._goto_waypoint(wp, use_noise=True):
                self._log(f"이동 실패: {wp['name']}")
                success = False
                break

        # 4. 홈 복귀
        if not self._stop_requested:
            self._goto_home()

        # 5. 레코딩 종료
        if rec_stop_cb:
            rec_stop_cb(success)

        return success

    def run_auto(self, n: int, rec_start_cb=None, rec_stop_cb=None,
                 done_cb=None):
        """N 에피소드 자동 수집 (별도 스레드에서 실행)."""
        def _loop():
            self._running = True
            self._stop_requested = False
            self._log(f"Auto 수집 시작: {n}회")
            done = 0
            for i in range(n):
                if self._stop_requested:
                    break
                self._log(f"[{i+1}/{n}] 에피소드 시작")
                ok = self.run_episode(rec_start_cb, rec_stop_cb)
                if ok:
                    done += 1
                time.sleep(0.3)
            self._running = False
            self._log(f"Auto 수집 완료: {done}/{n} 성공")
            if done_cb:
                done_cb(done, n)
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        self._stop_requested = True

    def goto_waypoint_by_name(self, name: str) -> bool:
        """웹 UI에서 특정 waypoint로 이동 (noise 없이)."""
        for wp in self.waypoints:
            if wp["name"] == name:
                return self._goto_waypoint(wp, use_noise=False)
        return False

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        print(f"[WP] {msg}")
        if self._log_cb:
            try:
                self._log_cb(msg)
            except Exception:
                pass

    @property
    def status(self) -> Dict:
        return {
            "running":      self._running,
            "home_set":     self.home_joints is not None,
            "home_joints":  self.home_joints,
            "noise_xy_mm":  self.noise_xy_mm,
            "noise_z_mm":   self.noise_z_mm,
            "waypoints":    self.waypoints,
        }
