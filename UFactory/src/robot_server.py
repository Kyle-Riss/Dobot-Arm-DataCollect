#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI web server for UFactory xArm Pick-Place data collection.
Dual camera: HIKRobot (wrist) + ZED (scene, LEFT view only)
Dashboard: J1-J6, TCP pose, Robot Mode live display

Usage:
    cd ~/dobot-xarm-datacollect/UFactory/src
    python3 robot_server.py

Open: http://<jetson-ip>:8000
"""

import sys
import os
import time
import threading
import asyncio
import shutil
from datetime import datetime
from typing import Optional, Set

# ═══════════════════════════════════════════════════════════════════════════
# PyQt5 Mock — PickPlaceStepWorker(QThread) → threading.Thread 교체
# (pick_place_gui_new import 전 반드시 먼저 선언)
# ═══════════════════════════════════════════════════════════════════════════
import types as _types

class _BoundSignal:
    def __init__(self):
        self._cbs = []
    def connect(self, cb):
        self._cbs.append(cb)
    def emit(self, *args):
        for cb in self._cbs:
            try:
                cb(*args)
            except Exception:
                pass
    def disconnect(self, cb=None):
        self._cbs = [] if cb is None else [c for c in self._cbs if c != cb]

class _SignalDescriptor:
    def __init__(self, *_):
        self._attr = None
    def __set_name__(self, owner, name):
        self._attr = f'_sig_{name}'
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        attr = self._attr or '_sig_unknown'
        if not hasattr(obj, attr):
            object.__setattr__(obj, attr, _BoundSignal())
        return object.__getattribute__(obj, attr)

def _pyqtSignal(*a, **kw):
    return _SignalDescriptor()

class _QThread(threading.Thread):
    def __init__(self, parent=None):
        super().__init__(daemon=True)
    def start(self):
        super().start()
    def isRunning(self):
        return self.is_alive()
    def wait(self, msecs=None):
        self.join(timeout=(msecs / 1000.0) if msecs else None)

class _MockQt:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _MockQt()
    def __getattr__(self, name): return _MockQt()

_qt5     = _types.ModuleType('PyQt5')
_qw      = _types.ModuleType('PyQt5.QtWidgets')
_qc      = _types.ModuleType('PyQt5.QtCore')
_qg      = _types.ModuleType('PyQt5.QtGui')

for _n in ['QApplication','QMainWindow','QWidget','QVBoxLayout','QHBoxLayout',
           'QGroupBox','QGridLayout','QLabel','QLineEdit','QPushButton',
           'QTextEdit','QDoubleSpinBox','QMessageBox','QCheckBox']:
    setattr(_qw, _n, _MockQt)
_qc.QThread    = _QThread
_qc.pyqtSignal = _pyqtSignal
_qc.QTimer     = _MockQt
_qc.Qt         = _MockQt()
for _n in ['QFont', 'QImage', 'QPixmap']:
    setattr(_qg, _n, _MockQt)

sys.modules['PyQt5']             = _qt5
sys.modules['PyQt5.QtWidgets']   = _qw
sys.modules['PyQt5.QtCore']      = _qc
sys.modules['PyQt5.QtGui']       = _qg

# ═══════════════════════════════════════════════════════════════════════════
# 모듈 import
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import cv2

_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)

if not os.environ.get('MVCAM_COMMON_RUNENV'):
    os.environ['MVCAM_COMMON_RUNENV'] = '/opt/MVS/lib'

import pick_place_gui_new as base
from pick_place_gui_random_pose import (
    RandomPosePickPlaceStepWorker,
    generate_random_initial_pose,
    INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ,
)

# 데이터 저장 경로 — 외장 드라이브 마운트 확인 필요
DATA_SAVE_DIR      = "/media/billye6/새 볼륨/UFactory/2CAM-Orange"
DATA_DRIVE_ROOT    = "/media/billye6/새 볼륨"   # 마운트 여부 판단 기준
from xarm_controller import XArmController as DobotE6Controller
from xarm_gripper import XArmGripper
from waypoint_collector import WaypointCollector

_hik_available = False
try:
    from camera_viewer import HikRobotCamera
    _hik_available = True
except Exception as e:
    print(f"[Server] HIK camera unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# ZED 카메라 래퍼 (LEFT 뷰 단일, 640×480 리사이즈)
# ═══════════════════════════════════════════════════════════════════════════
_zed_available = False
try:
    import pyzed.sl as _sl
    _zed_available = True
except Exception as e:
    print(f"[Server] ZED SDK unavailable: {e}")

class ZedCamera:
    """ZED 2i / ZED X — LEFT 뷰 전용 래퍼."""
    def __init__(self):
        if not _zed_available:
            raise RuntimeError("pyzed not installed")
        self.cam   = _sl.Camera()
        self._mat  = _sl.Mat()
        self._rt   = _sl.RuntimeParameters()
        self.initialized = False

    def init_camera(self) -> bool:
        params = _sl.InitParameters()
        params.camera_resolution = _sl.RESOLUTION.HD1080
        params.camera_fps        = 30
        params.depth_mode        = _sl.DEPTH_MODE.NONE   # 깊이 불필요
        err = self.cam.open(params)
        if err != _sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Open failed: {err}")
            return False
        self.initialized = True
        print("[ZED] Camera initialized (HD720, LEFT view)")
        return True

    def get_frame(self):
        """(ok, RGB ndarray 640×480) 반환."""
        if not self.initialized:
            return False, None
        err = self.cam.grab(self._rt)
        if err != _sl.ERROR_CODE.SUCCESS:
            return False, None
        self.cam.retrieve_image(self._mat, _sl.VIEW.LEFT)
        data = self._mat.get_data()          # (H, W, 4) BGRA
        bgr  = data[:, :, :3]               # BGR
        bgr  = cv2.resize(bgr, (640, 480))
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return True, rgb

    def cleanup(self):
        if self.initialized:
            self.cam.close()
            self.initialized = False

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ═══════════════════════════════════════════════════════════════════════════
# ROS2 레코더 (rclpy 미설치 시 graceful 비활성화)
# ═══════════════════════════════════════════════════════════════════════════
import ros2_recorder as _ros2
_ros2_ok: bool = False          # start() 성공 후 True 로 설정

# ═══════════════════════════════════════════════════════════════════════════
# 서버 상태
# ═══════════════════════════════════════════════════════════════════════════
_state = {
    "robot":          None,
    "gripper":        None,
    "wc":             None,   # WaypointCollector
    "camera_hik":     None,
    "camera_zed":     None,
    "worker":         None,
    "recording":      False,
    "recorded_data":  [],
    "record_save_dir":None,
    "record_frame_count": 0,
    "vacuum_pick":    0.0,
    "vacuum_place":   0.0,
    "episode_meta":   {},
    "auto_target":    0,
    "auto_done":      0,
    "pick_section":   "A",
    "last_place_x":   None,
    "last_place_y":   None,
    # v20+ 앵커 페어 선택
    "anchor_a_idx":       None,   # None=random, 0/1/2 = A3/A4/A5 고정
    "anchor_b_idx":       None,   # None=random, 0/1/2 = B1/B2/B3 고정
    "force_pick_section": None,   # None=auto-alternate, "A"/"B" = 방향 고정
    "pair_counts":        {},     # "AB_a_b" / "BA_b_a" → 성공 에피소드 수
}


_state_lock  = threading.Lock()
_ORIG_A_ANCHORS: list = []   # base.A_ANCHORS 원본 (startup 시 캡처)
_ORIG_B_ANCHORS: list = []
_ws_clients: Set[WebSocket] = set()
_log_queue: asyncio.Queue   = None
_main_loop: asyncio.AbstractEventLoop = None

FIXED_INIT = (89.3715, -378.5400, 250.0000, -179.5275, -2.4369, 2.3663)

ROBOT_MODE_LABELS = {
    5:"STANDBY", 7:"RUNNING", 9:"ERROR",
    10:"PAUSED", 11:"JOG"
}

# ─── 프레임 버퍼 (MJPEG + 레코딩 공용) ────────────────────────────────────
_buf_hik_jpg: Optional[bytes]       = None   # MJPEG용 JPEG 바이트
_buf_zed_jpg: Optional[bytes]       = None
_buf_hik_np:  Optional[np.ndarray]  = None   # 레코딩용 BGR numpy
_buf_zed_np:  Optional[np.ndarray]  = None
_buf_lock     = threading.Lock()

_cam_hik_thread: Optional[threading.Thread] = None
_cam_zed_thread: Optional[threading.Thread] = None
_cam_hik_running = False
_cam_zed_running = False

_robot_pub_running = False   # _robot_pub_loop 제어 플래그

# ═══════════════════════════════════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _main_loop and _log_queue:
        try:
            _main_loop.call_soon_threadsafe(_log_queue.put_nowait, line)
        except Exception:
            pass

_SAFE_INIT_FALLBACK_XYZ = (89.3715, -378.5400, 250.0)  # 실측 검증된 안전 대기 위치

def _set_random_init_pose(robot):
    rx, ry, rz = INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ
    cx, cy, cz = _SAFE_INIT_FALLBACK_XYZ
    ok = False
    for _ in range(30):
        tx, ty, tz, *_ = generate_random_initial_pose()
        if robot and robot.connected:
            ok, _ = robot.check_ik_solution(tx, ty, tz, rx, ry, rz)
        else:
            ok = True
        if ok:
            cx, cy, cz = tx, ty, tz
            break
    base.INIT_X = cx;  base.INIT_Y = cy;  base.INIT_Z = cz
    base.INIT_RX = rx; base.INIT_RY = ry; base.INIT_RZ = rz
    _log(f"[RandomPose] INIT X={cx:.1f} Y={cy:.1f} Z={cz:.1f} (IK={ok})")

def _get_next_folder(base_dir: str) -> int:
    if not os.path.exists(base_dir):
        return 1
    nums = set(int(d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit())
    n = 1
    while n in nums:
        n += 1
    return n

# ── 앵커 페어 카운팅 헬퍼 ──────────────────────────────────────────────────
_A_ANCHOR_REF = [(217.75, -405.65), (220.21, -368.72), (221.63, -318.39)]
_B_ANCHOR_REF = [(30.0, -340.0),   (30.0, -377.0),    (30.0, -415.0)]

def _anchor_to_idx(anchor, ref):
    if not anchor or not isinstance(anchor, (list, tuple)):
        return None
    for i, (ax, ay) in enumerate(ref):
        if abs(anchor[0] - ax) < 1.0 and abs(anchor[1] - ay) < 1.0:
            return i
    return None

def _update_pair_count(meta):
    pick_section = meta.get("pick_section")
    pick_anchor  = meta.get("pick_anchor")
    place_anchor = meta.get("place_anchor")
    if not pick_section or pick_anchor is None or place_anchor is None:
        return
    if pick_section == "A":
        a_idx = _anchor_to_idx(pick_anchor,  _A_ANCHOR_REF)
        b_idx = _anchor_to_idx(place_anchor, _B_ANCHOR_REF)
        if a_idx is None or b_idx is None:
            return
        key = f"AB_{a_idx}_{b_idx}"
    else:
        b_idx = _anchor_to_idx(pick_anchor,  _B_ANCHOR_REF)
        a_idx = _anchor_to_idx(place_anchor, _A_ANCHOR_REF)
        if b_idx is None or a_idx is None:
            return
        key = f"BA_{b_idx}_{a_idx}"
    with _state_lock:
        counts = _state.setdefault("pair_counts", {})
        counts[key] = counts.get(key, 0) + 1

# ═══════════════════════════════════════════════════════════════════════════
# 카메라 그랩 루프 (MJPEG 버퍼 + 레코딩 numpy 버퍼 동시 갱신)
# ═══════════════════════════════════════════════════════════════════════════

def _hik_grab_loop():
    global _buf_hik_jpg, _buf_hik_np, _cam_hik_running
    cam = _state["camera_hik"]
    while _cam_hik_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB
        if ret and frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _buf_lock:
                _buf_hik_jpg = enc.tobytes()
                _buf_hik_np  = bgr
            if _ros2_ok:
                _ros2.publish_hik(bgr)   # 캡처 직후 타임스탬프로 퍼블리시
        else:
            time.sleep(0.02)

def _zed_grab_loop():
    global _buf_zed_jpg, _buf_zed_np, _cam_zed_running
    cam = _state["camera_zed"]
    while _cam_zed_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB
        if ret and frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _buf_lock:
                _buf_zed_jpg = enc.tobytes()
                _buf_zed_np  = bgr
            if _ros2_ok:
                _ros2.publish_zed(bgr)   # 캡처 직후 타임스탬프로 퍼블리시
        else:
            time.sleep(0.02)

def _robot_pub_loop():
    """로봇 상태를 ~50Hz 로 ROS2 에 퍼블리시. startup 에서 데몬 스레드로 시작."""
    global _robot_pub_running
    while _robot_pub_running:
        if _ros2_ok:
            robot   = _state["robot"]
            gripper = _state["gripper"]
            if robot and robot.connected:
                try:
                    feed = robot.feed.feedBackData()
                    if feed is not None and len(feed) > 0:
                        joints     = feed['QActual'][0].tolist()
                        tcp_pose   = feed['ToolVectorActual'][0].tolist()
                        robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
                        gripper_on = 1 if (gripper and gripper.is_gripping) else 0
                        _ros2.publish_robot(joints, tcp_pose, gripper_on, robot_mode)
                except Exception:
                    pass
        time.sleep(0.02)   # ~50 Hz


def _mjpeg_gen(buf_getter):
    """공통 MJPEG 제너레이터."""
    placeholder = None
    while True:
        with _buf_lock:
            frame = buf_getter()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            if placeholder is None:
                blank = np.full((240, 320, 3), 60, dtype=np.uint8)
                cv2.putText(blank, "No Camera", (55, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
                _, enc = cv2.imencode('.jpg', blank)
                placeholder = enc.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
        time.sleep(0.04)

# ═══════════════════════════════════════════════════════════════════════════
# 20Hz 레코딩 (threading 기반, QTimer 대체)
# ═══════════════════════════════════════════════════════════════════════════

def _start_recording():
    if _state["recording"]:
        return

    # ── 외장 드라이브 마운트 확인 ──────────────────────────────────────────
    if not os.path.isdir(DATA_DRIVE_ROOT):
        _log(f"[ERROR] External drive not mounted: {DATA_DRIVE_ROOT}")
        _log("[ERROR] Data collection aborted — please connect the drive and retry")
        # 진행 중인 worker 도 중단
        w = _state.get("worker")
        if w:
            w._stop_requested = True
        _state.update(auto_target=0, auto_done=0)
        return
    # ────────────────────────────────────────────────────────────────────────

    base = DATA_SAVE_DIR
    n = _get_next_folder(base)
    save_dir = os.path.join(base, str(n))
    has_zed = bool(_state["camera_zed"] and _state["camera_zed"].initialized)
    try:
        os.makedirs(os.path.join(save_dir, "images", "hik"), exist_ok=True)
        if has_zed:
            os.makedirs(os.path.join(save_dir, "images", "zed"), exist_ok=True)
    except OSError as e:
        _log(f"[ERROR] Cannot create save directory: {e}")
        _log("[ERROR] Data collection aborted — check drive permissions")
        w = _state.get("worker")
        if w:
            w._stop_requested = True
        _state.update(auto_target=0, auto_done=0)
        return

    _state.update(recording=True, recorded_data=[], record_save_dir=save_dir,
                  record_frame_count=0)
    mode_str = "20Hz+ROS2pub" if _ros2_ok else "20Hz-legacy"
    _log(f"Recording started → {save_dir} (ZED={'ON' if has_zed else 'OFF'}, mode={mode_str})")
    # ROS2 는 퍼블리시 전용 — start_recording 호출 안 함 (_on_sync 가 디스크 쓰지 않도록)
    threading.Thread(target=_record_loop, daemon=True).start()

def _record_loop():
    while _state["recording"]:
        _record_tick()
        time.sleep(0.05)

def _record_tick():
    robot    = _state["robot"]
    gripper  = _state["gripper"]
    save_dir = _state["record_save_dir"]
    if not robot or not robot.connected or not save_dir:
        return
    try:
        feed = robot.feed.feedBackData()
        if feed is None or len(feed) == 0:
            return
        joints     = feed['QActual'][0].tolist()
        tcp_pose   = feed['ToolVectorActual'][0].tolist()
        robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
        gripper_on = 1 if (gripper and gripper.is_gripping) else 0
        fc = _state["record_frame_count"]
        fname = f"frame_{fc:06d}.jpg"

        # HIK 이미지 저장
        with _buf_lock:
            hik_np = _buf_hik_np.copy() if _buf_hik_np is not None else None
            zed_np = _buf_zed_np.copy() if _buf_zed_np is not None else None

        # HIK: 320×240 리사이즈 후 (x=60, y=16) 기준 224×224 크롭
        hik_path = os.path.join(save_dir, "images", "hik", fname)
        if hik_np is not None:
            hik_320  = cv2.resize(hik_np, (320, 240))
            hik_save = hik_320[16:240, 55:279]        # y:16~240, x:55~279 → 224×224
        else:
            hik_save = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(hik_path, hik_save)

        # ZED: 320×240 리사이즈
        has_zed = bool(_state["camera_zed"] and _state["camera_zed"].initialized)
        if has_zed:
            zed_path = os.path.join(save_dir, "images", "zed", fname)
            if zed_np is not None:
                zed_crop = zed_np[120:480, 150:510]         # (150,120) 시작 360×360 크롭
                zed_save = cv2.resize(zed_crop, (224, 224))
            else:
                zed_save = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(zed_path, zed_save)

        record = {
            'frame_id':       fc,
            'timestamp':      time.time(),
            'image_path_hik': f"hik/{fname}",
            'image_path_zed': f"zed/{fname}" if has_zed else "",
            'joint_angles':   joints,
            'tcp_pose':       tcp_pose,
            'gripper_tooldo1':gripper_on,
            'gripper_tooldo2':0,
            'robot_mode':     robot_mode,
        }
        _state["recorded_data"].append(record)
        _state["record_frame_count"] += 1
    except Exception as e:
        print(f"[record_tick] {e}")

def _stop_and_save(success: bool):
    _state["recording"] = False
    time.sleep(0.07)
    save_dir = _state["record_save_dir"]

    # ROS2 는 토픽 퍼블리시 전용 — 저장 데이터는 항상 legacy _record_tick 기준
    data = _state["recorded_data"]

    if not success:
        _log("Episode failed → saving as failed episode (no discard)")

    if not save_dir or not data:
        _log(f"No data recorded (dir={save_dir}, n={len(data) if data else 0})")
        return
    try:
        has_zed = bool(data[0].get('image_path_zed'))
        # CSV
        with open(os.path.join(save_dir, "robot_data.csv"), 'w', newline='') as f:
            f.write("frame_id,timestamp,image_path_hik")
            if has_zed:
                f.write(",image_path_zed")
            f.write(",j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz"
                    ",gripper_tooldo1,gripper_tooldo2,robot_mode\n")
            for r in data:
                f.write(f"{r['frame_id']},{r['timestamp']},{r['image_path_hik']}")
                if has_zed:
                    f.write(f",{r['image_path_zed']}")
                f.write(',' + ','.join(map(str, r['joint_angles'])))
                f.write(',' + ','.join(map(str, r['tcp_pose'])))
                f.write(f",{r['gripper_tooldo1']},{r['gripper_tooldo2']},{r['robot_mode']}\n")
        # NPY
        np.save(os.path.join(save_dir, "dataset.npy"), data)
        # episode_meta.json
        import json as _json
        folder_num = os.path.basename(save_dir)
        n_frames = len(data)
        if n_frames >= 2:
            actual_fps = round((n_frames - 1) / (data[-1]['timestamp'] - data[0]['timestamp']), 3)
        else:
            actual_fps = 15.0
        ep_meta = dict(_state.get("episode_meta") or {})
        ep_meta.update({
            "folder": folder_num,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": n_frames,
            "record_rate_hz": actual_fps,
            "cameras": "HIK+ZED" if has_zed else "HIK",
            "success": bool(success),
            "vacuum_pick_duration_s": round(_state['vacuum_pick'], 3),
            "vacuum_place_duration_s": round(_state['vacuum_place'], 3),
        })
        events_raw = ep_meta.pop("events", [])
        with open(os.path.join(save_dir, "episode_meta.json"), 'w', encoding='utf-8') as f:
            _json.dump(ep_meta, f, ensure_ascii=False, indent=2)
        # episode_events.csv
        if events_raw and data:
            ts_list = [(r['frame_id'], r['timestamp']) for r in data]
            with open(os.path.join(save_dir, "episode_events.csv"), 'w', newline='') as f:
                f.write("event,frame_id,timestamp\n")
                for ev_name, ev_ts in events_raw:
                    closest_fid = min(ts_list, key=lambda t: abs(t[1] - ev_ts))[0]
                    f.write(f"{ev_name},{closest_fid},{ev_ts:.6f}\n")
        # metadata.txt (호환성 유지)
        with open(os.path.join(save_dir, "metadata.txt"), 'w', encoding='utf-8') as f:
            f.write("VLA Dataset - Pick-Place Step\n" + "="*50 + "\n\n")
            f.write(f"Folder: {folder_num}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {len(data)}\n")
            f.write(f"Record Rate: {actual_fps}Hz\n")
            f.write(f"Cameras: HIK{'+ ZED (LEFT)' if has_zed else ' only'}\n")
            f.write(f"Step Success: {success}\n")
            f.write(f"VacuumCommandPickDuration_s: {_state['vacuum_pick']:.3f}\n")
            f.write(f"VacuumCommandPlaceDuration_s: {_state['vacuum_place']:.3f}\n")
            f.write(f"Prompt: {ep_meta.get('prompt', '')}\n")
            f.write(f"Object: {ep_meta.get('object_label', '')} | Marker: {ep_meta.get('object_marker', '')}\n")
            f.write(f"Workspace: {ep_meta.get('workspace', '')} | {ep_meta.get('source_zone', '')} → {ep_meta.get('target_zone', '')}\n")
        _state["episode_meta"] = {}
        _log(f"Saved {len(data)} frames → {save_dir}")
    except Exception as e:
        _log(f"Save error: {e}")
    finally:
        _state.update(recorded_data=[], record_save_dir=None)

# ═══════════════════════════════════════════════════════════════════════════
# Worker 콜백
# ═══════════════════════════════════════════════════════════════════════════

def _on_log(msg):    _log(msg)
def _on_vacuum(ph, pl): _state["vacuum_pick"] = ph; _state["vacuum_place"] = pl
def _on_rec_begin():  _start_recording()
def _on_episode_meta(meta): _state["episode_meta"] = meta

def _on_finished(success: bool):
    w = _state.get("worker")
    manual_stop = bool(w and getattr(w, "_stop_requested", False))
    _qc_data = list(_state.get("recorded_data") or [])
    _qc_dir  = _state.get("record_save_dir") or ""

    # 앵커 카운팅을 위해 episode_meta를 _stop_and_save 전에 캡처
    ep_meta_snap = dict(_state.get("episode_meta") or {})

    if manual_stop:
        # 수동 STOP: 기록 중단 후 폴더 삭제
        _state["recording"] = False
        time.sleep(0.07)
        _state.update(recorded_data=[], record_save_dir=None, episode_meta={})
        if _qc_dir and os.path.isdir(_qc_dir):
            shutil.rmtree(_qc_dir)
            _log(f"STOP: 미완성 에피소드 삭제 → {_qc_dir}")
        else:
            _log("STOP: 저장 중인 데이터 없음")
        _state.update(auto_target=0, auto_done=0)
        _log("⚠ 자동 수집 중단됨 (수동 STOP)")
        return

    _stop_and_save(success)
    if success:
        _update_pair_count(ep_meta_snap)
    if w and hasattr(w, 'place_x') and w.place_x is not None:
        _state["last_place_x"] = w.place_x
        _state["last_place_y"] = w.place_y
        # 방향 고정 모드에서는 자동 교번하지 않음
        if not _state.get("force_pick_section"):
            _state["pick_section"] = "B" if _state["pick_section"] == "A" else "A"
        _log(f"Place ({w.place_x:.1f},{w.place_y:.1f}) / next: {_state['pick_section']}")

    if _state["auto_target"] > 0:
        if not success:
            _log("⚠ 실패/폐기 에피소드 발생 — 다음 에피소드 자동 진행")
            threading.Timer(0.3, _run_step).start()
            return
        _state["auto_done"] += 1
        _log(f"Auto {_state['auto_done']}/{_state['auto_target']}")
        if _state["auto_done"] >= _state["auto_target"]:
            _state.update(auto_target=0, auto_done=0)
            _log("Auto collect complete")
        else:
            threading.Timer(0.3, _run_step).start()
    else:
        _log("Step complete")

DJ3_DISCARD_THRESHOLD = 0.4  # |dj3_mean|이 이 값 미만이면 느린 접근
DJ3_SIGN_EPS = 0.05          # 부호 판정 시 노이즈 무시 임계값

def _check_episode_quality_v10(data=None, save_dir=None) -> bool:
    """v10 수집 후 approach 품질 체크.

    경고 조건 (OR):
      - j3_dip : approach 중 dj3 방향 반전(부호 전환) 발생
      - dj3_low: approach |dj3_mean| < 0.4°/f

    Returns:
        True  — 에피소드 유지 (폐기하지 않고 경고만 기록)
    """
    if data is None:
        data = _state.get("recorded_data") or []
    if not save_dir:
        save_dir = _state.get("record_save_dir") or ""
    ep_num = os.path.basename(save_dir) if save_dir else "?"

    if len(data) < 10:
        return True

    # mode=9 (IK 에러) 프레임 → 데이터 오염 → 즉시 폐기
    mode9_count = sum(1 for r in data if r.get("robot_mode") == 9)
    if mode9_count > 0:
        _log(f"⚠ [ep{ep_num}] mode=9 {mode9_count}프레임 감지 — 폐기하지 않고 유지")
        return True

    # approach: 첫 mode=7 시작 ~ gripper_on 프레임 (전체 pick 접근 구간)
    ap_start = next((i for i, r in enumerate(data) if r.get("robot_mode") == 7), None)
    if ap_start is None:
        return True
    grip_frame = next(
        (i for i in range(ap_start, len(data)) if data[i].get("gripper_tooldo1") == 1),
        None,
    )
    if grip_frame is not None:
        j3_vals = [float(data[i]["joint_angles"][2]) for i in range(ap_start, grip_frame + 1)
                   if "joint_angles" in data[i]]
    else:
        # fallback: 첫 mode=7 블록만
        j3_vals = []
        in_ap = False
        for r in data:
            m = r.get("robot_mode")
            if not in_ap and m == 7:
                in_ap = True
            if in_ap:
                if m == 7 and "joint_angles" in r:
                    j3_vals.append(float(r["joint_angles"][2]))
                else:
                    break
    if len(j3_vals) < 5:
        return True

    dj3_list = [j3_vals[i+1] - j3_vals[i] for i in range(len(j3_vals)-1)]
    dj3_mean = sum(dj3_list) / len(dj3_list) if dj3_list else 0.0
    dj3_low = abs(dj3_mean) < DJ3_DISCARD_THRESHOLD

    # 방향 반전(진동) 체크: 노이즈 구간은 제외하고 부호 전환 횟수 계산
    nonzero_signs = []
    for d in dj3_list:
        if d > DJ3_SIGN_EPS:
            nonzero_signs.append(1)
        elif d < -DJ3_SIGN_EPS:
            nonzero_signs.append(-1)
    sign_flip_count = sum(
        1 for i in range(1, len(nonzero_signs))
        if nonzero_signs[i] != nonzero_signs[i - 1]
    )
    j3_dip = sign_flip_count > 0

    if j3_dip or dj3_low:
        reason = []
        if j3_dip:
            reason.append(f"j3 방향반전 {sign_flip_count}회")
        if dj3_low:
            reason.append(f"|dj3|={abs(dj3_mean):.3f}<{DJ3_DISCARD_THRESHOLD}")
        _log(f"⚠ [ep{ep_num}] 품질 경고: {', '.join(reason)} — 폐기하지 않고 유지")
        return True
    else:
        _log(f"✓ [ep{ep_num}] approach dj3_mean={dj3_mean:+.3f}°/f — 통과")
        return True

def _run_step():
    robot   = _state["robot"]
    gripper = _state["gripper"]
    if not robot or not robot.connected or not gripper:
        _log("Robot not connected"); return
    if _state["worker"] and _state["worker"].isRunning():
        _log("Worker already running"); return

    # ── 방향/앵커 오버라이드 적용 ────────────────────────────────────────────
    forced_dir = _state.get("force_pick_section")
    if forced_dir in ("A", "B"):
        _state["pick_section"] = forced_dir

    if _ORIG_A_ANCHORS:
        a_idx = _state.get("anchor_a_idx")
        b_idx = _state.get("anchor_b_idx")
        base.A_ANCHORS = [_ORIG_A_ANCHORS[a_idx]] if (a_idx is not None and 0 <= a_idx < 3) else list(_ORIG_A_ANCHORS)
        base.B_ANCHORS = [_ORIG_B_ANCHORS[b_idx]] if (b_idx is not None and 0 <= b_idx < 3) else list(_ORIG_B_ANCHORS)
    # ────────────────────────────────────────────────────────────────────────

    _set_random_init_pose(robot)

    cam_hik = _state["camera_hik"] if (_state["camera_hik"] and
                                        _state["camera_hik"].initialized) else None
    worker = RandomPosePickPlaceStepWorker(
        robot, gripper,
        pick_section   = _state["pick_section"],
        pick_x         = _state["last_place_x"],
        pick_y         = _state["last_place_y"],
        camera         = cam_hik,
        fallback_initial_pose = FIXED_INIT,
        velocity_scale = 1.0,
    )
    worker.log_signal.connect(_on_log)
    worker.finished.connect(_on_finished)
    worker.episode_vacuum_durations.connect(_on_vacuum)
    worker.episode_meta_ready.connect(_on_episode_meta)
    worker.recording_begin_at_initial.connect(_on_rec_begin)
    _state["worker"] = worker
    worker.start()

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="UFactory xArm Server")

@app.on_event("startup")
async def _startup():
    global _log_queue, _main_loop, _ros2_ok, _robot_pub_running, _ORIG_A_ANCHORS, _ORIG_B_ANCHORS
    _log_queue = asyncio.Queue()
    _main_loop = asyncio.get_event_loop()
    asyncio.create_task(_broadcast())

    # v20+ 앵커 오버라이드용 원본 리스트 캡처
    if hasattr(base, 'A_ANCHORS') and hasattr(base, 'B_ANCHORS'):
        _ORIG_A_ANCHORS = list(base.A_ANCHORS)
        _ORIG_B_ANCHORS = list(base.B_ANCHORS)
        _log(f"Anchor refs captured: A={_ORIG_A_ANCHORS} B={_ORIG_B_ANCHORS}")

    # ROS2 레코더 초기화 (rclpy 미설치 시 False 반환 → fallback 모드)
    _ros2_ok = _ros2.start()
    if _ros2_ok:
        _robot_pub_running = True
        threading.Thread(target=_robot_pub_loop, daemon=True).start()
        _log("ROS2 recorder ready (sync mode)")
    else:
        _log("ROS2 unavailable — legacy recording mode")

    _log("Server ready")

async def _broadcast():
    while True:
        msg = await _log_queue.get()
        dead = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)

# ─── 연결 ────────────────────────────────────────────────────────────────

@app.post("/connect")
def connect(ip: str = "192.168.1.225"):
    if _state["robot"] and _state["robot"].connected:
        return {"ok": True, "msg": "Already connected"}
    try:
        robot = DobotE6Controller(ip=ip)
        if not robot.connect():
            return JSONResponse({"ok": False, "msg": "Connect failed — robot unreachable"}, status_code=500)
        _state["robot"]   = robot
        _state["gripper"] = XArmGripper(robot._arm)
        wc = WaypointCollector(robot, _state["gripper"])
        wc._log_cb = _log
        _state["wc"] = wc
        _log(f"Robot connected @ {ip}")
        return {"ok": True, "msg": f"Connected @ {ip}"}
    except Exception as e:
        _log(f"Connect error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/enable")
def enable_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.dashboard.EnableRobot()
        _log("Robot enabled")
        return {"ok": True, "msg": "Robot enabled"}
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/disable")
def disable_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.dashboard.DisableRobot()
        _log("Robot disabled")
        return {"ok": True, "msg": "Robot disabled"}
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/clear-alarm")
def clear_alarm():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        result = robot.dashboard.ClearError()
        _log(f"ClearError → {result}")
        return {"ok": True, "msg": f"Alarm cleared ({result})"}
    except Exception as e:
        _log(f"ClearError failed: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/resume")
def resume_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.resume_robot()
        robot.clear_error()
        robot.enable_robot(sleep_after=0.1)
        _log("Resume → ClearError + EnableRobot 완료")
        return {"ok": True}
    except Exception as e:
        _log(f"Resume failed: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/disconnect")
def disconnect():
    if _state["robot"]:
        try:
            _state["robot"].disconnect()
        except Exception:
            pass
        _state["robot"] = _state["gripper"] = None
        _log("Robot disconnected")
    return {"ok": True}

@app.get("/status")
def status():
    robot     = _state["robot"]
    connected = bool(robot and robot.connected)
    pose = joints = None
    robot_mode = 0
    if connected:
        try:
            feed = robot.feed.feedBackData()
            if feed is not None and len(feed) > 0:
                joints     = [round(float(v), 3) for v in feed['QActual'][0]]
                pose       = [round(float(v), 3) for v in feed['ToolVectorActual'][0]]
                robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
        except Exception:
            pass
    gripper     = _state["gripper"]
    gripper_pos = None
    if gripper:
        try:
            gripper_pos = gripper.get_position()
        except Exception:
            pass
    return {
        "connected":      connected,
        "pose":           pose,
        "joints":         joints,
        "robot_mode":     robot_mode,
        "robot_mode_str": ROBOT_MODE_LABELS.get(robot_mode, str(robot_mode)),
        "cam_hik":        bool(_state["camera_hik"] and _state["camera_hik"].initialized),
        "cam_zed":        bool(_state["camera_zed"] and _state["camera_zed"].initialized),
        "recording":      _state["recording"],
        "frames":         _state["record_frame_count"],
        "auto_target":    _state["auto_target"],
        "auto_done":      _state["auto_done"],
        "worker_running": bool(_state["worker"] and _state["worker"].isRunning()),
        "gripper_pos":    round(gripper_pos, 1) if gripper_pos is not None else None,
    }

@app.post("/anchor")
def set_anchor(direction: str = "auto", a_idx: int = -1, b_idx: int = -1):
    if direction not in ("A", "B", "auto"):
        return JSONResponse({"ok": False, "msg": "direction must be A, B, or auto"}, status_code=400)
    _state["force_pick_section"] = None if direction == "auto" else direction
    _state["anchor_a_idx"] = None if a_idx < 0 else a_idx
    _state["anchor_b_idx"] = None if b_idx < 0 else b_idx
    A_LABELS = ["A3", "A4", "A5"]
    B_LABELS = ["B1", "B2", "B3"]
    a_str = A_LABELS[a_idx] if 0 <= a_idx < 3 else "랜덤"
    b_str = B_LABELS[b_idx] if 0 <= b_idx < 3 else "랜덤"
    _log(f"앵커 설정: 방향={direction} | A={a_str} | B={b_str}")
    return {"ok": True}

@app.get("/anchor-status")
def anchor_status():
    return {
        "force_pick_section": _state.get("force_pick_section"),
        "anchor_a_idx":       _state.get("anchor_a_idx"),
        "anchor_b_idx":       _state.get("anchor_b_idx"),
        "pick_section":       _state.get("pick_section"),
        "pair_counts":        _state.get("pair_counts", {}),
    }

# ─── 로봇 제어 ────────────────────────────────────────────────────────────

@app.post("/home")
def go_home():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    def _do():
        ok = robot.move_j(300, 0, 400, 180, 0, 0, coordinate_mode=0, use_waypoint=False)
        if ok: robot.wait_for_motion_complete()
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/move")
def move(x: float, y: float, z: float,
         rx: float = 180.0, ry: float = 0.0, rz: float = 0.0,
         velocity: float = 30.0):
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    def _do():
        ok = robot.move_j(x, y, z, rx, ry, rz, coordinate_mode=0,
                          velocity=velocity, use_waypoint=False)
        if ok: robot.wait_for_motion_complete()
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

_jog_lock = threading.Lock()
_jog_axis_active: list = [None]   # [0] = currently jogging axis or None
_jog_stop_time: list  = [0.0]     # [0] = last stop timestamp

@app.post("/jog/start")
def jog_start(axis: str, speed: int = 20):
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    speed = max(1, min(100, speed))
    def _do():
        with _jog_lock:
            # cooldown: wait until 200ms after last stop
            elapsed = time.time() - _jog_stop_time[0]
            if elapsed < 0.20:
                time.sleep(0.20 - elapsed)
            try:
                robot.dashboard.EnableRobot()
            except Exception:
                pass
            try:
                robot.dashboard.SpeedFactor(speed)
            except Exception:
                pass
            for attempt in range(2):
                try:
                    if axis.startswith('J'):
                        result = robot.dashboard.MoveJog(axis)
                    else:
                        result = robot.dashboard.MoveJog(axis, coordtype=1, user=0, tool=0)
                    result_str = str(result).strip() if result else ""
                    first = result_str.split(',')[0].strip() if result_str else ""
                    if first and first != "0":
                        if attempt == 0:
                            time.sleep(0.15)
                            continue  # retry once
                        _log(f"[jog] {axis} error: {result_str}")
                    else:
                        _jog_axis_active[0] = axis
                        _log(f"[jog] {axis} start (speed={speed}%)")
                    break
                except Exception as e:
                    _log(f"[jog] {axis} exception: {e}")
                    break
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/jog/stop")
def jog_stop():
    robot = _state["robot"]
    if robot and robot.connected:
        def _stop():
            with _jog_lock:
                try:
                    robot.dashboard.MoveJog("")
                except Exception as e:
                    _log(f"[jog] stop error: {e}")
                try:
                    robot.dashboard.SpeedFactor(100)
                except Exception:
                    pass
                was = _jog_axis_active[0]
                _jog_axis_active[0] = None
                _jog_stop_time[0] = time.time()
                if was:
                    _log(f"[jog] {was} stopped")
        threading.Thread(target=_stop, daemon=True).start()
    return {"ok": True}

@app.get("/pose")
def get_pose():
    """현재 TCP 좌표 반환 (조그 후 위치 확인용)."""
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        pose = robot.get_current_pose_from_feedback()
        if pose and len(pose) >= 6:
            return {"ok": True, "x": round(pose[0],3), "y": round(pose[1],3), "z": round(pose[2],3),
                    "rx": round(pose[3],3), "ry": round(pose[4],3), "rz": round(pose[5],3)}
        return JSONResponse({"ok": False, "msg": "No pose data"}, status_code=503)
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/gripper/grip")
def grip():
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    threading.Thread(target=g.grip, daemon=True).start()
    return {"ok": True}

@app.post("/gripper/release")
def release():
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    threading.Thread(target=g.release, daemon=True).start()
    return {"ok": True}

@app.post("/gripper/set")
def gripper_set(pos: int, speed: int = 60, force: int = 50):
    """슬라이더 Set 버튼 — 지정 위치로 이동."""
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    pos = max(0, min(85, pos))
    threading.Thread(target=g.set_position, args=(pos,), kwargs={"speed": speed, "force": force}, daemon=True).start()
    return {"ok": True, "msg": f"Gripper → {pos}mm"}

@app.post("/gripper/jog")
def gripper_jog(direction: str = "close"):
    """V=close / C=open: 목표까지 천천히 이동 시작 (wait=False)."""
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    target = 0 if direction == "close" else 85
    threading.Thread(target=g.set_position, args=(target,),
                     kwargs={"speed": 20, "force": 50, "wait": False}, daemon=True).start()
    return {"ok": True}

@app.post("/gripper/jog/stop")
def gripper_jog_stop():
    """키 떼는 순간 현재 위치에서 정지."""
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    def _hold():
        pos = g.get_position()
        if pos is not None:
            g.set_position(int(round(pos)), speed=100, force=50)
    threading.Thread(target=_hold, daemon=True).start()
    return {"ok": True}

# ─── Manual Recording ────────────────────────────────────────────────────────

@app.post("/record/start")
def record_start():
    if _state["recording"]:
        return JSONResponse({"ok": False, "msg": "Already recording"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _start_recording()
    _log("Manual recording started")
    return {"ok": True, "msg": "Recording started"}

@app.post("/record/stop")
def record_stop(success: bool = True):
    if not _state["recording"]:
        return JSONResponse({"ok": False, "msg": "Not recording"}, status_code=400)
    _stop_and_save(success)
    msg = f"Saved ({_state.get('record_frame_count',0)} frames)" if success else "Discarded"
    _log(f"Manual recording stopped — {msg}")
    return {"ok": True, "msg": msg}

# ─── Waypoint Collector ──────────────────────────────────────────────────────

def _wc_check():
    wc = _state.get("wc")
    if not wc: return None, JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    return wc, None

@app.get("/wc/status")
def wc_status():
    wc, err = _wc_check()
    if err: return err
    return {"ok": True, **wc.status}

@app.post("/wc/home/capture")
def wc_capture_home():
    wc, err = _wc_check()
    if err: return err
    ok = wc.capture_home()
    return {"ok": ok, "home_joints": wc.home_joints}

@app.post("/wc/home/speed")
def wc_home_speed(speed: float = 25.0):
    wc, err = _wc_check()
    if err: return err
    wc.home_speed = max(5.0, min(80.0, speed))
    wc.save()
    return {"ok": True, "home_speed": wc.home_speed}

@app.post("/wc/waypoint/capture")
def wc_capture_wp(name: str, noise: bool = False,
                  gripper: str = "", speed: float = 40.0):
    wc, err = _wc_check()
    if err: return err
    g = gripper if gripper in ("grip", "release") else None
    ok = wc.capture_waypoint(name, noise=noise, gripper=g, speed=speed)
    return {"ok": ok, "waypoints": wc.waypoints}

@app.post("/wc/waypoint/delete")
def wc_delete_wp(name: str):
    wc, err = _wc_check()
    if err: return err
    ok = wc.delete_waypoint(name)
    return {"ok": ok, "waypoints": wc.waypoints}

@app.post("/wc/waypoint/goto")
def wc_goto_wp(name: str):
    wc, err = _wc_check()
    if err: return err
    def _do(): wc.goto_waypoint_by_name(name)
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/wc/waypoint/update")
def wc_update_wp(name: str, noise: bool = False,
                 gripper: str = "", speed: float = 40.0):
    """noise/gripper/speed 만 업데이트 (위치 유지)."""
    wc, err = _wc_check()
    if err: return err
    g = gripper if gripper in ("grip", "release") else None
    for wp in wc.waypoints:
        if wp["name"] == name:
            wp["noise"]   = noise
            wp["gripper"] = g
            wp["speed"]   = speed
            wc.save()
            return {"ok": True, "waypoints": wc.waypoints}
    return JSONResponse({"ok": False, "msg": f"{name} not found"}, status_code=404)

@app.post("/wc/noise")
def wc_set_noise(xy_mm: float = 10.0, z_mm: float = 3.0):
    wc, err = _wc_check()
    if err: return err
    wc.noise_xy_mm = max(0.0, min(50.0, xy_mm))
    wc.noise_z_mm  = max(0.0, min(20.0, z_mm))
    wc.save()
    return {"ok": True, "noise_xy_mm": wc.noise_xy_mm, "noise_z_mm": wc.noise_z_mm}

@app.post("/wc/collect/step")
def wc_step():
    wc, err = _wc_check()
    if err: return err
    if wc._running:
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    def _do():
        wc.run_episode(
            rec_start_cb=_start_recording,
            rec_stop_cb=_stop_and_save
        )
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/wc/collect/auto")
def wc_auto(n: int = 10):
    wc, err = _wc_check()
    if err: return err
    if wc._running:
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    wc.run_auto(
        n=n,
        rec_start_cb=_start_recording,
        rec_stop_cb=_stop_and_save,
        done_cb=lambda done, total: _log(f"WC Auto 완료: {done}/{total}")
    )
    return {"ok": True}

@app.post("/wc/collect/stop")
def wc_stop():
    wc, err = _wc_check()
    if err: return err
    wc.stop()
    return {"ok": True}

@app.post("/estop")
def estop():
    w = _state["worker"]
    if w: w._stop_requested = True
    wc = _state.get("wc")
    if wc: wc.stop()
    if _state["gripper"]:
        try: _state["gripper"].emergency_release()
        except Exception: pass
    if _state["robot"]: _state["robot"].disable_robot()
    _log("E-STOP triggered")
    return {"ok": True}

# ─── Pick-Place ───────────────────────────────────────────────────────────

@app.post("/pick-place/step")
def step():
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _state["auto_target"] = 0
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/pick-place/auto")
def auto_collect(n: int = 10):
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _state.update(auto_target=n, auto_done=0)
    _log(f"Auto collect started: {n} episodes")
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/pick-place/stop")
def stop_collect():
    if _state["worker"]: _state["worker"]._stop_requested = True
    _state["auto_target"] = 0
    _log("Stop requested")
    return {"ok": True}

# ─── 카메라 ──────────────────────────────────────────────────────────────

@app.post("/camera/hik/start")
def hik_start():
    global _cam_hik_thread, _cam_hik_running
    if not _hik_available:
        return JSONResponse({"ok": False, "msg": "HIK SDK not available"}, status_code=400)
    if _state["camera_hik"] and _state["camera_hik"].initialized:
        return {"ok": True, "msg": "Already running"}
    try:
        cam = HikRobotCamera()
        if not cam.init_camera():
            return JSONResponse({"ok": False, "msg": "Exterior Cam 1 (HIK) init failed — check USB connection"}, status_code=500)
        _state["camera_hik"] = cam
        _cam_hik_running = True
        _cam_hik_thread  = threading.Thread(target=_hik_grab_loop, daemon=True)
        _cam_hik_thread.start()
        _log("Exterior Cam 1 (HIKRobot) started")
        return {"ok": True}
    except Exception as e:
        _log(f"HIK start error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/camera/hik/stop")
def hik_stop():
    global _cam_hik_running
    _cam_hik_running = False
    if _state["camera_hik"]:
        try: _state["camera_hik"].cleanup()
        except Exception: pass
        _state["camera_hik"] = None
    _log("Exterior Cam 1 (HIKRobot) stopped")
    return {"ok": True}

@app.post("/camera/zed/start")
def zed_start():
    global _cam_zed_thread, _cam_zed_running
    if not _zed_available:
        return JSONResponse({"ok": False, "msg": "ZED SDK not available"}, status_code=400)
    if _state["camera_zed"] and _state["camera_zed"].initialized:
        return {"ok": True, "msg": "Already running"}
    try:
        cam = ZedCamera()
        if not cam.init_camera():
            return JSONResponse({"ok": False, "msg": "Exterior Cam 2 (ZED) init failed — check USB3 connection"}, status_code=500)
        _state["camera_zed"] = cam
        _cam_zed_running = True
        _cam_zed_thread  = threading.Thread(target=_zed_grab_loop, daemon=True)
        _cam_zed_thread.start()
        _log("Exterior Cam 2 (ZED) started — LEFT view only")
        return {"ok": True}
    except Exception as e:
        _log(f"ZED start error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/camera/zed/stop")
def zed_stop():
    global _cam_zed_running
    _cam_zed_running = False
    if _state["camera_zed"]:
        try: _state["camera_zed"].cleanup()
        except Exception: pass
        _state["camera_zed"] = None
    _log("Exterior Cam 2 (ZED) stopped")
    return {"ok": True}

@app.get("/camera/hik/stream")
async def hik_stream():
    return StreamingResponse(
        _mjpeg_gen(lambda: _buf_hik_jpg),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/zed/stream")
async def zed_stream():
    return StreamingResponse(
        _mjpeg_gen(lambda: _buf_zed_jpg),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ─── WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)

# ─── Web UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)

_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UFactory xArm</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d0d0d;color:#e5e5e5;font-size:13px;min-width:960px}

/* ── TOP HEADER ─────────────────────────────────────────────────────────── */
#topbar{
  position:sticky;top:0;z-index:100;
  height:52px;background:#111;border-bottom:1px solid #2a2a2a;
  display:flex;align-items:center;padding:0 14px;gap:12px;
}
#topbar .brand{display:flex;align-items:center;gap:8px;min-width:160px}
#topbar .brand-icon{width:26px;height:26px;background:#FF6B00;border-radius:5px;
  display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px;color:#000}
#topbar .brand-name{font-size:.85rem;font-weight:700;letter-spacing:.5px;color:#fff}
#topbar .brand-sub{font-size:.65rem;color:#555;margin-top:1px}

#status-badge{
  display:flex;align-items:center;gap:6px;padding:4px 10px;
  background:#1a1a1a;border:1px solid #2a2a2a;border-radius:20px;
  font-size:.72rem;font-weight:600;white-space:nowrap;
}
.badge-dot{width:8px;height:8px;border-radius:50%;background:#333}
.badge-dot.standby{background:#22C55E}
.badge-dot.running{background:#FF6B00;animation:pulse .8s infinite}
.badge-dot.error{background:#EF4444;animation:pulse .5s infinite}
.badge-dot.off{background:#444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

#mini-joints{
  display:flex;gap:6px;align-items:center;flex:1;
  overflow:hidden;
}
.mj{display:flex;flex-direction:column;align-items:center;
  background:#1a1a1a;border:1px solid #222;border-radius:5px;
  padding:3px 7px;min-width:52px}
.mj-label{font-size:.53rem;color:#555;text-transform:uppercase}
.mj-val{font-size:.75rem;font-weight:700;color:#FF6B00;font-family:monospace}

#topbar-right{display:flex;align-items:center;gap:8px;margin-left:auto}
#rec-indicator{display:none;align-items:center;gap:5px;
  background:#1a0000;border:1px solid #5a0000;border-radius:4px;
  padding:3px 9px;font-size:.7rem;color:#ff4444;font-weight:700}
#rec-indicator .rdot{width:7px;height:7px;border-radius:50%;background:#EF4444;animation:pulse .8s infinite}
#estop-btn{
  background:#1a0000;border:2px solid #EF4444;color:#EF4444;
  padding:6px 14px;border-radius:5px;cursor:pointer;font-size:.8rem;font-weight:800;
  letter-spacing:.5px;transition:all .15s;white-space:nowrap;
}
#estop-btn:hover{background:#EF4444;color:#fff}

/* ── LAYOUT ─────────────────────────────────────────────────────────────── */
.layout{display:grid;grid-template-columns:290px 1fr 320px;gap:8px;padding:8px;align-items:start}
.col{display:flex;flex-direction:column;gap:8px;min-width:0}

/* ── CARD ────────────────────────────────────────────────────────────────── */
.card{background:#161616;border:1px solid #222;border-radius:8px;padding:12px;overflow:hidden}
h3{
  font-size:.68rem;text-transform:uppercase;letter-spacing:.8px;
  color:#e5e5e5;margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid #222;
  display:flex;align-items:center;gap:6px;
}
h3::before{content:'';display:inline-block;width:3px;height:12px;background:#FF6B00;border-radius:2px}

/* ── INPUTS / BUTTONS ────────────────────────────────────────────────────── */
.row{display:flex;gap:5px;margin-bottom:6px;align-items:center;flex-wrap:wrap}
input[type=text],input[type=number]{
  background:#0d0d0d;color:#e5e5e5;border:1px solid #333;
  padding:5px 8px;border-radius:5px;flex:1;min-width:0;font-size:.8rem;
  transition:border-color .15s;
}
input[type=text]:focus,input[type=number]:focus{outline:none;border-color:#FF6B00}
button{
  background:#0d0d0d;color:#aaa;border:1px solid #333;padding:5px 10px;
  border-radius:5px;cursor:pointer;font-size:.76rem;white-space:nowrap;transition:all .12s;
}
button:hover{background:#FF6B00;color:#000;border-color:#FF6B00}
button:active{filter:brightness(1.2)}
.btn-g{border-color:#22C55E;color:#22C55E}.btn-g:hover{background:#22C55E;color:#000;border-color:#22C55E}
.btn-r{border-color:#EF4444;color:#EF4444}.btn-r:hover{background:#EF4444;color:#fff;border-color:#EF4444}
.btn-y{border-color:#F59E0B;color:#F59E0B}.btn-y:hover{background:#F59E0B;color:#000;border-color:#F59E0B}
.btn-o{border-color:#FF6B00;color:#FF6B00}.btn-o:hover{background:#FF6B00;color:#000}

/* ── STATUS BARS ─────────────────────────────────────────────────────────── */
.sbar{background:#0d0d0d;border:1px solid #222;padding:5px 9px;border-radius:5px;font-size:.71rem;margin-bottom:5px;color:#aaa}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.on{background:#22C55E}.off{background:#555}.rec{background:#EF4444;animation:pulse .8s infinite}

/* ── DASHBOARD GRID ──────────────────────────────────────────────────────── */
.dash-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.dash-cell{background:#0d0d0d;border:1px solid #1e1e1e;border-radius:5px;padding:6px 4px;text-align:center}
.dash-label{font-size:.57rem;color:#444;text-transform:uppercase;letter-spacing:.4px}
.dash-val{font-size:.9rem;font-weight:700;color:#FF6B00;font-family:'Courier New',monospace;margin-top:2px}

/* ── CAMERA ──────────────────────────────────────────────────────────────── */
.cam-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.cam-box{background:#0d0d0d;border:1px solid #222;border-radius:6px;overflow:hidden}
.cam-label{font-size:.62rem;color:#555;padding:4px 8px;background:#111;border-bottom:1px solid #1a1a1a;letter-spacing:.3px}
img.stream{width:100%;height:460px;object-fit:contain;display:block;background:#050505}

/* ── LOG ─────────────────────────────────────────────────────────────────── */
#log{background:#080808;font-family:'Courier New',monospace;font-size:.66rem;
  height:130px;overflow-y:auto;padding:6px 8px;border-radius:5px;
  color:#22C55E;word-break:break-all;border:1px solid #1a1a1a}

/* ── SEPARATOR / SUB ─────────────────────────────────────────────────────── */
.sep{border-top:1px solid #1e1e1e;margin:8px 0}
.sub{font-size:.62rem;color:#444;margin-bottom:4px;margin-top:3px;letter-spacing:.2px}

/* ── JOG ─────────────────────────────────────────────────────────────────── */
.dpad{display:grid;grid-template-columns:repeat(3,50px);grid-template-rows:repeat(3,38px);gap:4px}
.dpad .jb{font-size:.85rem;font-weight:700;padding:0;display:flex;align-items:center;justify-content:center;
  border-radius:6px;user-select:none;-webkit-user-select:none;touch-action:none}
.dpad .jb.center{background:#1a1a1a;color:#333;font-size:.58rem;cursor:default;border:1px solid #1e1e1e}
.dpad .jb.center:hover{background:#1a1a1a;color:#333;border-color:#1e1e1e}
.zcol{display:flex;flex-direction:column;gap:4px;margin-left:8px}
.zcol .jb{width:46px;height:38px;font-size:.82rem;font-weight:700;display:flex;align-items:center;
  justify-content:center;border-radius:6px;user-select:none;-webkit-user-select:none;touch-action:none}
.rot-row{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.rot-row .jb{padding:5px 2px;font-size:.7rem;text-align:center;font-weight:600;
  border-radius:5px;user-select:none;-webkit-user-select:none;touch-action:none}
.joint-table{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.joint-table .jb{padding:6px 2px;font-size:.7rem;text-align:center;
  border-radius:5px;user-select:none;-webkit-user-select:none;touch-action:none}
.jb.jogging{background:#FF6B00 !important;color:#000 !important;border-color:#FF6B00 !important}
input[type=range]{width:100%;accent-color:#FF6B00}

/* ── POSE DISPLAY ────────────────────────────────────────────────────────── */
#pose-display{font-size:.67rem;color:#FF6B00;margin-top:4px;font-family:'Courier New',monospace;
  word-break:break-all;min-height:16px;background:#0d0d0d;border:1px solid #1e1e1e;
  border-radius:4px;padding:4px 7px}

/* ── ANCHOR TABLE ────────────────────────────────────────────────────────── */
.pc{text-align:center;padding:2px 4px;background:#0d0d0d;border-radius:3px;
  font-family:'Courier New',monospace;color:#555;border:1px solid #1e1e1e}
.pc.has-data{color:#FF6B00;font-weight:700;border-color:#FF6B00;background:#1a0e00}
.anc-active{background:#1a0e00 !important;border-color:#FF6B00 !important;color:#FF6B00 !important}

/* ── GRIPPER BUTTONS ─────────────────────────────────────────────────────── */
.grip-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px}
.grip-btn{padding:10px;font-size:.8rem;font-weight:700;border-radius:6px;
  cursor:pointer;transition:all .15s;user-select:none;text-align:center}
.grip-on{background:#1a0e00;border:2px solid #FF6B00;color:#FF6B00}
.grip-on:hover{background:#FF6B00;color:#000}
.grip-off{background:#0d0d0d;border:2px solid #333;color:#888}
.grip-off:hover{background:#333;color:#fff}
</style>
</head>
<body>

<!-- ═══════════════ TOP HEADER ═══════════════ -->
<div id="topbar">
  <div class="brand">
    <div class="brand-icon">X</div>
    <div>
      <div class="brand-name">UFACTORY</div>
      <div class="brand-sub">xArm6 Control Server</div>
    </div>
  </div>

  <div id="status-badge">
    <span class="badge-dot off" id="badge-dot"></span>
    <span id="badge-text">OFFLINE</span>
  </div>

  <div id="mini-joints">
    <div class="mj"><div class="mj-label">J1</div><div class="mj-val" id="hJ1">—</div></div>
    <div class="mj"><div class="mj-label">J2</div><div class="mj-val" id="hJ2">—</div></div>
    <div class="mj"><div class="mj-label">J3</div><div class="mj-val" id="hJ3">—</div></div>
    <div class="mj"><div class="mj-label">J4</div><div class="mj-val" id="hJ4">—</div></div>
    <div class="mj"><div class="mj-label">J5</div><div class="mj-val" id="hJ5">—</div></div>
    <div class="mj"><div class="mj-label">J6</div><div class="mj-val" id="hJ6">—</div></div>
  </div>

  <div id="topbar-right">
    <div id="rec-indicator"><span class="rdot"></span>REC</div>
    <button id="estop-btn" onclick="doEstop()">⚠ E-STOP</button>
  </div>
</div>

<!-- ═══════════════ MAIN LAYOUT ═══════════════ -->
<div class="layout">

<!-- ══════ LEFT COL ══════ -->
<div class="col">

  <!-- Connection -->
  <div class="card">
    <h3>Connection</h3>
    <div class="row">
      <input id="ip" type="text" value="192.168.1.225" style="max-width:125px">
      <button class="btn-g" onclick="api('POST','/connect',{ip:$('ip').value})">Connect</button>
      <button onclick="api('POST','/disconnect')">Disconnect</button>
    </div>
    <div id="conn-bar" class="sbar"><span class="dot off"></span>Disconnected</div>
    <div id="mode-bar" class="sbar" style="margin-bottom:7px">State: —</div>
    <div class="row" style="margin-bottom:0;gap:4px;flex-wrap:wrap">
      <button class="btn-g" onclick="api('POST','/enable')">Enable</button>
      <button onclick="api('POST','/disable')">Disable</button>
      <button class="btn-y" onclick="clearAlarm()">Clear Error</button>
      <button class="btn-y" onclick="api('POST','/resume').then(()=>addLog('▶ Resume sent'))">Resume</button>
      <button onclick="api('POST','/home')">Home</button>
    </div>
  </div>

  <!-- TCP Pose -->
  <div class="card">
    <h3>TCP Pose &amp; Joints</h3>
    <div class="sub">Position (mm) / Orientation (deg)</div>
    <div class="dash-grid" style="margin-bottom:6px">
      <div class="dash-cell"><div class="dash-label">X</div><div class="dash-val" id="dX">—</div></div>
      <div class="dash-cell"><div class="dash-label">Y</div><div class="dash-val" id="dY">—</div></div>
      <div class="dash-cell"><div class="dash-label">Z</div><div class="dash-val" id="dZ">—</div></div>
      <div class="dash-cell"><div class="dash-label">RX</div><div class="dash-val" id="dRX">—</div></div>
      <div class="dash-cell"><div class="dash-label">RY</div><div class="dash-val" id="dRY">—</div></div>
      <div class="dash-cell"><div class="dash-label">RZ</div><div class="dash-val" id="dRZ">—</div></div>
    </div>
    <div class="sub">Joint Angles (deg)</div>
    <div class="dash-grid">
      <div class="dash-cell"><div class="dash-label">J1</div><div class="dash-val" id="dJ1">—</div></div>
      <div class="dash-cell"><div class="dash-label">J2</div><div class="dash-val" id="dJ2">—</div></div>
      <div class="dash-cell"><div class="dash-label">J3</div><div class="dash-val" id="dJ3">—</div></div>
      <div class="dash-cell"><div class="dash-label">J4</div><div class="dash-val" id="dJ4">—</div></div>
      <div class="dash-cell"><div class="dash-label">J5</div><div class="dash-val" id="dJ5">—</div></div>
      <div class="dash-cell"><div class="dash-label">J6</div><div class="dash-val" id="dJ6">—</div></div>
    </div>
  </div>

  <!-- Data Collection -->
  <div class="card">
    <h3>Data Collection</h3>

    <div class="sub">앵커 페어 (v20+)</div>
    <div class="sub" style="font-size:.59rem;color:#333;margin-bottom:3px">방향</div>
    <div class="row" style="margin-bottom:5px">
      <button id="dir-A"    onclick="anchorSetDir('A')"    style="flex:1;padding:4px;font-size:.72rem">A→B</button>
      <button id="dir-B"    onclick="anchorSetDir('B')"    style="flex:1;padding:4px;font-size:.72rem">B→A</button>
      <button id="dir-auto" onclick="anchorSetDir('auto')" style="flex:1;padding:4px;font-size:.72rem">Auto</button>
    </div>

    <div class="sub" style="font-size:.59rem;color:#333;margin-bottom:3px">A앵커</div>
    <div class="row" style="margin-bottom:5px">
      <button id="aa-rand" onclick="anchorSetA(-1)" style="flex:1;padding:3px;font-size:.71rem">랜덤</button>
      <button id="aa-0"    onclick="anchorSetA(0)"  style="flex:1;padding:3px;font-size:.71rem">A3</button>
      <button id="aa-1"    onclick="anchorSetA(1)"  style="flex:1;padding:3px;font-size:.71rem">A4</button>
      <button id="aa-2"    onclick="anchorSetA(2)"  style="flex:1;padding:3px;font-size:.71rem">A5</button>
    </div>

    <div class="sub" style="font-size:.59rem;color:#333;margin-bottom:3px">B앵커</div>
    <div class="row" style="margin-bottom:5px">
      <button id="ba-rand" onclick="anchorSetB(-1)" style="flex:1;padding:3px;font-size:.71rem">랜덤</button>
      <button id="ba-0"    onclick="anchorSetB(0)"  style="flex:1;padding:3px;font-size:.71rem">B1</button>
      <button id="ba-1"    onclick="anchorSetB(1)"  style="flex:1;padding:3px;font-size:.71rem">B2</button>
      <button id="ba-2"    onclick="anchorSetB(2)"  style="flex:1;padding:3px;font-size:.71rem">B3</button>
    </div>

    <div id="anchor-display" class="sbar" style="margin-bottom:6px;font-size:.7rem">앵커: —</div>

    <div class="sub" style="font-size:.59rem;color:#333;margin-bottom:2px">A→B 진행 현황</div>
    <table style="width:100%;border-collapse:separate;border-spacing:2px;font-size:.63rem;margin-bottom:5px">
      <tr>
        <td style="color:#333;width:22px"></td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">B1</td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">B2</td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">B3</td>
      </tr>
      <tr><td style="color:#FF6B00;font-weight:700">A3</td>
        <td><div id="pl-AB_0_0" class="pc">0</div></td>
        <td><div id="pl-AB_0_1" class="pc">0</div></td>
        <td><div id="pl-AB_0_2" class="pc">0</div></td></tr>
      <tr><td style="color:#FF6B00;font-weight:700">A4</td>
        <td><div id="pl-AB_1_0" class="pc">0</div></td>
        <td><div id="pl-AB_1_1" class="pc">0</div></td>
        <td><div id="pl-AB_1_2" class="pc">0</div></td></tr>
      <tr><td style="color:#FF6B00;font-weight:700">A5</td>
        <td><div id="pl-AB_2_0" class="pc">0</div></td>
        <td><div id="pl-AB_2_1" class="pc">0</div></td>
        <td><div id="pl-AB_2_2" class="pc">0</div></td></tr>
    </table>

    <div class="sub" style="font-size:.59rem;color:#333;margin-bottom:2px">B→A 진행 현황</div>
    <table style="width:100%;border-collapse:separate;border-spacing:2px;font-size:.63rem;margin-bottom:8px">
      <tr>
        <td style="color:#333;width:22px"></td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">A3</td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">A4</td>
        <td style="color:#FF6B00;text-align:center;font-weight:700">A5</td>
      </tr>
      <tr><td style="color:#FF6B00;font-weight:700">B1</td>
        <td><div id="pl-BA_0_0" class="pc">0</div></td>
        <td><div id="pl-BA_0_1" class="pc">0</div></td>
        <td><div id="pl-BA_0_2" class="pc">0</div></td></tr>
      <tr><td style="color:#FF6B00;font-weight:700">B2</td>
        <td><div id="pl-BA_1_0" class="pc">0</div></td>
        <td><div id="pl-BA_1_1" class="pc">0</div></td>
        <td><div id="pl-BA_1_2" class="pc">0</div></td></tr>
      <tr><td style="color:#FF6B00;font-weight:700">B3</td>
        <td><div id="pl-BA_2_0" class="pc">0</div></td>
        <td><div id="pl-BA_2_1" class="pc">0</div></td>
        <td><div id="pl-BA_2_2" class="pc">0</div></td></tr>
    </table>

    <div class="sep"></div>

    <!-- 수동 레코딩 -->
    <div class="sub">수동 레코딩</div>
    <div id="rec-bar" class="sbar" style="margin-bottom:6px">대기 중</div>
    <div class="row" style="margin-bottom:8px">
      <button id="rec-start-btn" class="btn-r" style="flex:1;padding:7px;font-size:.78rem;font-weight:700"
        onclick="recStart()">● REC Start</button>
      <button id="rec-save-btn" class="btn-g" style="flex:1;padding:7px;font-size:.78rem" disabled
        onclick="recStop(true)">■ Save</button>
      <button id="rec-disc-btn" style="flex:1;padding:7px;font-size:.78rem;border-color:#555;color:#666" disabled
        onclick="recStop(false)">✕ Discard</button>
    </div>

    <div class="sep"></div>

    <!-- 자동 Pick-Place -->
    <div class="sub">자동 Pick-Place</div>
    <div id="auto-bar" class="sbar">Ready</div>
    <div class="row" style="margin-bottom:5px">
      <button class="btn-g" onclick="api('POST','/pick-place/step')" style="flex:1">▶ Step</button>
      <input id="auto-n" type="number" value="10" min="1" style="max-width:52px">
      <button class="btn-g" onclick="autoCollect()" style="flex:1">▶ Auto(n)</button>
      <button class="btn-y" onclick="api('POST','/pick-place/stop')">■ Stop</button>
    </div>
  </div>

  <!-- Waypoint Collector -->
  <div class="card">
    <h3>Waypoint Collector</h3>

    <!-- Home -->
    <div style="background:#0d0d0d;border:1px solid #222;border-radius:5px;padding:7px;margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
        <span style="font-size:.7rem;color:#aaa;font-weight:600">Home (Joint·고정)</span>
        <span id="wc-home-status" style="font-size:.65rem;color:#555">미설정</span>
      </div>
      <div class="row" style="margin-bottom:4px">
        <button class="btn-g" style="flex:1" onclick="wcCaptureHome()">📍 Home 캡처</button>
        <button onclick="api('POST','/wc/waypoint/goto',{name:'__home__'}).then(()=>wcRefresh())"
          style="flex:1" id="wc-goto-home-btn" disabled>→ Home</button>
      </div>
      <div style="display:flex;gap:5px;align-items:center">
        <span style="font-size:.65rem;color:#555">속도</span>
        <input type="range" min="5" max="60" value="25" id="wc-home-speed"
               style="flex:1;accent-color:#FF6B00"
               oninput="$('wc-home-spd-val').textContent=this.value">
        <span id="wc-home-spd-val" style="font-size:.65rem;color:#FF6B00;min-width:22px">25</span>
        <button onclick="api('POST','/wc/home/speed',{speed:parseInt($('wc-home-speed').value)})"
          style="padding:3px 7px;font-size:.65rem;border-color:#FF6B00;color:#FF6B00">Set</button>
      </div>
    </div>

    <!-- Noise 설정 -->
    <div style="display:flex;gap:6px;margin-bottom:8px;align-items:center">
      <span style="font-size:.65rem;color:#555;white-space:nowrap">XY noise</span>
      <input type="number" id="wc-nxy" value="10" min="0" max="50" style="max-width:48px;font-size:.72rem">
      <span style="font-size:.65rem;color:#555">mm &nbsp; Z</span>
      <input type="number" id="wc-nz" value="3" min="0" max="20" style="max-width:40px;font-size:.72rem">
      <span style="font-size:.65rem;color:#555">mm</span>
      <button onclick="wcSetNoise()" style="padding:3px 7px;font-size:.65rem;border-color:#FF6B00;color:#FF6B00">Set</button>
    </div>

    <!-- Waypoint 목록 -->
    <div class="sub">Waypoints (순서대로 실행)</div>
    <div id="wc-wp-list" style="margin-bottom:8px;display:flex;flex-direction:column;gap:3px">
      <div style="font-size:.65rem;color:#333;padding:4px">없음 — 아래에서 추가</div>
    </div>

    <!-- Waypoint 추가 -->
    <div style="background:#0d0d0d;border:1px solid #222;border-radius:5px;padding:7px;margin-bottom:8px">
      <div class="sub" style="margin-bottom:4px">현재 위치를 Waypoint로 추가</div>
      <div class="row" style="margin-bottom:4px">
        <input id="wc-wp-name" type="text" placeholder="이름 (예: pick)" style="max-width:90px;font-size:.75rem">
        <select id="wc-wp-gripper" style="background:#0d0d0d;color:#dde;border:1px solid #333;border-radius:4px;padding:4px;font-size:.72rem">
          <option value="">그리퍼 없음</option>
          <option value="grip">Grip (닫기)</option>
          <option value="release">Release (열기)</option>
        </select>
      </div>
      <div class="row" style="margin-bottom:4px;align-items:center">
        <label style="font-size:.65rem;color:#aaa;display:flex;align-items:center;gap:4px">
          <input type="checkbox" id="wc-wp-noise"> Noise 적용
        </label>
        <span style="font-size:.65rem;color:#555">속도</span>
        <input type="number" id="wc-wp-speed" value="40" min="5" max="100" style="max-width:45px;font-size:.72rem">
        <span style="font-size:.65rem;color:#555">mm/s</span>
        <button class="btn-g" style="flex:1" onclick="wcCaptureWp()">+ 추가</button>
      </div>
    </div>

    <!-- 수집 실행 -->
    <div class="sep"></div>
    <div id="wc-status-bar" class="sbar" style="margin-bottom:6px">대기</div>
    <div class="row" style="margin-bottom:5px">
      <button class="btn-g" style="flex:1;padding:7px" onclick="api('POST','/wc/collect/step').then(wcRefresh)">▶ Step</button>
      <input id="wc-auto-n" type="number" value="10" min="1" style="max-width:48px">
      <button class="btn-g" style="flex:1;padding:7px" onclick="wcAutoCollect()">▶ Auto(n)</button>
      <button class="btn-y" onclick="api('POST','/wc/collect/stop')">■ Stop</button>
    </div>
  </div>

</div><!-- end left col -->

<!-- ══════ CENTER COL ══════ -->
<div class="col">

  <div class="card">
    <h3>Cameras</h3>
    <div class="row" style="margin-bottom:4px;align-items:center">
      <span style="font-size:.71rem;color:#FF6B00;min-width:110px">HIKRobot (Wrist)</span>
      <button class="btn-g" onclick="api('POST','/camera/hik/start')">Start</button>
      <button onclick="api('POST','/camera/hik/stop')">Stop</button>
      <span id="hik-stat" style="font-size:.7rem;color:#444;margin-left:6px">● OFF</span>
    </div>
    <div class="row" style="margin-bottom:0;align-items:center">
      <span style="font-size:.71rem;color:#FF6B00;min-width:110px">ZED (Scene)</span>
      <button class="btn-g" onclick="api('POST','/camera/zed/start')">Start</button>
      <button onclick="api('POST','/camera/zed/stop')">Stop</button>
      <span id="zed-stat" style="font-size:.7rem;color:#444;margin-left:6px">● OFF</span>
    </div>
  </div>

  <div class="card" style="padding:8px">
    <div class="cam-row">
      <div class="cam-box">
        <div class="cam-label">CAM 1 — HIKRobot Wrist</div>
        <img class="stream" src="/camera/hik/stream" alt="HIK">
      </div>
      <div class="cam-box">
        <div class="cam-label">CAM 2 — ZED Scene (LEFT)</div>
        <img class="stream" src="/camera/zed/stream" alt="ZED">
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Log</h3>
    <div id="log"></div>
  </div>

</div><!-- end center col -->

<!-- ══════ RIGHT COL ══════ -->
<div class="col">
  <div class="card">
    <h3>Jog Control</h3>

    <!-- Gripper G2 -->
    <div style="background:#0d0d0d;border:1px solid #222;border-radius:6px;padding:8px;margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:.7rem;color:#aaa;font-weight:600">Gripper G2</span>
        <span style="font-size:.8rem;font-weight:700;color:#FF6B00;font-family:monospace">
          <span id="grip-pos">—</span> mm
        </span>
      </div>
      <div style="height:6px;background:#1a1a1a;border-radius:3px;margin-bottom:8px;overflow:hidden">
        <div id="grip-bar" style="height:100%;background:#FF6B00;border-radius:3px;width:0%;transition:width .3s"></div>
      </div>
      <div class="grip-row" style="margin-bottom:6px">
        <button class="grip-btn grip-on" onclick="api('POST','/gripper/grip')">
          ◉ CLOSE ALL<br><small style="font-size:.6rem;font-weight:400">[Q] 완전닫기</small>
        </button>
        <button class="grip-btn grip-off" onclick="api('POST','/gripper/release')">
          ○ OPEN ALL<br><small style="font-size:.6rem;font-weight:400">[W] 완전열기</small>
        </button>
      </div>
      <div style="display:flex;gap:5px;align-items:center;margin-bottom:5px">
        <input type="range" id="grip-slider" min="0" max="85" value="85"
               style="flex:1;accent-color:#FF6B00"
               oninput="$('grip-mm').textContent=this.value">
        <span id="grip-mm" style="font-size:.72rem;color:#FF6B00;min-width:28px;font-family:monospace">85</span>
        <button onclick="api('POST','/gripper/set',{pos:parseInt($('grip-slider').value)})"
          style="padding:4px 8px;font-size:.72rem;border-color:#FF6B00;color:#FF6B00">Set</button>
      </div>
      <div style="font-size:.62rem;color:#444;line-height:1.6">
        <b style="color:#666">[V]</b> 꾹=천천히 닫기 &nbsp;
        <b style="color:#666">[C]</b> 꾹=천천히 열기
      </div>
    </div>

    <!-- Speed -->
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:.71rem;color:#aaa;font-weight:600">Jog Speed</span>
        <span style="font-size:.85rem;font-weight:700;color:#FF6B00"><span id="speed-val">15</span>%</span>
      </div>
      <input type="range" id="jog-speed" min="1" max="50" value="15"
             oninput="$('speed-val').textContent=this.value">
    </div>

    <!-- Capture pose -->
    <div style="margin-bottom:10px">
      <button onclick="capturePose()"
        style="width:100%;padding:6px;font-size:.74rem;background:#1a0e00;border-color:#FF6B00;color:#FF6B00">
        📍 현재 좌표 캡처
      </button>
      <div id="pose-display" style="margin-top:4px">—</div>
    </div>

    <div class="sep"></div>

    <!-- TCP XY D-pad + Z -->
    <div class="sub">TCP XY / Z &nbsp;·&nbsp; 키보드: ←→↑↓ / Z=Z+ X=Z-</div>
    <div style="display:flex;align-items:center;margin-bottom:10px">
      <div class="dpad">
        <div></div>
        <button class="jb btn-g" id="jb-X+" data-axis="X+">↑<br><small style="font-size:.5rem">X+</small></button>
        <div></div>
        <button class="jb btn-g" id="jb-Y+" data-axis="Y+">←<br><small style="font-size:.5rem">Y+</small></button>
        <div class="jb center">XY</div>
        <button class="jb btn-g" id="jb-Y-" data-axis="Y-">→<br><small style="font-size:.5rem">Y-</small></button>
        <div></div>
        <button class="jb btn-g" id="jb-X-" data-axis="X-">↓<br><small style="font-size:.5rem">X-</small></button>
        <div></div>
      </div>
      <div class="zcol">
        <button class="jb btn-g" id="jb-Z+" data-axis="Z+">Z+<br><small style="font-size:.5rem">▲</small></button>
        <button class="jb btn-g" id="jb-Z-" data-axis="Z-">Z-<br><small style="font-size:.5rem">▼</small></button>
      </div>
    </div>

    <div class="sub">Rotation</div>
    <div class="rot-row" style="margin-bottom:10px">
      <button class="jb" id="jb-Rx+" data-axis="Rx+">Rx+</button>
      <button class="jb" id="jb-Rx-" data-axis="Rx-">Rx-</button>
      <button class="jb" id="jb-Ry+" data-axis="Ry+">Ry+</button>
      <button class="jb" id="jb-Ry-" data-axis="Ry-">Ry-</button>
      <button class="jb" id="jb-Rz+" data-axis="Rz+">Rz+</button>
      <button class="jb" id="jb-Rz-" data-axis="Rz-">Rz-</button>
    </div>

    <div class="sep"></div>

    <div class="sub">Joint Jog</div>
    <div class="joint-table">
      <button class="jb" id="jb-J1+" data-axis="J1+">J1+</button>
      <button class="jb" id="jb-J1-" data-axis="J1-">J1-</button>
      <button class="jb" id="jb-J2+" data-axis="J2+">J2+</button>
      <button class="jb" id="jb-J2-" data-axis="J2-">J2-</button>
      <button class="jb" id="jb-J3+" data-axis="J3+">J3+</button>
      <button class="jb" id="jb-J3-" data-axis="J3-">J3-</button>
      <button class="jb" id="jb-J4+" data-axis="J4+">J4+</button>
      <button class="jb" id="jb-J4-" data-axis="J4-">J4-</button>
      <button class="jb" id="jb-J5+" data-axis="J5+">J5+</button>
      <button class="jb" id="jb-J5-" data-axis="J5-">J5-</button>
      <button class="jb" id="jb-J6+" data-axis="J6+">J6+</button>
      <button class="jb" id="jb-J6-" data-axis="J6-">J6-</button>
    </div>
  </div>
</div><!-- end right col -->

</div><!-- end layout -->

<script>
const $ = id => document.getElementById(id);
const api = async (m, p, q={}) => {
  const url = p + (Object.keys(q).length ? '?' + new URLSearchParams(q) : '');
  try {
    const res = await fetch(url, {method:m});
    const data = await res.json();
    if (data && data.msg) addLog((data.ok ? '✓ ' : '✗ ') + data.msg);
    if (!res.ok && !data.msg) addLog(`✗ ${m} ${p} → HTTP ${res.status}`);
    return data;
  } catch(e) { addLog('✗ Network: ' + e); }
};
const addLog = msg => {
  const b=$('log'); b.innerHTML += msg+'<br>'; b.scrollTop=b.scrollHeight;
};

// WebSocket
const ws = new WebSocket(`ws://${location.host}/ws/logs`);
ws.onmessage = e => addLog(e.data);
ws.onopen = () => addLog('[WS] connected');
setInterval(() => { if(ws.readyState===1) ws.send('ping'); }, 10000);

// Status polling
setInterval(async () => {
  const s = await api('GET','/status');
  if(!s) return;

  // Top header status badge
  const dot = $('badge-dot');
  const txt = $('badge-text');
  if (!s.connected) {
    dot.className = 'badge-dot off';
    txt.textContent = 'OFFLINE';
  } else if (s.robot_mode === 9) {
    dot.className = 'badge-dot error';
    txt.textContent = 'ERROR';
  } else if (s.robot_mode === 7) {
    dot.className = 'badge-dot running';
    txt.textContent = 'RUNNING';
  } else {
    dot.className = 'badge-dot standby';
    txt.textContent = s.robot_mode_str || 'STANDBY';
  }

  // REC indicator in header
  const recEl = $('rec-indicator');
  if (recEl) recEl.style.display = s.recording ? 'flex' : 'none';

  // conn-bar
  $('conn-bar').innerHTML = `<span class="dot ${s.connected?'on':'off'}"></span>`
    + (s.connected ? 'Connected' : 'Disconnected')
    + (s.recording ? ' &nbsp;<span class="dot rec"></span><b style="color:#EF4444"> REC</b>' : '');

  // mode bar
  const isError = s.robot_mode === 9;
  $('mode-bar').textContent = (isError ? '⚠ ERROR — ' : 'State: ') + (s.robot_mode_str||'—')
    + (s.recording ? ` · REC ${s.frames??''}f` : '');
  $('mode-bar').style.color = isError ? '#EF4444' : '#aaa';

  // Auto-stop on repeated error
  if (!window._errCount) window._errCount = 0;
  if (isError) { window._errCount++; } else { window._errCount = 0; }
  if (window._errCount >= 2 && s.auto_target > 0) {
    window._errCount = 0;
    addLog('⚠ 로봇 에러 감지 — 자동 수집 중단. Clear Error 후 재시작하세요.');
    api('POST','/pick-place/stop');
  }

  // Joint values — header + card
  if(s.joints) {
    ['J1','J2','J3','J4','J5','J6'].forEach((k,i)=>{
      const v = s.joints[i]?.toFixed(1) ?? '—';
      const el=$('d'+k); if(el) el.textContent=v;
      const hEl=$('h'+k); if(hEl) hEl.textContent=v+'°';
    });
  }
  // Gripper position
  const gp = s.gripper_pos;
  const gpEl = $('grip-pos');
  const gpBar = $('grip-bar');
  if (gpEl) gpEl.textContent = gp !== null && gp !== undefined ? gp.toFixed(1) : '—';
  if (gpBar && gp !== null && gp !== undefined) gpBar.style.width = Math.min(100, (gp/85)*100).toFixed(1)+'%';
  // Manual rec bar
  const recBarEl = $('rec-bar');
  const recStartBtn = $('rec-start-btn');
  const recSaveBtn  = $('rec-save-btn');
  const recDiscBtn  = $('rec-disc-btn');
  if (recBarEl) {
    if (s.recording) {
      recBarEl.innerHTML = '<span class="dot rec"></span> <b style="color:#EF4444">REC</b> ' + (s.frames||0) + ' frames';
      recBarEl.style.color = '#EF4444';
      if(recStartBtn) { recStartBtn.disabled=true; recStartBtn.style.opacity='0.4'; }
      if(recSaveBtn)  { recSaveBtn.disabled=false; recSaveBtn.style.opacity='1'; }
      if(recDiscBtn)  { recDiscBtn.disabled=false; recDiscBtn.style.opacity='1'; recDiscBtn.style.color='#aaa'; recDiscBtn.style.borderColor='#aaa'; }
    } else {
      recBarEl.textContent = '대기 중';
      recBarEl.style.color = '#aaa';
      if(recStartBtn) { recStartBtn.disabled=false; recStartBtn.style.opacity='1'; }
      if(recSaveBtn)  { recSaveBtn.disabled=true;  recSaveBtn.style.opacity='0.4'; }
      if(recDiscBtn)  { recDiscBtn.disabled=true;  recDiscBtn.style.opacity='0.4'; recDiscBtn.style.color='#666'; recDiscBtn.style.borderColor='#555'; }
    }
  }
  if(s.pose) ['X','Y','Z','RX','RY','RZ'].forEach((k,i)=>{
    const el=$('d'+k); if(el) el.textContent=s.pose[i]?.toFixed(1)??'—';
  });

  $('hik-stat').textContent = s.cam_hik ? '● ON' : '● OFF';
  $('hik-stat').style.color  = s.cam_hik ? '#22C55E' : '#444';
  $('zed-stat').textContent  = s.cam_zed ? '● ON' : '● OFF';
  $('zed-stat').style.color  = s.cam_zed ? '#22C55E' : '#444';

  if(s.auto_target > 0)
    $('auto-bar').textContent = `Auto: ${s.auto_done}/${s.auto_target}`;
  else if(s.worker_running)
    $('auto-bar').textContent = 'Running…';
  else
    $('auto-bar').textContent = 'Ready';


}, 800);

// Jog
const getSpeed = () => parseInt($('jog-speed').value) || 5;
let _jogAxis = null;
const jogStart = axis => {
  if (_jogAxis === axis) return;
  _jogAxis = axis;
  const el = document.getElementById('jb-' + axis);
  if (el) el.classList.add('jogging');
  api('POST','/jog/start',{axis, speed:getSpeed()});
};
const jogStop = () => {
  if (!_jogAxis) return;
  const el = document.getElementById('jb-' + _jogAxis);
  if (el) el.classList.remove('jogging');
  _jogAxis = null;
  api('POST','/jog/stop');
};
document.querySelectorAll('.jb[data-axis]').forEach(btn => {
  const ax = btn.dataset.axis;
  btn.addEventListener('mousedown',  e => { e.preventDefault(); jogStart(ax); });
  btn.addEventListener('touchstart', e => { e.preventDefault(); jogStart(ax); });
  btn.addEventListener('mouseup',    () => jogStop());
  btn.addEventListener('touchend',   () => jogStop());
  btn.addEventListener('mouseleave', () => { if(_jogAxis===ax) jogStop(); });
});
const KEY_MAP = {
  'ArrowLeft':'Y+',  'ArrowRight':'Y-',
  'ArrowUp':'X+',    'ArrowDown':'X-',
  'z':'Z+', 'Z':'Z+', 'x':'Z-', 'X':'Z-',
};
let _gripJogging = false;
document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  const k = e.key;
  // V = close jog, C = open jog (keydown 첫 번만, 꾹 누르면 계속 이동 중)
  if (k==='v'||k==='V') {
    if (!e.repeat) { _gripJogging=true; api('POST','/gripper/jog',{direction:'close'}); }
    e.preventDefault(); return;
  }
  if (k==='c'||k==='C') {
    if (!e.repeat) { _gripJogging=true; api('POST','/gripper/jog',{direction:'open'}); }
    e.preventDefault(); return;
  }
  if (e.repeat) return;
  if (k==='q'||k==='Q') { api('POST','/gripper/grip');    return; }
  if (k==='w'||k==='W') { api('POST','/gripper/release'); return; }
  const ax = KEY_MAP[k];
  if (ax) { jogStart(ax); e.preventDefault(); }
});
document.addEventListener('keyup', e => {
  const k = e.key;
  if ((k==='v'||k==='V'||k==='c'||k==='C') && _gripJogging) {
    _gripJogging = false;
    api('POST','/gripper/jog/stop');
    e.preventDefault(); return;
  }
  const ax = KEY_MAP[k];
  if (ax && _jogAxis===ax) { jogStop(); e.preventDefault(); }
});
window.addEventListener('blur', () => jogStop());

// Capture pose
const capturePose = async () => {
  const d = await api('GET','/pose');
  if(d && d.ok)
    $('pose-display').textContent = `X:${d.x}  Y:${d.y}  Z:${d.z} | Rx:${d.rx}  Ry:${d.ry}  Rz:${d.rz}`;
  else
    $('pose-display').textContent = 'fetch failed';
};

const autoCollect = () => api('POST','/pick-place/auto',{n:$('auto-n').value});
const recStart = async () => {
  const d = await api('POST','/record/start');
  if(d && d.ok) addLog('● REC started');
};
const recStop = async (success) => {
  const d = await api('POST', '/record/stop', {success});
  if(d && d.ok) addLog(success ? '■ ' + d.msg : '✕ Discarded');
};
const doEstop = () => { if(confirm('E-STOP?')) api('POST','/estop'); };

const clearAlarm = async () => {
  const d = await api('POST','/clear-alarm');
  if(d&&d.ok) addLog('✓ Error cleared');
};

// Anchor pair
let _ancDir = 'auto', _ancA = -1, _ancB = -1;
const _anchorApply = async () => {
  try {
    const params = new URLSearchParams({direction:_ancDir, a_idx:_ancA, b_idx:_ancB});
    await fetch('/anchor?' + params, {method:'POST'});
  } catch(e) { addLog('✗ anchor: ' + e); }
  _anchorRefreshUI();
};
const anchorSetDir = dir  => { _ancDir = dir; _anchorApply(); };
const anchorSetA   = idx  => { _ancA = idx;   _anchorApply(); };
const anchorSetB   = idx  => { _ancB = idx;   _anchorApply(); };
const _anchorRefreshUI = () => {
  ['A','B','auto'].forEach(v => {
    const el = $('dir-' + v);
    if (el) el.className = (v === _ancDir) ? 'anc-active' : '';
  });
  ['rand',0,1,2].forEach((v,i) => {
    const el = $('aa-' + v);
    if (el) el.className = ((i===0 ? -1 : i-1) === _ancA) ? 'anc-active' : '';
  });
  ['rand',0,1,2].forEach((v,i) => {
    const el = $('ba-' + v);
    if (el) el.className = ((i===0 ? -1 : i-1) === _ancB) ? 'anc-active' : '';
  });
  const aLbl = _ancA >= 0 ? (['A3','A4','A5'][_ancA]??'?') : '랜덤';
  const bLbl = _ancB >= 0 ? (['B1','B2','B3'][_ancB]??'?') : '랜덤';
  const dLbl = {A:'A→B', B:'B→A', auto:'Auto(교번)'}[_ancDir] ?? _ancDir;
  const el = $('anchor-display');
  if (el) el.textContent = `${dLbl} | A=${aLbl} | B=${bLbl}`;
};

setInterval(async () => {
  try {
    const r = await fetch('/anchor-status');
    const d = await r.json();
    const counts = d.pair_counts || {};
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) {
        const el = $(`pl-AB_${a}_${b}`);
        if (el) { const n=counts[`AB_${a}_${b}`]||0; el.textContent=n; el.className=n>0?'pc has-data':'pc'; }
      }
    for (let b = 0; b < 3; b++)
      for (let a = 0; a < 3; a++) {
        const el = $(`pl-BA_${b}_${a}`);
        if (el) { const n=counts[`BA_${b}_${a}`]||0; el.textContent=n; el.className=n>0?'pc has-data':'pc'; }
      }
  } catch(e) {}
}, 2000);

_anchorRefreshUI();

// ── Waypoint Collector ─────────────────────────────────────────────────────
const wcRefresh = async () => {
  const d = await api('GET', '/wc/status');
  if (!d || !d.ok) return;

  // Home status
  const hEl = $('wc-home-status');
  if (hEl) {
    if (d.home_set) {
      const j = d.home_joints.map(v => v.toFixed(1)).join(', ');
      hEl.textContent = '설정됨';
      hEl.style.color = '#22C55E';
    } else {
      hEl.textContent = '미설정';
      hEl.style.color = '#555';
    }
  }
  const ghBtn = $('wc-goto-home-btn');
  if (ghBtn) ghBtn.disabled = !d.home_set;

  // Noise
  const nxy = $('wc-nxy'); const nz = $('wc-nz');
  if (nxy) nxy.value = d.noise_xy_mm;
  if (nz)  nz.value  = d.noise_z_mm;

  // Waypoints list
  const listEl = $('wc-wp-list');
  if (listEl) {
    if (!d.waypoints || d.waypoints.length === 0) {
      listEl.innerHTML = '<div style="font-size:.65rem;color:#333;padding:4px">없음 — 아래에서 추가</div>';
    } else {
      listEl.innerHTML = d.waypoints.map((wp, i) => {
        const noiseTag  = wp.noise   ? '<span style="color:#FF6B00;font-size:.58rem">±noise</span>' : '';
        const gripTag   = wp.gripper ? `<span style="color:#22C55E;font-size:.58rem">${wp.gripper}</span>` : '';
        const pos = wp.values ? `(${wp.values[0].toFixed(0)}, ${wp.values[1].toFixed(0)}, ${wp.values[2].toFixed(0)})` : '';
        return `<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;padding:5px 7px;display:flex;align-items:center;gap:5px">
          <span style="font-size:.62rem;color:#555;min-width:14px">${i+1}</span>
          <span style="font-size:.72rem;color:#FF6B00;font-weight:600;min-width:70px">${wp.name}</span>
          <span style="font-size:.62rem;color:#444;flex:1">${pos}</span>
          ${noiseTag} ${gripTag}
          <button onclick="api('POST','/wc/waypoint/goto',{name:'${wp.name}'})" style="padding:2px 6px;font-size:.62rem">→</button>
          <button onclick="wcDeleteWp('${wp.name}')" style="padding:2px 6px;font-size:.62rem;border-color:#EF4444;color:#EF4444">✕</button>
        </div>`;
      }).join('');
    }
  }

  // Status bar
  const sb = $('wc-status-bar');
  if (sb) {
    sb.textContent = d.running ? '실행 중…' : `대기  |  Waypoints: ${d.waypoints?.length || 0}`;
    sb.style.color = d.running ? '#FF6B00' : '#aaa';
  }
};

const wcCaptureHome = async () => {
  const d = await api('POST', '/wc/home/capture');
  if (d && d.ok) { addLog('✓ Home 캡처됨'); wcRefresh(); }
};
const wcCaptureWp = async () => {
  const name    = $('wc-wp-name').value.trim();
  const gripper = $('wc-wp-gripper').value;
  const noise   = $('wc-wp-noise').checked;
  const speed   = parseFloat($('wc-wp-speed').value) || 40;
  if (!name) { addLog('✗ 이름 입력 필요'); return; }
  const d = await api('POST', '/wc/waypoint/capture', {name, noise, gripper, speed});
  if (d && d.ok) { addLog(`✓ Waypoint 추가: ${name}`); wcRefresh(); }
};
const wcDeleteWp = async (name) => {
  const d = await api('POST', '/wc/waypoint/delete', {name});
  if (d && d.ok) { addLog(`✓ 삭제: ${name}`); wcRefresh(); }
};
const wcSetNoise = async () => {
  const xy = parseFloat($('wc-nxy').value) || 10;
  const z  = parseFloat($('wc-nz').value)  || 3;
  const d = await api('POST', '/wc/noise', {xy_mm: xy, z_mm: z});
  if (d && d.ok) addLog(`✓ Noise: XY±${xy}mm Z±${z}mm`);
};
const wcAutoCollect = () => {
  const n = parseInt($('wc-auto-n').value) || 10;
  api('POST', '/wc/collect/auto', {n}).then(wcRefresh);
  addLog(`▶ WC Auto ${n}회 시작`);
};

// WC 상태 2초마다 갱신
setInterval(wcRefresh, 2000);
wcRefresh();
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn, socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        ip = "localhost"
    print(f"\n  UFactory xArm Robot Server")
    print(f"  Open: http://{ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
