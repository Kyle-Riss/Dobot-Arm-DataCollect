# Dobot E6 Pick-and-Place VLA 데이터 수집 프로젝트

## 프로젝트 개요
- **목적**: Dobot E6 로봇으로 orange box pick-and-place VLA 학습 데이터 수집
- **데이터셋 이름**: `billy/dobot_e6_pick_place_orange_v2` (HF 업로드 예정명 미확정)
- **이전 데이터셋**: `billy/dobot_e6_pick_place_random_v1` (학습 실험명: `e6_2cam_lora_v1`)
- **새 학습 실험명**: `pi05_e6_v2_lora` / `e6_2cam_lora_v2`

---

## 현재 데이터셋 상태 (2026-04-25 기준)
- **저장 경로**: `/media/billye6/새 볼륨/Dobot/2CAM-Orange/`
- **총 에피소드**: 401개 수집, **유효 400개** (ep 382 제외)
  - ep 382: episode_meta.json 불완전(prompt/zone 없음), episode_events.csv 없음 → 학습 제외
- **방향 분포**: A→B(left→right) 200개 / B→A(right→left) 200개
- **프레임**: 평균 320프레임/에피소드 (min 241, max 402), 16Hz
- **mode=9 필터링**: 완료 (191개 에피소드 dataset.npy에서 1511프레임 제거)
- **검증 결과**: HIK=ZED 프레임 수 전 에피소드 일치, 프롬프트/zone 전부 정상

---

## 핵심 소스 파일
| 파일 | 역할 |
|------|------|
| `src/robot_server.py` | FastAPI 수집 서버, `_record_tick()` 50ms 폴링으로 HIK+ZED 동시 저장 |
| `src/pick_place_gui_new.py` | 메인 GUI + PickPlaceStepWorker (pick/place 로직, 메타데이터 생성) |
| `src/pick_place_gui_define.py` | Phase1 고정 init 포즈 GUI (상속) |
| `src/dobot_e6_controller.py` | 로봇 TCP/IP 제어 (move_j, 충돌 복구) |
| `src/ros2_recorder.py` | ROS2 토픽 퍼블리시 전용 (저장 안 함) |

---

## 중요 설정값 (pick_place_gui_new.py)
```python
RELEASE_Z = 120.0          # pick/place 하강 목표 Z (mm) — 실측 기준
POS_1~9 Z = 120.0          # 섹션별 pick 위치 Z (mm)
_zone = {"A": "left", "B": "right"}   # A=왼쪽, B=오른쪽
```

## 프롬프트 (확정)
- A→B: `"pick up the orange box from the left side and place it on the right side"`
- B→A: `"pick up the orange box from the right side and place it on the left side"`

---

## 에피소드 구조
```
/media/billye6/새 볼륨/Dobot/2CAM-Orange/{N}/
  images/hik/frame_000000.jpg ...   # HIK 손목 카메라 (224×224)
  images/zed/frame_000000.jpg ...   # ZED 장면 카메라 (224×224)
  dataset.npy                        # 프레임별 dict array (frame_id, joint_angles[6], robot_mode 등)
  episode_meta.json                  # prompt, source_zone, target_zone, success 등
  episode_events.csv                 # gripper_on/off 이벤트
  robot_data.csv                     # 프레임별 관절각/TCP/gripper/robot_mode
  metadata.txt
```

### dataset.npy 한 행 구조
```python
{
  'frame_id': int,
  'timestamp': float,
  'image_path_hik': 'hik/frame_XXXXXX.jpg',
  'image_path_zed': 'zed/frame_XXXXXX.jpg',
  'joint_angles': [j1, j2, j3, j4, j5, j6],  # 절대값 degrees, 6-dim
  'tcp_pose': [x, y, z, rx, ry, rz],
  'gripper_tooldo1': 0 or 1,
  'gripper_tooldo2': 0 or 1,
  'robot_mode': int,  # 5=ready, 7=running (mode=9 제거 완료)
}
```

---

## action 스펙 (v1/v2 공통)
- **타입**: 절대 next-position (degrees)
- **차원**: 6 (joint 1~6), gripper 미포함
- **robot_mode**: 5=준비(정지), 7=이동 중 → mode=5 유지, mode=9(에러) 제거 완료

---

## 주요 변경 이력 요약
1. **_record_tick() 복원**: ROS2+ZED 조건으로 스킵되던 것 → 항상 실행 (16Hz 안정화)
2. **ROS2 완전 분리**: `_ros2.start_recording()` 제거, 저장은 `_record_tick()` 단독
3. **RELEASE_Z**: 101.7 → 121.0 → **120.0** (orange box 실측)
4. **프롬프트**: "section A/B" → "left/right side" (카메라 이미지에서 시각적 구분 불가)
5. **충돌 복구**: 매 이동마다 Resume/Clear/Enable → `_recover_if_needed()` (mode 확인 후만 실행)
6. **JS debounce**: 에러 즉시 stop → 2회 연속 감지 시만 stop

---

## 앵커 페어 수집 시스템 (v20+)

### 개념
기존 완전 랜덤 수집 대신 **고정 타점(앵커) 중심 ±10mm** 로 수집해서 모델이 특정 위치를 학습할 수 있도록 개선.

### 앵커 좌표
| 이름 | X (mm) | Y (mm) | 비고 |
|------|--------|--------|------|
| A3   | 217.75 | -405.65 | A섹션 하단 |
| A4   | 220.21 | -368.72 | A섹션 중앙 (기존 고정 pick 위치) |
| A5   | 221.63 | -318.39 | A섹션 상단 |
| B1   |  30.0  | -340.0  | B섹션 상단 |
| B2   |  30.0  | -377.0  | B섹션 중앙 |
| B3   |  30.0  | -415.0  | B섹션 하단 |

- B앵커 X=30 고정: J1 회전량 최소화 목적 (로봇 형상이 'ㄹ'형태라 수평 이격 시 과회전 발생)
- 노이즈: ±10mm 균등 샘플링 → 실제 도달 범위 앵커 기준 ±10mm 이내

### 18 페어 구성
- A→B 9쌍: (A3,A4,A5) × (B1,B2,B3)
- B→A 9쌍: (B1,B2,B3) × (A3,A4,A5)
- 목표: 쌍당 50~100 에피소드 균등 분포

### robot_server.py UI 조작법
Web UI(`http://<ip>:8000`) → Data Collection 카드 하단 **앵커 페어 선택** 섹션

| 버튼 | 동작 |
|------|------|
| 방향: A→B | 매 에피소드 A에서 pick, B에 place만 수행 |
| 방향: B→A | 매 에피소드 B에서 pick, A에 place만 수행 |
| 방향: Auto | A→B, B→A 교번 자동 수행 (손 안 대도 됨) |
| A앵커: 랜덤 | A3/A4/A5 중 매번 무작위 선택 |
| A앵커: A3/A4/A5 | 해당 앵커로만 고정 |
| B앵커: 랜덤 | B1/B2/B3 중 매번 무작위 선택 |
| B앵커: B1/B2/B3 | 해당 앵커로만 고정 |

### 수집 예시
```
[A4 고정 + B2 고정 + Auto + ▶ Auto(100)]
  ep1: A4(±10mm) pick → B2(±10mm) place
  ep2: B2(±10mm) pick → A4(±10mm) place
  ... 100회 교번 → A→B표 [A4][B2]=50, B→A표 [B2][A4]=50

[A3 고정 + B 랜덤 + Auto + ▶ Auto(90)]
  ep1: A3(±10mm) pick → B2(±10mm) place  (B가 랜덤으로 B2 선택)
  ep2: B1(±10mm) pick → A3(±10mm) place  (B가 랜덤으로 B1 선택)
  ep3: A3(±10mm) pick → B3(±10mm) place
  ... → A3행의 B1/B2/B3 카운트가 분산 누적
```

### episode_meta.json 추가 필드 (v20+)
```json
{
  "pick_anchor": [220.21, -368.72],   // 실제 사용된 앵커 좌표 (±10mm 샘플의 기준점)
  "place_anchor": [30.0, -377.0],
  "pick_section": "A",
  "place_section": "B"
}
```
앵커 없이 수집된 구버전 에피소드는 `pick_anchor: null`.

### 진행 현황 표 (UI 실시간 갱신, 2초 폴링)
- 서버 재시작 시 카운트 리셋 (인메모리). 영구 저장 필요 시 episode_meta.json 에서 재집계 가능.
- 셀이 초록색으로 변하면 해당 쌍 수집 완료 기록 있음.

---

## 남은 작업 (학습 서버에서 진행)
- [ ] `norm_stats.json` 계산 (j5/rx/ry/rz near-zero std 주의, min_std=1e-3)
- [ ] HuggingFace 업로드 (`billy/dobot_e6_pick_place_orange_v2`)
- [ ] ep 382 폴더 삭제 또는 excluded 목록 관리
- [ ] openpi config 작성 (`pi05_e6_v2_lora`, fps=16, action_dim=6, img=224×224)
