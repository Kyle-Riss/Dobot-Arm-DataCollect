#!/usr/bin/env python3
"""
Step 6 검증: 메타데이터 앵커 필드 확인
- __init__에 _selected_a/b_anchor 초기화
- episode_meta_ready emit에 pick_anchor / place_anchor 추가
- 실행: python verify_step6_metadata.py
"""
import re, json, os, tempfile

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_place_gui_new.py")
code = open(SRC).read()

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
errors = 0

def check(cond, msg):
    global errors
    if cond:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        errors += 1

A_ANCHORS = [(217.75,-405.65),(220.21,-368.72),(221.63,-318.39)]
B_ANCHORS = [(30.0,-340.0),(30.0,-377.0),(30.0,-415.0)]

print("=" * 60)
print("  Step 6 검증: 메타데이터 앵커 필드")
print("=" * 60)

# ── 1. __init__ 초기화 확인 ───────────────────────────
print("\n[1] __init__ 초기화 확인")
check(bool(re.search(r'self\._selected_a_anchor\s*=\s*None', code)),
      "self._selected_a_anchor = None 초기화 존재")
check(bool(re.search(r'self\._selected_b_anchor\s*=\s*None', code)),
      "self._selected_b_anchor = None 초기화 존재")

# PickPlaceStepWorker.__init__ 블록 내에 있는지 확인
m_init = re.search(r'def __init__\(self, robot, gripper.*?def \w', code, re.DOTALL)
init_block = m_init.group(0) if m_init else ""
check('self._selected_a_anchor = None' in init_block,
      "PickPlaceStepWorker.__init__ 내 _selected_a_anchor = None")
check('self._selected_b_anchor = None' in init_block,
      "PickPlaceStepWorker.__init__ 내 _selected_b_anchor = None")

# ── 2. emit 딕셔너리에 필드 존재 확인 ─────────────────
print("\n[2] episode_meta_ready.emit() 필드 확인")
m_emit = re.search(r'episode_meta_ready\.emit\(\{(.+?)\}\)', code, re.DOTALL)
emit_block = m_emit.group(1) if m_emit else ""

check('"pick_anchor"' in emit_block,  '"pick_anchor" 필드 존재')
check('"place_anchor"' in emit_block, '"place_anchor" 필드 존재')
check('_selected_a_anchor' in emit_block, '_selected_a_anchor 참조')
check('_selected_b_anchor' in emit_block, '_selected_b_anchor 참조')

# ── 3. A->B 시나리오 시뮬레이션 ──────────────────────
print("\n[3] A->B 메타 필드 시뮬레이션")

def make_meta(pick_section, sel_a, sel_b, pick_x, pick_y, place_x, place_y):
    place_section = "B" if pick_section == "A" else "A"
    return {
        "pick_section": pick_section,
        "pick_x": round(pick_x, 3),
        "pick_y": round(pick_y, 3),
        "pick_anchor": list(sel_a) if pick_section == "A" and sel_a else
                       list(sel_b) if pick_section == "B" and sel_b else None,
        "place_section": place_section,
        "place_x": round(place_x, 3),
        "place_y": round(place_y, 3),
        "place_anchor": list(sel_b) if place_section == "B" and sel_b else
                        list(sel_a) if place_section == "A" and sel_a else None,
    }

# A->B: A4 pick → B2 place
sel_a = A_ANCHORS[1]   # A4
sel_b = B_ANCHORS[1]   # B2
meta_ab = make_meta("A", sel_a, sel_b,
                    pick_x=221.5, pick_y=-369.0,
                    place_x=31.2, place_y=-378.5)

check(meta_ab["pick_anchor"] == list(A_ANCHORS[1]),
      f"A->B pick_anchor = A4 {A_ANCHORS[1]}")
check(meta_ab["place_anchor"] == list(B_ANCHORS[1]),
      f"A->B place_anchor = B2 {B_ANCHORS[1]}")
check(meta_ab["pick_section"] == "A",  "A->B pick_section = A")
check(meta_ab["place_section"] == "B", "A->B place_section = B")

# ── 4. B->A 시나리오 시뮬레이션 ──────────────────────
print("\n[4] B->A 메타 필드 시뮬레이션")

# B->A: B3 pick → A3 place
sel_a = A_ANCHORS[0]   # A3
sel_b = B_ANCHORS[2]   # B3
meta_ba = make_meta("B", sel_a, sel_b,
                    pick_x=29.5, pick_y=-416.0,
                    place_x=218.0, place_y=-406.5)

check(meta_ba["pick_anchor"] == list(B_ANCHORS[2]),
      f"B->A pick_anchor = B3 {B_ANCHORS[2]}")
check(meta_ba["place_anchor"] == list(A_ANCHORS[0]),
      f"B->A place_anchor = A3 {A_ANCHORS[0]}")
check(meta_ba["pick_section"] == "B",  "B->A pick_section = B")
check(meta_ba["place_section"] == "A", "B->A place_section = A")

# ── 5. Phase1 모드 (앵커 None) ────────────────────────
print("\n[5] Phase1 모드 (앵커 미선택 → None)")
meta_ph1 = make_meta("A", None, None,
                     pick_x=156.2, pick_y=-378.4,
                     place_x=34.4, place_y=-377.5)
check(meta_ph1["pick_anchor"] is None,  "Phase1 pick_anchor = None")
check(meta_ph1["place_anchor"] is None, "Phase1 place_anchor = None")

# ── 6. JSON 직렬화 가능 확인 ──────────────────────────
print("\n[6] JSON 직렬화 확인")
for label, meta in [("A->B", meta_ab), ("B->A", meta_ba), ("Phase1", meta_ph1)]:
    try:
        s = json.dumps(meta)
        parsed = json.loads(s)
        check(parsed["pick_anchor"] == meta["pick_anchor"],
              f"{label} JSON 직렬화/역직렬화 일치")
    except Exception as e:
        check(False, f"{label} JSON 직렬화 실패: {e}")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*60)
if errors == 0:
    print("\033[92m  ALL PASS — Step 6 완료! 전체 수정 완료\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*60)
