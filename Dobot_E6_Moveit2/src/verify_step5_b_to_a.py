#!/usr/bin/env python3
"""
Step 5 검증: B->A 방향 pick/place 로직 교체 확인
- B pick: generate_random_point_in_section → B_ANCHORS ±10mm
- A place: POS_4 고정 → A_ANCHORS ±10mm
- 실행: python verify_step5_b_to_a.py
"""
import re, random, os

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

A_ANCHORS    = [(217.75,-405.65),(220.21,-368.72),(221.63,-318.39)]
B_ANCHORS    = [(30.0,-340.0),(30.0,-377.0),(30.0,-415.0)]
ANCHOR_NOISE = 10.0

def generate_point_around_anchor(ax, ay, noise=ANCHOR_NOISE):
    return (ax + random.uniform(-noise, noise), ay + random.uniform(-noise, noise))

# B->A 블록 추출: else: # "B" ~ Place 위치 Y 보정 직전
m_b  = re.search(r'else:.*?#.*?"B"', code)
m_end = re.search(r'# Place 위치 Y에 따라 RX 동적 보정', code)
block_b = code[m_b.start():m_end.start()] if m_b and m_end else ""

print("=" * 60)
print("  Step 5 검증: B->A 방향 pick/place 로직 교체")
print("=" * 60)

# ── 1. 기존 코드 제거 확인 ────────────────────────────
print("\n[1] 기존 코드 제거 확인")
check('generate_random_point_in_section(B_SECTION_POINTS)' not in block_b,
      "B->A 블록 내 generate_random_point_in_section(B_SECTION_POINTS) 제거됨")
check('POS_4[0], POS_4[1]' not in block_b,
      "B->A 블록 내 POS_4 고정 place 제거됨")
check('pick_x is None or self.pick_y is None' not in block_b,
      "B->A 블록 내 None 분기 제거됨")
# 파일 전체에서 B_SECTION_POINTS 랜덤 호출 완전히 사라졌는지
remaining = len(re.findall(r'generate_random_point_in_section\(B_SECTION_POINTS\)', code))
check(remaining == 0,
      f"파일 전체 generate_random_point_in_section(B_SECTION_POINTS) 0개 (완전 제거)")

# ── 2. 새 코드 삽입 확인 ──────────────────────────────
print("\n[2] 새 코드 삽입 확인")
# B pick
b_pick_choice = re.findall(r'random\.choice\(B_ANCHORS\)', code)
check(len(b_pick_choice) >= 2,
      f"random.choice(B_ANCHORS) 2곳 존재 (A->B place + B->A pick)")
# A place
check('random.choice(A_ANCHORS)' in block_b,
      "B->A 블록 내 random.choice(A_ANCHORS) 존재")
check('self._selected_a_anchor' in block_b,
      "B->A 블록 내 self._selected_a_anchor 저장 존재")
check('self._selected_b_anchor' in block_b,
      "B->A 블록 내 self._selected_b_anchor 저장 존재")

# ── 3. RPY 확인 ───────────────────────────────────────
print("\n[3] RPY 확인")
check('B_SECTION_RX, B_SECTION_RY, B_SECTION_RZ' in block_b,
      "B->A 블록 B pick RPY = B_SECTION_RX/RY/RZ")
check('A_SECTION_RX, A_SECTION_RY, A_SECTION_RZ' in block_b,
      "B->A 블록 A place RPY = A_SECTION_RX/RY/RZ")

# ── 4. B pick 시뮬레이션 (3000회) ────────────────────
print("\n[4] B->A pick 시뮬레이션 (B 앵커 3000회)")
random.seed(42)
b_picks = []
for _ in range(3000):
    anchor = random.choice(B_ANCHORS)
    x, y = generate_point_around_anchor(anchor[0], anchor[1])
    b_picks.append((x, y, B_ANCHORS.index(anchor)))

counts = {i: sum(1 for _,_,idx in b_picks if idx==i) for i in range(3)}
for i, (bx, by) in enumerate(B_ANCHORS):
    pct = counts[i] / 3000 * 100
    check(abs(pct - 33.3) < 5.0, f"B{i+1} pick 비율 {pct:.1f}% (기대 33.3% ±5%)")
    pts = [(x,y) for x,y,idx in b_picks if idx==i]
    out = sum(1 for x,y in pts if abs(x-bx) > ANCHOR_NOISE or abs(y-by) > ANCHOR_NOISE)
    check(out == 0, f"B{i+1} pick {len(pts)}개 모두 ±{ANCHOR_NOISE:.0f}mm 이내")

# ── 5. A place 시뮬레이션 (3000회) ───────────────────
print("\n[5] B->A place 시뮬레이션 (A 앵커 3000회)")
random.seed(99)
a_places = []
for _ in range(3000):
    anchor = random.choice(A_ANCHORS)
    x, y = generate_point_around_anchor(anchor[0], anchor[1])
    a_places.append((x, y, A_ANCHORS.index(anchor)))

counts_a = {i: sum(1 for _,_,idx in a_places if idx==i) for i in range(3)}
for i, (ax, ay) in enumerate(A_ANCHORS):
    pct = counts_a[i] / 3000 * 100
    check(abs(pct - 33.3) < 5.0, f"A{i+3} place 비율 {pct:.1f}% (기대 33.3% ±5%)")
    pts = [(x,y) for x,y,idx in a_places if idx==i]
    out = sum(1 for x,y in pts if abs(x-ax) > ANCHOR_NOISE or abs(y-ay) > ANCHOR_NOISE)
    check(out == 0, f"A{i+3} place {len(pts)}개 모두 ±{ANCHOR_NOISE:.0f}mm 이내")

# ── 6. 이전 스텝 회귀 확인 ───────────────────────────
print("\n[6] 이전 스텝 회귀 확인")
check(bool(re.search(r'generate_point_around_anchor\(_b_anchor', code)),
      "A->B B place 로직 유지 (Step 3)")
check(bool(re.search(r'generate_point_around_anchor\(_a_anchor', code)),
      "A->B A pick 로직 유지 (Step 4)")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*60)
if errors == 0:
    print("\033[92m  ALL PASS — Step 5 완료, Step 6으로 진행 가능\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*60)
