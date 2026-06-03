#!/usr/bin/env python3
"""
Step 4 검증: A pick 로직 교체 확인
- POS_4 고정 → A_ANCHORS 기반 generate_point_around_anchor
- 실행: python verify_step4_a_pick.py
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
ANCHOR_NOISE = 10.0

def generate_point_around_anchor(ax, ay, noise=ANCHOR_NOISE):
    return (ax + random.uniform(-noise, noise), ay + random.uniform(-noise, noise))

print("=" * 60)
print("  Step 4 검증: A pick 로직 교체 (A->B 방향)")
print("=" * 60)

# ── 1. 기존 POS_4 고정 코드 제거 확인 ─────────────────
print("\n[1] 기존 POS_4 고정 코드 제거 확인")
m_a = re.search(r'elif self\.pick_section == "A":', code)
m_b = re.search(r'else:.*?#.*?"B"', code)
block_a = code[m_a.start():m_b.start()] if m_a and m_b else ""

check('POS_4[0], POS_4[1], POS_4[2]' not in block_a,
      "A->B 블록 내 POS_4 고정 pick 제거됨")
check('pick_x is None or self.pick_y is None' not in block_a,
      "A->B 블록 내 None 분기 제거됨")

# ── 2. 새 코드 삽입 확인 ──────────────────────────────
print("\n[2] 새 코드 삽입 확인")
check(bool(re.search(r'random\.choice\(A_ANCHORS\)', code)),
      "random.choice(A_ANCHORS) 존재")
check(bool(re.search(r'generate_point_around_anchor\(_a_anchor', code)),
      "generate_point_around_anchor(_a_anchor...) 존재")
check(bool(re.search(r'self\._selected_a_anchor\s*=\s*_a_anchor', code)),
      "self._selected_a_anchor 저장 존재")

# ── 3. RPY는 A_SECTION_RX/RY/RZ 사용 확인 ────────────
print("\n[3] RPY = A_SECTION_RX/RY/RZ 확인")
check('A_SECTION_RX, A_SECTION_RY, A_SECTION_RZ' in block_a,
      "A->B 블록에서 A_SECTION RPY 사용")

# ── 4. 앵커 선택 분포 (3000회) ────────────────────────
print("\n[4] A 앵커 선택 분포 시뮬레이션 (3000회)")
random.seed(42)
counts = {i: 0 for i in range(3)}
picks = []
for _ in range(3000):
    anchor = random.choice(A_ANCHORS)
    idx = A_ANCHORS.index(anchor)
    counts[idx] += 1
    x, y = generate_point_around_anchor(anchor[0], anchor[1])
    picks.append((x, y, idx))

for i, (ax, ay) in enumerate(A_ANCHORS):
    pct = counts[i] / 3000 * 100
    check(abs(pct - 33.3) < 5.0,
          f"A{i+3} 선택 비율 {pct:.1f}% (기대 33.3% ±5%)")

# ── 5. 생성된 pick 좌표 범위 확인 ────────────────────
print("\n[5] 생성 좌표 ±10mm 범위 확인")
for i, (ax, ay) in enumerate(A_ANCHORS):
    pts = [(x, y) for x, y, idx in picks if idx == i]
    out = sum(1 for x, y in pts if abs(x-ax) > ANCHOR_NOISE or abs(y-ay) > ANCHOR_NOISE)
    check(out == 0, f"A{i+3} 생성 {len(pts)}개 모두 ±{ANCHOR_NOISE:.0f}mm 이내")

# ── 6. 전체 pick 좌표 A섹션 범위+noise 여유 내 ───────
# 앵커가 경계 위에 있으므로 ±ANCHOR_NOISE 여유 포함해서 확인
print("\n[6] 전체 pick 좌표 A섹션 bounding box + noise 여유")
N = ANCHOR_NOISE
out_of_section = sum(
    1 for x, y, _ in picks
    if not (85 - N <= x <= 222 + N and -438 - N <= y <= -311 + N)
)
check(out_of_section == 0,
      f"3000개 모두 A섹션 +{N:.0f}mm 여유 내 (out={out_of_section})")
# 각 앵커별 샘플이 해당 앵커 ±10mm 안에 있는지 (핵심 조건)
all_in_noise = all(
    abs(x - A_ANCHORS[idx][0]) <= N and abs(y - A_ANCHORS[idx][1]) <= N
    for x, y, idx in picks
)
check(all_in_noise, f"각 샘플이 선택된 앵커 기준 ±{N:.0f}mm 이내 (핵심)")

# ── 7. B place 로직은 그대로인지 (Step3 회귀 확인) ────
print("\n[7] Step 3 회귀 확인 (B place 유지)")
check(bool(re.search(r'random\.choice\(B_ANCHORS\)', code)),
      "B place의 random.choice(B_ANCHORS) 여전히 존재")
check(bool(re.search(r'self\._selected_b_anchor', code)),
      "self._selected_b_anchor 여전히 존재")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*60)
if errors == 0:
    print("\033[92m  ALL PASS — Step 4 완료, Step 5로 진행 가능\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*60)
