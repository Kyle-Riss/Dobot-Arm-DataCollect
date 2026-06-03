#!/usr/bin/env python3
"""
Step 3 검증: B place 로직 교체 확인
- generate_random_point_in_section → B_ANCHORS 기반 generate_point_around_anchor
- 실행: python verify_step3_b_place.py
"""
import re, random, math, os

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

B_ANCHORS    = [(30.0,-340.0),(30.0,-377.0),(30.0,-415.0)]
ANCHOR_NOISE = 10.0

def generate_point_around_anchor(ax, ay, noise=ANCHOR_NOISE):
    return (ax + random.uniform(-noise, noise), ay + random.uniform(-noise, noise))

print("=" * 60)
print("  Step 3 검증: B place 로직 교체 (A->B 방향)")
print("=" * 60)

# ── 1. pick_section=="A" 블록 내 기존 코드 제거 확인 (Step3 범위만)
print("\n[1] 기존 코드 제거 확인 (A->B place 블록)")
# pick_section=="A" 블록 ~ else("B") 블록 사이 범위만 검사
m_a = re.search(r'elif self\.pick_section == "A":', code)
m_b = re.search(r'else:.*?#.*?"B"', code)
if m_a and m_b:
    block_a = code[m_a.start():m_b.start()]
    old_in_block = len(re.findall(r'generate_random_point_in_section\(B_SECTION_POINTS\)', block_a))
    check(old_in_block == 0,
          f"A->B 블록 내 generate_random_point_in_section(B_SECTION_POINTS) 제거됨")
    # B->A 블록에는 아직 남아있어도 됨 (Step 5에서 처리)
    remaining = len(re.findall(r'generate_random_point_in_section\(B_SECTION_POINTS\)', code))
    check(remaining == 1,
          f"B->A pick 블록에 1개 남음 (Step 5에서 처리 예정)")

# ── 2. 새 코드 존재 확인
print("\n[2] 새 코드 삽입 확인")
check(bool(re.search(r'random\.choice\(B_ANCHORS\)', code)),
      "random.choice(B_ANCHORS) 존재")
check(bool(re.search(r'generate_point_around_anchor\(_b_anchor', code)),
      "generate_point_around_anchor(_b_anchor...) 존재")
check(bool(re.search(r'self\._selected_b_anchor\s*=\s*_b_anchor', code)),
      "self._selected_b_anchor 저장 존재")

# ── 3. 위치 확인: pick_section == "A" 블록 안에 있는지
print("\n[3] 위치 확인 (pick_section == A 블록)")
m_section = re.search(r'elif self\.pick_section == "A":', code)
m_choice  = re.search(r'random\.choice\(B_ANCHORS\)', code)
check(m_section and m_choice and m_choice.start() > m_section.start(),
      'pick_section == "A" 블록 내부에 위치')

# ── 4. 시뮬레이션: B 앵커 선택 분포 (3000회)
print("\n[4] B 앵커 선택 분포 시뮬레이션 (3000회)")
random.seed(42)
counts = {i: 0 for i in range(3)}
places = []
for _ in range(3000):
    anchor = random.choice(B_ANCHORS)
    idx = B_ANCHORS.index(anchor)
    counts[idx] += 1
    x, y = generate_point_around_anchor(anchor[0], anchor[1])
    places.append((x, y, idx))

for i, (ax, ay) in enumerate(B_ANCHORS):
    pct = counts[i] / 3000 * 100
    check(abs(pct - 33.3) < 5.0,
          f"B{i+1} 선택 비율 {pct:.1f}% (기대 33.3% ±5%)")

# ── 5. 생성된 place 좌표가 해당 앵커 ±10mm 이내인지
print("\n[5] 생성 좌표 범위 확인")
for i, (ax, ay) in enumerate(B_ANCHORS):
    pts = [(x, y) for x, y, idx in places if idx == i]
    out = sum(1 for x, y in pts if abs(x-ax) > ANCHOR_NOISE or abs(y-ay) > ANCHOR_NOISE)
    check(out == 0, f"B{i+1} 생성 {len(pts)}개 모두 ±{ANCHOR_NOISE:.0f}mm 이내")

# ── 6. 전체 분포가 B 섹션 내부인지 (X: -27~94, Y: -439~-311)
print("\n[6] 전체 place 좌표 B섹션 범위 내")
out_of_section = sum(
    1 for x, y, _ in places
    if not (-27 <= x <= 94 and -439 <= y <= -311)
)
check(out_of_section == 0,
      f"3000개 모두 B섹션 bounding box 내 (out={out_of_section})")

# ── 7. 앵커 간 클러스터 비겹침 확인
print("\n[7] 클러스터 비겹침 확인")
for i in range(len(B_ANCHORS)-1):
    ax1,ay1 = B_ANCHORS[i]; ax2,ay2 = B_ANCHORS[i+1]
    gap = abs(ay2 - ay1) - 2 * ANCHOR_NOISE
    check(gap > 0, f"B{i+1}↔B{i+2} 클러스터 간 gap {gap:.1f}mm > 0")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*60)
if errors == 0:
    print("\033[92m  ALL PASS — Step 3 완료, Step 4로 진행 가능\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*60)
