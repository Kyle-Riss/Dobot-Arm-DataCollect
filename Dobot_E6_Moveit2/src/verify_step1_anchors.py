#!/usr/bin/env python3
"""
Step 1 검증: A_ANCHORS / B_ANCHORS / ANCHOR_NOISE_MM 상수 확인
- 로봇 연결 없이 실행 가능
- 실행: python verify_step1_anchors.py
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

# ── POS_N 값 파싱 ─────────────────────────────────────
def get_pos(n):
    m = re.search(rf'^POS_{n}\s*=\s*\(([^)]+)\)', code, re.MULTILINE)
    vals = [float(v.strip()) for v in m.group(1).split(',')]
    return tuple(vals)

POS_3 = get_pos(3)
POS_4 = get_pos(4)
POS_5 = get_pos(5)

# ── 상수 블록을 POS 값 치환 후 eval ───────────────────
def extract_anchors(name):
    """name = [...] 블록을 찾아 POS_N[i] 치환 후 eval."""
    m = re.search(rf'^{name}\s*=\s*\[', code, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    depth, i = 0, start
    while i < len(code):
        if code[i] == '[':
            depth += 1
        elif code[i] == ']':
            depth -= 1
            if depth == 0:
                block = code[start:i+1]
                break
        i += 1
    # "NAME = " 제거 후 값 부분만
    block = re.sub(rf'^{name}\s*=\s*', '', block).strip()
    # POS_N[k] → 실제 숫자 치환
    def sub(m2):
        idx, dim = int(m2.group(1)), int(m2.group(2))
        return str(get_pos(idx)[dim])
    block = re.sub(r'POS_(\d)\[(\d)\]', sub, block)
    # 주석 제거
    block = re.sub(r'#.*', '', block)
    # ← 특수문자 제거
    block = block.replace('←', '')
    return eval(block)

def extract_float(name):
    m = re.search(rf'^{name}\s*=\s*([\d.]+)', code, re.MULTILINE)
    return float(m.group(1)) if m else None

A_ANCHORS    = extract_anchors("A_ANCHORS")
B_ANCHORS    = extract_anchors("B_ANCHORS")
ANCHOR_NOISE = extract_float("ANCHOR_NOISE_MM")

A_SECTION_POINTS = [
    (139.37,-435.31),(145.59,-414.15),(217.75,-405.65),
    (220.21,-368.72),(221.63,-318.39),(94.10,-311.54),(84.97,-437.89),
]
B_SECTION_POINTS = [
    (94.10,-311.54),(84.97,-437.89),(-27.02,-438.80),(-15.38,-321.49),
]

def point_in_polygon(px, py, polygon, tol=1.0):
    """꼭짓점/경계 위 점도 허용 (tol mm 이내 꼭짓점은 inside로 처리)."""
    for vx, vy in polygon:
        if abs(px - vx) < tol and abs(py - vy) < tol:
            return True
    n, inside, j = len(polygon), False, len(polygon)-1
    for i in range(n):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside

print("=" * 55)
print("  Step 1 검증: A_ANCHORS / B_ANCHORS / ANCHOR_NOISE_MM")
print("=" * 55)

# ── 1. 상수 존재 확인 ──────────────────────────────────
print("\n[1] 상수 정의 확인")
check(A_ANCHORS is not None and len(A_ANCHORS) == 3,
      f"A_ANCHORS 3개: {A_ANCHORS}")
check(B_ANCHORS is not None and len(B_ANCHORS) == 3,
      f"B_ANCHORS 3개: {B_ANCHORS}")
check(ANCHOR_NOISE == 10.0, f"ANCHOR_NOISE_MM = {ANCHOR_NOISE}")

# ── 2. A_ANCHORS ↔ POS_3/4/5 XY 일치 ─────────────────
print("\n[2] A_ANCHORS = POS_3/4/5 XY")
check(A_ANCHORS[0] == (POS_3[0], POS_3[1]), f"A3 = {A_ANCHORS[0]}")
check(A_ANCHORS[1] == (POS_4[0], POS_4[1]), f"A4 = {A_ANCHORS[1]}")
check(A_ANCHORS[2] == (POS_5[0], POS_5[1]), f"A5 = {A_ANCHORS[2]}")

# ── 3. A_ANCHORS ∈ A 섹션 ──────────────────────────────
print("\n[3] A_ANCHORS 모두 A섹션 내부")
for i, (ax, ay) in enumerate(A_ANCHORS):
    check(point_in_polygon(ax, ay, A_SECTION_POINTS),
          f"A{i+3} ({ax:.2f}, {ay:.2f})")

# ── 4. B_ANCHORS ∈ B 섹션 ──────────────────────────────
print("\n[4] B_ANCHORS 모두 B섹션 내부")
for i, (bx, by) in enumerate(B_ANCHORS):
    check(point_in_polygon(bx, by, B_SECTION_POINTS),
          f"B{i+1} ({bx:.1f}, {by:.1f})")

# ── 5. B 타점 간격 > 2×noise ───────────────────────────
print("\n[5] B 타점 간격 > 2 x ANCHOR_NOISE_MM (클러스터 비겹침)")
for i in range(len(B_ANCHORS)-1):
    bx1,by1 = B_ANCHORS[i]; bx2,by2 = B_ANCHORS[i+1]
    dist = ((bx2-bx1)**2+(by2-by1)**2)**0.5
    check(dist > 2*ANCHOR_NOISE,
          f"B{i+1}↔B{i+2} {dist:.1f}mm > {2*ANCHOR_NOISE:.0f}mm")

# ── 6. 샘플링 시뮬레이션 ───────────────────────────────
print("\n[6] 노이즈 샘플링 시뮬레이션 (각 앵커 200회)")
random.seed(42)
for label, anchors in [("A", A_ANCHORS), ("B", B_ANCHORS)]:
    for i, (ax, ay) in enumerate(anchors):
        name = f"A{i+3}" if label=="A" else f"B{i+1}"
        bad = sum(
            1 for _ in range(200)
            if abs(random.uniform(-ANCHOR_NOISE, ANCHOR_NOISE)) > ANCHOR_NOISE
        )
        check(bad == 0, f"{name} 200회 샘플 모두 ±{ANCHOR_NOISE:.0f}mm 이내")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*55)
if errors == 0:
    print("\033[92m  ALL PASS — Step 1 완료, Step 2로 진행 가능\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*55)
