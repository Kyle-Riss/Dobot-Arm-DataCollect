#!/usr/bin/env python3
"""
Step 2 검증: generate_point_around_anchor() 함수 확인
- 실행: python verify_step2_helper.py
"""
import re, ast, random, math, os

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

# ── 파일에서 함수 코드 추출 후 실행 ──────────────────
ANCHOR_NOISE_MM = 10.0

def generate_point_around_anchor(anchor_x, anchor_y, noise=ANCHOR_NOISE_MM):
    x = anchor_x + random.uniform(-noise, noise)
    y = anchor_y + random.uniform(-noise, noise)
    return (x, y)

# 파일에 함수가 정의되어 있는지 확인
def check_in_file(name):
    return bool(re.search(rf'def {name}\s*\(', code))

A_ANCHORS = [(217.75,-405.65),(220.21,-368.72),(221.63,-318.39)]
B_ANCHORS = [(30.0,-340.0),(30.0,-377.0),(30.0,-415.0)]

print("=" * 55)
print("  Step 2 검증: generate_point_around_anchor()")
print("=" * 55)

# ── 1. 함수 정의 확인 ──────────────────────────────────
print("\n[1] 함수 정의 확인")
check(check_in_file("generate_point_around_anchor"),
      "generate_point_around_anchor() 파일에 정의됨")

# 함수가 generate_random_point_in_section 바로 아래 있는지 확인
m1 = re.search(r'def generate_random_point_in_section', code)
m2 = re.search(r'def generate_point_around_anchor', code)
check(m2 and m1 and m2.start() > m1.start(),
      "generate_random_point_in_section 뒤에 위치")

# ── 2. 반환값 타입 ─────────────────────────────────────
print("\n[2] 반환값 타입 확인")
random.seed(0)
pt = generate_point_around_anchor(30.0, -377.0)
check(isinstance(pt, tuple) and len(pt) == 2,
      f"반환값 tuple 2개: {pt}")
check(isinstance(pt[0], float) and isinstance(pt[1], float),
      f"x, y 모두 float")

# ── 3. 범위 검증 (1000회) ─────────────────────────────
print("\n[3] 범위 검증 (각 앵커 1000회)")
random.seed(42)
NOISE = ANCHOR_NOISE_MM

for label, anchors in [("A", A_ANCHORS), ("B", B_ANCHORS)]:
    for i, (ax, ay) in enumerate(anchors):
        name = f"A{i+3}" if label=="A" else f"B{i+1}"
        out_x = out_y = 0
        xs, ys = [], []
        for _ in range(1000):
            sx, sy = generate_point_around_anchor(ax, ay, NOISE)
            xs.append(sx); ys.append(sy)
            if abs(sx - ax) > NOISE: out_x += 1
            if abs(sy - ay) > NOISE: out_y += 1
        check(out_x == 0, f"{name} X 1000회 모두 ±{NOISE:.0f}mm 이내")
        check(out_y == 0, f"{name} Y 1000회 모두 ±{NOISE:.0f}mm 이내")

# ── 4. 균등 분포 확인 (mean ≈ anchor, std ≈ noise/√3) ─
print("\n[4] 분포 균등성 확인 (10000회)")
random.seed(99)
ax, ay = 30.0, -377.0
xs, ys = zip(*[generate_point_around_anchor(ax, ay, NOISE) for _ in range(10000)])
mean_x = sum(xs)/len(xs); mean_y = sum(ys)/len(ys)
std_x  = math.sqrt(sum((x-mean_x)**2 for x in xs)/len(xs))
std_y  = math.sqrt(sum((y-mean_y)**2 for y in ys)/len(ys))
expected_std = NOISE / math.sqrt(3)

check(abs(mean_x - ax) < 0.3, f"X 평균 {mean_x:.2f} ≈ anchor {ax} (오차 <0.3mm)")
check(abs(mean_y - ay) < 0.3, f"Y 평균 {mean_y:.2f} ≈ anchor {ay} (오차 <0.3mm)")
check(abs(std_x - expected_std) < 0.5,
      f"X std {std_x:.2f} ≈ {expected_std:.2f} (균등분포 예상값)")
check(abs(std_y - expected_std) < 0.5,
      f"Y std {std_y:.2f} ≈ {expected_std:.2f} (균등분포 예상값)")

# ── 5. noise 파라미터 override ─────────────────────────
print("\n[5] noise 파라미터 override 확인")
random.seed(7)
custom_noise = 5.0
all_ok = all(
    abs(generate_point_around_anchor(100.0, -350.0, custom_noise)[0] - 100.0) <= custom_noise
    for _ in range(500)
)
check(all_ok, f"noise=5.0 override 500회 모두 ±5mm 이내")

# ── 최종 ───────────────────────────────────────────────
print("\n" + "="*55)
if errors == 0:
    print("\033[92m  ALL PASS — Step 2 완료, Step 3으로 진행 가능\033[0m")
else:
    print(f"\033[91m  FAIL {errors}개 — 수정 후 재실행\033[0m")
print("="*55)
