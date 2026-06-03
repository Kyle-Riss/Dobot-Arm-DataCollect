#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_norm_stats.py
v10 수집 완료 후 norm_stats.json 계산 스크립트

Usage:
    python3 compute_norm_stats.py
    python3 compute_norm_stats.py --data-dir /path/to/episodes --output norm_stats.json

Action 타입: 절대 next-position (joint_angles[t+1]), 6-dim, gripper 별도
Observation: joint_angles[t] 6-dim + gripper[t] 1-dim
min_std=1e-3 적용 (j5/rx/ry/rz near-zero 보호)
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

DATA_DIR = "/media/billye6/새 볼륨/Dobot/2CAM-Orange-init"
MIN_STD  = 1e-3   # near-zero std 보호 (j5, rx, ry, rz)

# --------------------------------------------------------------------------
def _load_episode(ep_dir: Path):
    npy = ep_dir / "dataset.npy"
    if not npy.exists():
        return None
    data = np.load(npy, allow_pickle=True)
    if len(data) < 2:
        return None

    joints, tcp, grippers, modes = [], [], [], []
    for r in data:
        if "joint_angles" not in r or "tcp_pose" not in r:
            continue
        joints.append([float(v) for v in r["joint_angles"][:6]])
        tcp.append([float(v) for v in r["tcp_pose"][:6]])
        grippers.append(float(r.get("gripper_tooldo1", 0)))
        modes.append(int(r.get("robot_mode", 0)))

    if len(joints) < 2:
        return None

    # mode=9 에피소드는 학습 제외 (quality check에서 이미 걸러지지만 이중 확인)
    if any(m == 9 for m in modes):
        return None

    return {
        "joints":   np.array(joints,   dtype=np.float32),   # [T, 6]
        "tcp":      np.array(tcp,       dtype=np.float32),   # [T, 6]
        "grippers": np.array(grippers,  dtype=np.float32),   # [T]
    }

# --------------------------------------------------------------------------
def _stats(arr: np.ndarray, min_std: float = MIN_STD) -> dict:
    """mean/std/min/max/q01/q99, std floor at min_std."""
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  np.maximum(arr.std(axis=0), min_std).tolist(),
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "q01":  np.percentile(arr, 1,  axis=0).tolist(),
        "q99":  np.percentile(arr, 99, axis=0).tolist(),
    }

# --------------------------------------------------------------------------
def compute(data_dir: str) -> dict:
    data_dir = Path(data_dir)
    eps = sorted(d for d in data_dir.iterdir()
                 if d.is_dir() and d.name.isdigit())
    print(f"Found {len(eps)} episode folders in {data_dir}")

    obs_joints, obs_grippers = [], []   # observation.state
    act_joints, act_grippers = [], []   # action (absolute next-position)

    ok, skip = 0, 0
    for ep in eps:
        ep_data = _load_episode(ep)
        if ep_data is None:
            skip += 1
            continue

        J = ep_data["joints"]    # [T, 6]
        G = ep_data["grippers"]  # [T]

        # observation = current frame (all but last)
        obs_joints.append(J[:-1])
        obs_grippers.append(G[:-1])

        # action = absolute next joint position (frames 1..T)
        act_joints.append(J[1:])
        act_grippers.append(G[1:])
        ok += 1

    print(f"Loaded: {ok} episodes  Skipped (bad/mode9): {skip}")
    if ok == 0:
        print("ERROR: no valid episodes found")
        sys.exit(1)

    obs_J  = np.concatenate(obs_joints,   axis=0)   # [N, 6]
    obs_G  = np.concatenate(obs_grippers, axis=0).reshape(-1, 1)  # [N, 1]
    act_J  = np.concatenate(act_joints,   axis=0)   # [N, 6]
    act_G  = np.concatenate(act_grippers, axis=0).reshape(-1, 1)  # [N, 1]

    obs_all = np.concatenate([obs_J, obs_G], axis=1)  # [N, 7]
    act_all = np.concatenate([act_J, act_G], axis=1)  # [N, 7]

    dims = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    print(f"\n{'='*65}")
    print(f"{'dim':<10} {'obs_mean':>9} {'obs_std':>8} "
          f"{'act_mean':>9} {'act_std':>8} {'act_q01':>8} {'act_q99':>8}")
    print(f"{'-'*65}")
    obs_s = _stats(obs_all)
    act_s = _stats(act_all)
    for i, name in enumerate(dims):
        print(f"{name:<10} "
              f"{obs_s['mean'][i]:>9.3f} {obs_s['std'][i]:>8.4f} "
              f"{act_s['mean'][i]:>9.3f} {act_s['std'][i]:>8.4f} "
              f"{act_s['q01'][i]:>8.3f} {act_s['q99'][i]:>8.3f}")

    # delta 참고값 (학습 action이 delta인 경우 확인용)
    delta_J = np.concatenate(act_joints, axis=0) - np.concatenate(obs_joints, axis=0)
    print(f"\n=== Delta-action 참고 (j3 SNR 확인용) ===")
    print(f"{'dim':<10} {'mean':>9} {'std':>8} {'q01':>8} {'q99':>8} {'SNR=|mean|/std':>15}")
    for i, name in enumerate(["j1","j2","j3","j4","j5","j6"]):
        m = delta_J[:, i].mean()
        s = max(delta_J[:, i].std(), MIN_STD)
        q1 = np.percentile(delta_J[:, i], 1)
        q99 = np.percentile(delta_J[:, i], 99)
        snr = abs(m) / s
        print(f"{name:<10} {m:>9.4f} {s:>8.4f} {q1:>8.3f} {q99:>8.3f} {snr:>15.3f}")

    print(f"\nTotal frames: {len(obs_J)}")

    norm_stats = {
        "action": act_s,
        "observation.state": obs_s,
        "_meta": {
            "n_episodes": ok,
            "n_frames":   int(len(obs_J)),
            "action_type": "absolute_next_position",
            "action_dim":  7,
            "joint_names": dims,
            "min_std":     MIN_STD,
            "data_dir":    str(data_dir),
        },
    }
    return norm_stats

# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output",   default="norm_stats.json")
    args = parser.parse_args()

    ns = compute(args.data_dir)

    with open(args.output, "w") as f:
        json.dump(ns, f, indent=2)
    print(f"\nSaved → {args.output}")
