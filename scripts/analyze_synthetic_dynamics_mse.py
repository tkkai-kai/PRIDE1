"""One-step dynamics MSE: diffusion s' vs MuJoCo s' from the same (s, a).

Loads npz written by Workspace.save_synthetic_transitions (or snapshot
syn_* fields). Reconstructs qpos/qvel, set_state, steps the raw MuJoCo
env, then writes per-file npz plus a summary csv.

Gym v2 observations drop root x (qpos[0]). That coordinate is set to 0,
which does not change Walker2d / Hopper / HalfCheetah planar dynamics.
Walker2d and Hopper also clip qvel at ±10 in the observation, so the
restored velocity can differ from the true one.

Does not touch the training env. Run on a GPU node if mujoco_py needs EGL.

Example:
  python scripts/analyze_synthetic_dynamics_mse.py \\
      --npz-dir outputs/<run>/synthetic_transitions/Walker2d-v2 \\
      --env Walker2d-v2
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_mujoco_ld_path():
    """mujoco_py refuses to import unless mujoco210/bin is on LD_LIBRARY_PATH."""
    parts = []
    mujoco_bin = os.path.join(os.path.expanduser("~"), ".mujoco", "mujoco210", "bin")
    if os.path.isdir(mujoco_bin):
        parts.append(mujoco_bin)
    nvidia = "/usr/lib/nvidia"
    if os.path.isdir(nvidia):
        parts.append(nvidia)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    if parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


_ensure_mujoco_ld_path()
import utils


GYM_V2_ENVS = ("HalfCheetah-v2", "Walker2d-v2", "Hopper-v2")
QVEL_CLIP_ENVS = ("Walker2d-v2", "Hopper-v2")
QVEL_CLIP = 10.0


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return int(arr.reshape(-1)[0])


def _as_str(value) -> Optional[str]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_transitions(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "synthetic_obs" in data:
            obs = data["synthetic_obs"]
            actions = data["synthetic_actions"]
            next_obs = data["synthetic_next_obs"]
        elif "syn_obs" in data:
            obs = data["syn_obs"]
            actions = data["syn_actions"]
            next_obs = data["syn_next_obs"]
        else:
            raise KeyError(
                f"{path} has neither synthetic_obs nor syn_obs: {list(data.files)}"
            )
        payload = {
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "step": _as_int(data["step"]) if "step" in data else None,
            "seed": _as_int(data["seed"]) if "seed" in data else None,
            "env": _as_str(data["env"]) if "env" in data else None,
        }
    if not (payload["obs"].shape[0] == payload["actions"].shape[0]
            == payload["next_obs"].shape[0]):
        raise ValueError(f"{path}: obs/actions/next_obs row counts differ")
    return payload


def obs_to_qpos_qvel(
    env_id: str, obs: np.ndarray, nq: int, nv: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert the Gym v2 observation map. qpos[0] is not in obs; set to 0."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    qpos = np.zeros(nq, dtype=np.float64)
    qvel = np.zeros(nv, dtype=np.float64)
    qpos[1:] = obs[: nq - 1]
    qvel[:] = obs[nq - 1: nq - 1 + nv]
    expected = (nq - 1) + nv
    if obs.shape[0] != expected:
        raise ValueError(
            f"{env_id}: obs dim {obs.shape[0]} != (nq-1)+nv = {expected} "
            f"(nq={nq}, nv={nv})"
        )
    return qpos, qvel


def make_gym_env(env_id: str, seed: int):
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.env = env_id
    cfg.seed = int(seed)
    return utils.make_env(cfg)


def simulate_next_obs(raw_env, env_id: str, obs: np.ndarray, action: np.ndarray):
    """set_state from obs, one physics step, return (recon_obs, sim_next_obs)."""
    nq = int(raw_env.model.nq)
    nv = int(raw_env.model.nv)
    qpos, qvel = obs_to_qpos_qvel(env_id, obs, nq, nv)
    raw_env.set_state(qpos, qvel)
    recon = np.asarray(raw_env._get_obs(), dtype=np.float32)
    clipped = np.clip(action, raw_env.action_space.low, raw_env.action_space.high)
    raw_env.do_simulation(clipped, raw_env.frame_skip)
    sim_next = np.asarray(raw_env._get_obs(), dtype=np.float32)
    return recon, sim_next


def analyze_file(
    path: str,
    env_id: str,
    seed: int,
    raw_env,
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    payload = load_transitions(path)
    file_env = payload["env"] or env_id
    if file_env != env_id:
        raise ValueError(f"{path}: npz env={file_env} but --env={env_id}")

    n = payload["obs"].shape[0]
    if max_samples is not None and n > max_samples:
        take = rng.choice(n, size=int(max_samples), replace=False)
        take.sort()
        obs = payload["obs"][take]
        actions = payload["actions"][take]
        next_obs = payload["next_obs"][take]
    else:
        obs = payload["obs"]
        actions = payload["actions"]
        next_obs = payload["next_obs"]

    n = obs.shape[0]
    obs_dim = obs.shape[1]
    recon = np.full((n, obs_dim), np.nan, dtype=np.float32)
    sim_next = np.full((n, obs_dim), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        try:
            rec_i, sim_i = simulate_next_obs(raw_env, env_id, obs[i], actions[i])
        except Exception:
            continue
        if rec_i.shape[0] != obs_dim or sim_i.shape[0] != obs_dim:
            continue
        if not (np.isfinite(rec_i).all() and np.isfinite(sim_i).all()):
            continue
        recon[i] = rec_i
        sim_next[i] = sim_i
        valid[i] = True

    delta = sim_next - next_obs
    per_sample = np.mean(np.square(delta), axis=1)
    per_sample[~valid] = np.nan
    recon_err = np.mean(np.square(recon - obs), axis=1)
    recon_err[~valid] = np.nan

    n_valid = int(valid.sum())
    if n_valid > 0:
        mean_mse = float(np.nanmean(per_sample))
        median_mse = float(np.nanmedian(per_sample))
        per_dim = np.nanmean(np.square(delta[valid]), axis=0)
        mean_recon = float(np.nanmean(recon_err))
        p90_mse = float(np.nanpercentile(per_sample, 90))
        p95_mse = float(np.nanpercentile(per_sample, 95))
        p99_mse = float(np.nanpercentile(per_sample, 99))
    else:
        mean_mse = float("nan")
        median_mse = float("nan")
        per_dim = np.full(obs_dim, np.nan, dtype=np.float32)
        mean_recon = float("nan")

    return {
        "path": path,
        "step": payload["step"],
        "seed": payload["seed"] if payload["seed"] is not None else seed,
        "n": n,
        "n_valid": n_valid,
        "mean_mse": mean_mse,
        "median_mse": median_mse,
        "mean_recon_mse": mean_recon,
        "p90_mse": p90_mse,
        "p95_mse": p95_mse,
        "p99_mse": p99_mse,
        "per_sample_mse": per_sample.astype(np.float32),
        "per_dim_mse": per_dim.astype(np.float32),
        "recon_mse": recon_err.astype(np.float32),
        "valid": valid,
        "syn_next_obs": next_obs,
        "sim_next_obs": sim_next,
    }


def write_summary_csv(rows: List[Dict], dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fieldnames = [
        "npz",
        "step",
        "n",
        "n_valid",
        "valid_frac",
        "mean_mse",
        "median_mse",
        "mean_recon_mse",
        "p90_mse",
        "p95_mse",
        "p99_mse",
    ]
    with open(dest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            n = int(row["n"])
            n_valid = int(row["n_valid"])
            writer.writerow({
                "npz": os.path.basename(row["path"]),
                "step": "" if row["step"] is None else row["step"],
                "n": n,
                "n_valid": n_valid,
                "valid_frac": "" if n == 0 else f"{n_valid / n:.4f}",
                "mean_mse": row["mean_mse"],
                "median_mse": row["median_mse"],
                "mean_recon_mse": row["mean_recon_mse"],
                "p90_mse": row["p90_mse"],
                "p95_mse": row["p95_mse"],
                "p99_mse": row["p99_mse"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSE between diffusion next-obs and MuJoCo next-obs."
    )
    parser.add_argument(
        "--npz",
        action="append",
        default=[],
        help="One npz file. Repeat to pass several.",
    )
    parser.add_argument(
        "--npz-dir",
        default="",
        help="Directory of synthetic_transitions_step_*.npz (or snapshot_step_*.npz).",
    )
    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap samples per file (0 = use all). Useful for a smoke test.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Where to write summary csv and per-file mse npz. Default: <npz-dir>/dynamics_mse",
    )
    return parser.parse_args()


CHECKPOINT_NPZ = (
    "synthetic_transitions_step_100000.npz",
    "synthetic_transitions_step_300000.npz",
    "synthetic_transitions_step_500000.npz",
    "synthetic_transitions_step_700000.npz",
    "synthetic_transitions_step_900000.npz",
)


def collect_paths(args: argparse.Namespace) -> List[str]:
    paths = list(args.npz)
    if args.npz_dir:
        if os.path.isdir(args.npz_dir):
            paths.extend(
                os.path.join(args.npz_dir, name)
                for name in CHECKPOINT_NPZ
                if os.path.isfile(os.path.join(args.npz_dir, name))
            )
        elif os.path.isfile(args.npz_dir):
            paths.append(args.npz_dir)
    paths = [p for p in paths if not os.path.basename(p).endswith("_dynamics_mse.npz")]
    if not paths:
        raise SystemExit("No npz files. Pass --npz or --npz-dir.")
    return paths


def main() -> None:
    """
        选 100k / 300k / 500k / 700k / 900k。
        每个 checkpoint 随机取 10K synthetic transitions。
        同时取 10K real transitions 做 reconstruction reference。
        输出 mean / median / p90 / p95 / p99 mse。
        画 MSE probability-density/histogram（最好 log-x）。
    """
    args = parse_args()
    if args.env not in GYM_V2_ENVS:
        raise SystemExit(
            f"--env {args.env} is not a Gym v2 loco env; "
            f"obs->qpos reconstruction is only implemented for {GYM_V2_ENVS}."
        )

    paths = collect_paths(args)
    out_dir = args.out_dir
    if not out_dir:
        if args.npz_dir:
            out_dir = os.path.join(args.npz_dir, "dynamics_mse")
        else:
            out_dir = os.path.join(os.path.dirname(paths[0]) or ".", "dynamics_mse")
    os.makedirs(out_dir, exist_ok=True)

    env = make_gym_env(args.env, args.seed)
    raw_env = env.unwrapped
    rng = np.random.default_rng(int(args.seed) + 24680)
    max_samples = args.max_samples if args.max_samples > 0 else None

    print(f"env={args.env} nq={raw_env.model.nq} nv={raw_env.model.nv} "
          f"files={len(paths)}")
    if args.env in QVEL_CLIP_ENVS:
        print(f"note: {args.env} clips qvel at ±{QVEL_CLIP} in the observation")

    rows = []
    for path in paths:
        print(f"analyzing {path}")
        result = analyze_file(
            path, args.env, args.seed, raw_env, max_samples, rng
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(out_dir, f"{stem}_dynamics_mse.npz")
        np.savez_compressed(
            dest,
            step=np.array([-1 if result["step"] is None else result["step"]], dtype=np.int64),
            n=np.array([result["n"]], dtype=np.int64),
            n_valid=np.array([result["n_valid"]], dtype=np.int64),
            mean_mse=np.array([result["mean_mse"]], dtype=np.float32),
            median_mse=np.array([result["median_mse"]], dtype=np.float32),
            mean_recon_mse=np.array([result["mean_recon_mse"]], dtype=np.float32),
            per_sample_mse=result["per_sample_mse"],
            per_dim_mse=result["per_dim_mse"],
            recon_mse=result["recon_mse"],
            valid=result["valid"],
            syn_next_obs=result["syn_next_obs"],
            sim_next_obs=result["sim_next_obs"],
        )
        print(
            f"  valid={result['n_valid']}/{result['n']} "
            f"mean_mse={result['mean_mse']:.6f} "
            f"median_mse={result['median_mse']:.6f} "
            f"recon_mse={result['mean_recon_mse']:.6f}"
            f"p90_mse={result['p90_mse']:.6f}"
            f"p95_mse={result['p95_mse']:.6f}"
            f"p99_mse={result['p99_mse']:.6f}"
        )
        rows.append(result)

    summary_path = os.path.join(out_dir, "dynamics_mse_summary.csv")
    write_summary_csv(rows, summary_path)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
