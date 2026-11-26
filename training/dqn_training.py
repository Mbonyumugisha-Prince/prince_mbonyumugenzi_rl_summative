import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.custom_env import InventoryEnv  

MODELS_DIR = ROOT / "models" / "dqn"
LOGS_DIR = ROOT / "logs" / "dqn"

REWARD_SCALE = 0.1  

BASE = {
    "lr": 1e-3,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 50_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.5,     
    "target_update": 500,     
    "hidden": [128, 128],
    "double_dqn": False,
    "frames_per_batch": 200,
    "updates_per_batch": 50,
    "init_random_frames": 2_000,
}

EXPERIMENTS = [
    {"name": "dqn_01_baseline"},
    {"name": "dqn_02_low_lr", "lr": 1e-4},
    {"name": "dqn_03_high_lr", "lr": 5e-3},
    {"name": "dqn_04_low_gamma", "gamma": 0.90},
    {"name": "dqn_05_big_batch", "batch_size": 256},
    {"name": "dqn_06_small_buffer", "buffer_size": 5_000},
    {"name": "dqn_07_fast_eps_decay", "eps_decay_frac": 0.1},
    {"name": "dqn_08_high_eps_end", "eps_end": 0.20},
    {"name": "dqn_09_fast_target", "target_update": 50},
    {"name": "dqn_10_double_dqn_wide", "double_dqn": True, "hidden": [256, 256]},
]


def get_config(index):
    cfg = dict(BASE)
    cfg.update(EXPERIMENTS[index])
    return cfg


def make_env(mode="train", seed=0):
    env = GymWrapper(InventoryEnv(mode=mode))
    env = TransformedEnv(
        env,
        Compose(
            RewardSum(),                           
            RewardScaling(loc=0.0, scale=REWARD_SCALE),
        ),
    )
    env.set_seed(seed)
    return env


def make_actor(env, hidden):
    n_obs = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.shape[-1]

    q_net = MLP(
        in_features=n_obs,
        out_features=n_actions,
        num_cells=hidden,
        activation_class=nn.ReLU,
    )
    return QValueActor(q_net, in_keys=["observation"], spec=env.action_spec)


def evaluate(actor, mode="test", episodes=20, seed=1234):
    env = make_env(mode, seed)
    rewards = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(episodes):
            td = env.rollout(max_steps=500, policy=actor)
            rewards.append(td["next", "episode_reward"][-1].item())

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_random(mode="test", episodes=20, seed=1234):
    env = make_env(mode, seed)
    rewards = []
    for _ in range(episodes):
        td = env.rollout(max_steps=500)  
        rewards.append(td["next", "episode_reward"][-1].item())
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def train(cfg, total_frames, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env("train", seed)
    actor = make_actor(env, cfg["hidden"])

    explore = EGreedyModule(
        env.action_spec,
        eps_init=cfg["eps_start"],
        eps_end=cfg["eps_end"],
        annealing_num_steps=int(cfg["eps_decay_frac"] * total_frames),
    )
    policy = TensorDictSequential(actor, explore)

    collector = Collector(
        env,
        policy,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=total_frames,
        init_random_frames=cfg["init_random_frames"],
        auto_register_policy_transforms=False,
    )

    buffer = ReplayBuffer(
        storage=LazyTensorStorage(cfg["buffer_size"]),
        batch_size=cfg["batch_size"],
    )

    loss_fn = DQNLoss(
        actor,
        action_space=env.action_spec,
        delay_value=True,
        double_dqn=cfg["double_dqn"],
    )
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    target_updater = HardUpdate(loss_fn, value_network_update_interval=cfg["target_update"])
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    history = {"frames": [], "episode_reward": [], "loss": [], "epsilon": []}
    episode_rewards = []
    frames = 0
    start = time.time()

    for data in collector:
        frames += data.numel()
        buffer.extend(data.reshape(-1))

        done = data["next", "done"].squeeze(-1)
        if done.any():
            episode_rewards += data["next", "episode_reward"][done].squeeze(-1).tolist()

        losses = []
        if frames >= cfg["init_random_frames"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                loss = loss_fn(sample)["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optimizer.step()
                target_updater.step()

                losses.append(loss.item())

            explore.step(data.numel())

        collector.update_policy_weights_()

        recent = float(np.mean(episode_rewards[-20:])) if episode_rewards else float("nan")
        history["frames"].append(frames)
        history["episode_reward"].append(recent)
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["epsilon"].append(float(explore.eps))

        if len(history["frames"]) % 25 == 0:
            print(
                f"  frames {frames:>7,} | avg reward (last 20 eps) {recent:>9.1f} | "
                f"loss {history['loss'][-1]:>8.4f} | eps {explore.eps.item():.3f} | "
                f"{time.time() - start:>5.0f}s"
            )

    collector.shutdown()
    env.close()

    return actor, history, episode_rewards, time.time() - start


def run_experiment(index, total_frames, seed=0):
    cfg = get_config(index)
    name = cfg["name"]

    print(f"\n=== {name} ===")
    print("  " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "name"))

    actor, history, episode_rewards, seconds = train(cfg, total_frames, seed)

    train_mean, train_std = evaluate(actor, mode="train")
    test_mean, test_std = evaluate(actor, mode="test")

    print(f"  Eval on TRAIN period: {train_mean:.1f} ± {train_std:.1f}")
    print(f"  Eval on TEST period:  {test_mean:.1f} ± {test_std:.1f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": actor.state_dict(), "config": cfg},
        MODELS_DIR / f"{name}.pt",
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / f"{name}.json", "w") as f:
        json.dump(
            {"config": cfg, "history": history, "episode_rewards": episode_rewards},
            f,
        )
        
    summary = {
        "name": name,
        **{k: v for k, v in cfg.items() if k != "name"},
        "total_frames": total_frames,
        "episodes": len(episode_rewards),
        "final_train_reward": round(float(np.mean(episode_rewards[-20:])), 1),
        "eval_train_mean": round(train_mean, 1),
        "eval_test_mean": round(test_mean, 1),
        "eval_test_std": round(test_std, 1),
        "minutes": round(seconds / 60, 1),
    }
    results_file = LOGS_DIR / "results.csv"
    new_file = not results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train DQN on the StockMind inventory environment")
    parser.add_argument("--runs", nargs="+", default=["0"], help="experiment numbers (0-9) or 'all'")
    parser.add_argument("--frames", type=int, default=100_000, help="environment steps per run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.runs == ["all"]:
        indices = list(range(len(EXPERIMENTS)))
    else:
        indices = [int(r) for r in args.runs]

    random_mean, random_std = evaluate_random(mode="test")
    print(f"Random agent on TEST period: {random_mean:.1f} ± {random_std:.1f}")

    results = [run_experiment(i, args.frames, args.seed) for i in indices]

    print("\n=== SUMMARY (test period, greedy policy) ===")
    print(f"  {'random_agent':<26} {random_mean:>9.1f}")
    for r in sorted(results, key=lambda r: r["eval_test_mean"], reverse=True):
        print(f"  {r['name']:<26} {r['eval_test_mean']:>9.1f}")


if __name__ == "__main__":
    main()