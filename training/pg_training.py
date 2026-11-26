import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RL_WARNINGS", "False") 

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dqn_training import evaluate, evaluate_random, make_env  

MODELS_DIR = ROOT / "models" / "pg"
LOGS_DIR = ROOT / "logs" / "pg"

BASE = {
    "reinforce": {
        "lr": 1e-3,
        "gamma": 0.99,
        "frames_per_batch": 900,     
        "baseline": "normalize",     
        "entropy_coeff": 0.01,
        "hidden": [64, 64],
        "max_grad_norm": 1.0,
    },
    "ppo": {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "critic_coeff": 0.5,
        "frames_per_batch": 2048,   
        "epochs": 10,
        "minibatch_size": 256,
        "hidden": [64, 64],
        "max_grad_norm": 0.5,
    },
    "a2c": {
        "lr": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coeff": 0.01,
        "critic_coeff": 0.5,
        "frames_per_batch": 128,     
        "hidden": [64, 64],
        "max_grad_norm": 0.5,
    },
}

EXPERIMENTS = {
    "reinforce": [
        {"name": "reinforce_01_baseline"},
        {"name": "reinforce_02_low_lr", "lr": 1e-4},
        {"name": "reinforce_03_high_lr", "lr": 5e-3},
        {"name": "reinforce_04_low_gamma", "gamma": 0.90},
        {"name": "reinforce_05_no_baseline", "baseline": "none"},
        {"name": "reinforce_06_no_entropy", "entropy_coeff": 0.0},
        {"name": "reinforce_07_high_entropy", "entropy_coeff": 0.05},
        {"name": "reinforce_08_small_batch", "frames_per_batch": 450},
        {"name": "reinforce_09_large_batch", "frames_per_batch": 2700},
        {"name": "reinforce_10_wide_net", "hidden": [256, 256]},
    ],
    "ppo": [
        {"name": "ppo_01_baseline"},
        {"name": "ppo_02_low_lr", "lr": 1e-4},
        {"name": "ppo_03_high_lr", "lr": 1e-3},
        {"name": "ppo_04_small_clip", "clip_epsilon": 0.1},
        {"name": "ppo_05_large_clip", "clip_epsilon": 0.3},
        {"name": "ppo_06_no_entropy", "entropy_coeff": 0.0},
        {"name": "ppo_07_high_entropy", "entropy_coeff": 0.05},
        {"name": "ppo_08_short_rollout", "frames_per_batch": 512},
        {"name": "ppo_09_low_lambda", "gae_lambda": 0.80},
        {"name": "ppo_10_few_epochs", "epochs": 4},
    ],
    "a2c": [
        {"name": "a2c_01_baseline"},
        {"name": "a2c_02_low_lr", "lr": 1e-4},
        {"name": "a2c_03_high_lr", "lr": 3e-3},
        {"name": "a2c_04_low_gamma", "gamma": 0.90},
        {"name": "a2c_05_no_entropy", "entropy_coeff": 0.0},
        {"name": "a2c_06_high_entropy", "entropy_coeff": 0.05},
        {"name": "a2c_07_low_critic", "critic_coeff": 0.25},
        {"name": "a2c_08_high_critic", "critic_coeff": 1.0},
        {"name": "a2c_09_short_rollout", "frames_per_batch": 32},
        {"name": "a2c_10_lambda_one", "gae_lambda": 1.0},
    ],
}


def get_config(algo, index):
    cfg = dict(BASE[algo])
    cfg.update(EXPERIMENTS[algo][index])
    cfg["algo"] = algo
    return cfg


def make_actor(env, hidden):
    n_obs = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.shape[-1]

    policy_net = MLP(
        in_features=n_obs,
        out_features=n_actions,
        num_cells=hidden,
        activation_class=nn.Tanh,
    )
    return ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )


def make_critic(env, hidden):
    n_obs = env.observation_spec["observation"].shape[-1]
    value_net = MLP(
        in_features=n_obs,
        out_features=1,
        num_cells=hidden,
        activation_class=nn.Tanh,
    )
    return ValueOperator(value_net, in_keys=["observation"])


def make_collector(env, actor, cfg, total_frames):
    return Collector(
        env,
        actor,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=total_frames,
        auto_register_policy_transforms=False,
    )


def finished_episode_rewards(data):
    """Real (unscaled) rewards of episodes that finished in this batch."""
    done = data["next", "done"].squeeze(-1)
    if done.any():
        return data["next", "episode_reward"][done].squeeze(-1).tolist()
    return []


def log_progress(history, frames, recent, start):
    if len(history["frames"]) % 10 == 0:
        print(
            f"  frames {frames:>7,} | avg reward (last 20 eps) {recent:>9.1f} | "
            f"loss {history['loss'][-1]:>8.4f} | entropy {history['entropy'][-1]:.3f} | "
            f"{time.time() - start:>5.0f}s"
        )

def train_reinforce(cfg, total_frames, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env("train", seed)
    actor = make_actor(env, cfg["hidden"])
    collector = make_collector(env, actor, cfg, total_frames)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])

    history = {"frames": [], "episode_reward": [], "loss": [], "entropy": [], "value_loss": []}
    episode_rewards = []
    frames = 0
    start = time.time()

    for data in collector:
        frames += data.numel()
        episode_rewards += finished_episode_rewards(data)

        rewards = data["next", "reward"].squeeze(-1)
        dones = data["next", "done"].squeeze(-1)

        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running = 0.0
            running = rewards[t] + cfg["gamma"] * running
            returns[t] = running

        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) == 0:
            continue
        valid = torch.arange(len(rewards)) <= done_idx[-1]
        batch = data[valid]
        returns = returns[valid]

        if cfg["baseline"] == "normalize":
            advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantage = returns

        dist = actor.get_dist(batch)
        log_prob = dist.log_prob(batch["action"])
        entropy = dist.entropy().mean()

        loss = -(log_prob * advantage).mean() - cfg["entropy_coeff"] * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
        optimizer.step()

        collector.update_policy_weights_()

        recent = float(np.mean(episode_rewards[-20:])) if episode_rewards else float("nan")
        history["frames"].append(frames)
        history["episode_reward"].append(recent)
        history["loss"].append(loss.item())
        history["entropy"].append(entropy.item())
        history["value_loss"].append(float("nan"))  
        log_progress(history, frames, recent, start)

    collector.shutdown()
    env.close()
    return actor, history, episode_rewards, time.time() - start


def train_actor_critic(cfg, total_frames, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env("train", seed)
    actor = make_actor(env, cfg["hidden"])
    critic = make_critic(env, cfg["hidden"])
    collector = make_collector(env, actor, cfg, total_frames)

    advantage_module = GAE(
        gamma=cfg["gamma"],
        lmbda=cfg["gae_lambda"],
        value_network=critic,
        average_gae=False,
    )

    if cfg["algo"] == "ppo":
        loss_fn = ClipPPOLoss(
            actor,
            critic,
            clip_epsilon=cfg["clip_epsilon"],
            entropy_coeff=cfg["entropy_coeff"],
            critic_coeff=cfg["critic_coeff"],
            normalize_advantage=True,
        )
        epochs = cfg["epochs"]
        minibatch_size = cfg["minibatch_size"]
    else:  
        loss_fn = A2CLoss(
            actor,
            critic,
            entropy_coeff=cfg["entropy_coeff"],
            critic_coeff=cfg["critic_coeff"],
        )
        epochs = 1
        minibatch_size = cfg["frames_per_batch"]

    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    history = {"frames": [], "episode_reward": [], "loss": [], "entropy": [], "value_loss": []}
    episode_rewards = []
    frames = 0
    start = time.time()

    for data in collector:
        frames += data.numel()
        episode_rewards += finished_episode_rewards(data)

        with torch.no_grad():
            advantage_module(data)

        flat = data.reshape(-1)
        n = flat.numel()
        losses, entropies, value_losses = [], [], []

        for _ in range(epochs):
            order = torch.randperm(n)
            for i in range(0, n, minibatch_size):
                minibatch = flat[order[i:i + minibatch_size]]
                out = loss_fn(minibatch)

                loss = sum(v for k, v in out.items() if k.startswith("loss_"))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["max_grad_norm"])
                optimizer.step()

                losses.append(loss.item())
                value_losses.append(out["loss_critic"].item())

                if "entropy" in out.keys():
                    entropies.append(out["entropy"].mean().item())
                else:
                    with torch.no_grad():
                        entropies.append(actor.get_dist(minibatch).entropy().mean().item())

        collector.update_policy_weights_()

        recent = float(np.mean(episode_rewards[-20:])) if episode_rewards else float("nan")
        history["frames"].append(frames)
        history["episode_reward"].append(recent)
        history["loss"].append(float(np.mean(losses)))
        history["entropy"].append(float(np.mean(entropies)))
        history["value_loss"].append(float(np.mean(value_losses)))
        log_progress(history, frames, recent, start)

    collector.shutdown()
    env.close()
    return actor, history, episode_rewards, time.time() - start


def evaluate_stochastic(actor, mode="test", episodes=20, seed=1234):
    env = make_env(mode, seed)
    rewards = []

    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        for _ in range(episodes):
            td = env.rollout(max_steps=500, policy=actor)
            rewards.append(td["next", "episode_reward"][-1].item())

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ----------------------------------------------------------------------
# Run one experiment and save everything
# ----------------------------------------------------------------------
def run_experiment(algo, index, total_frames, seed=0):
    cfg = get_config(algo, index)
    name = cfg["name"]

    print(f"\n=== {name} ===")
    print("  " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in ("name", "algo")))

    if algo == "reinforce":
        actor, history, episode_rewards, seconds = train_reinforce(cfg, total_frames, seed)
    else:
        actor, history, episode_rewards, seconds = train_actor_critic(cfg, total_frames, seed)

    train_mean, train_std = evaluate(actor, mode="train")
    test_mean, test_std = evaluate(actor, mode="test")
    stoch_mean, stoch_std = evaluate_stochastic(actor, mode="test")

    print(f"  Eval on TRAIN period (greedy):     {train_mean:.1f} ± {train_std:.1f}")
    print(f"  Eval on TEST period  (greedy):     {test_mean:.1f} ± {test_std:.1f}")
    print(f"  Eval on TEST period  (stochastic): {stoch_mean:.1f} ± {stoch_std:.1f}")

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
        **{k: v for k, v in cfg.items() if k not in ("name", "algo")},
        "total_frames": total_frames,
        "episodes": len(episode_rewards),
        "final_train_reward": round(float(np.mean(episode_rewards[-20:])), 1),
        "final_entropy": round(history["entropy"][-1], 3),
        "eval_train_mean": round(train_mean, 1),
        "eval_test_mean": round(test_mean, 1),
        "eval_test_std": round(test_std, 1),
        "eval_test_stochastic_mean": round(stoch_mean, 1),
        "eval_test_stochastic_std": round(stoch_std, 1),
        "minutes": round(seconds / 60, 1),
    }
    results_file = LOGS_DIR / f"results_{algo}.csv"
    new_file = not results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE / PPO / A2C on the StockMind environment")
    parser.add_argument("--algo", default="ppo", choices=["reinforce", "ppo", "a2c", "all"])
    parser.add_argument("--runs", nargs="+", default=["0"], help="experiment numbers (0-9) or 'all'")
    parser.add_argument("--frames", type=int, default=100_000, help="environment steps per run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]

    random_mean, _ = evaluate_random(mode="test")
    print(f"Random agent on TEST period: {random_mean:.1f}")

    results = []
    for algo in algos:
        if args.runs == ["all"]:
            indices = list(range(len(EXPERIMENTS[algo])))
        else:
            indices = [int(r) for r in args.runs]
        for i in indices:
            results.append(run_experiment(algo, i, args.frames, args.seed))

    print("\n=== SUMMARY (test period) ===")
    print(f"  {'agent':<28} {'greedy':>9} {'stochastic':>11}")
    print(f"  {'random_agent':<28} {random_mean:>9.1f}")
    for r in sorted(results, key=lambda r: r["eval_test_mean"], reverse=True):
        print(f"  {r['name']:<28} {r['eval_test_mean']:>9.1f} {r['eval_test_stochastic_mean']:>11.1f}")


if __name__ == "__main__":
    main()