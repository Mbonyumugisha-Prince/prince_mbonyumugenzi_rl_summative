import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("RL_WARNINGS", "False")

import numpy as np  
import torch  
from tensordict import TensorDict  
from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import InventoryEnv 
from training import dqn_training, pg_training  

SUMMARY_FILE = ROOT / "logs" / "summary_best_runs.csv"
ASSETS = ROOT / "assets"

DEFAULT_BEST = {
    "dqn": "dqn_03_high_lr",
    "reinforce": "reinforce_03_high_lr",
    "ppo": "ppo_08_short_rollout",
    "a2c": "a2c_09_short_rollout",
}


def best_run_names():
    """Best run per algorithm, ranked by test-period reward."""
    if not SUMMARY_FILE.exists():
        return dict(DEFAULT_BEST), "ppo"

    names, scores = {}, {}
    with open(SUMMARY_FILE) as f:
        for row in csv.DictReader(f):
            algo = row["algo"].lower()
            names[algo] = row["name"]
            scores[algo] = float(row["eval_test_mean"])
    overall = max(scores, key=scores.get)
    return names, overall


def load_agent(algo, name):
    folder = "dqn" if algo == "dqn" else "pg"
    path = ROOT / "models" / folder / f"{name}.pt"
    checkpoint = torch.load(path, map_location="cpu")
    hidden = checkpoint["config"]["hidden"]

    spec_env = dqn_training.make_env("test") 
    if algo == "dqn":
        actor = dqn_training.make_actor(spec_env, hidden)
    else:
        actor = pg_training.make_actor(spec_env, hidden)
    spec_env.close()

    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    return actor, path


def choose_action(actor, obs):
    """Greedy action: highest Q-value (DQN) or most likely action (policy gradient)."""
    td = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[])
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = actor(td)
    return int(td["action"].argmax().item())


def print_intro(label, mode):
    line = "=" * 78
    print(line)
    print("STOCKMIND AI - Reinforcement learning inventory manager")
    print(line)
    print("Problem:   A store must decide every day how many units of a product to order.")
    print("           Orders arrive 3 days later. Demand is real Walmart sales (M5, store CA_1).")
    print("Objective: Maximise profit: sell as much as possible while keeping stock low.")
    print("Actions:   Order 0, 5, 10, 20, 40 or 80 units.")
    print("Reward:    +1 per unit sold   -0.05 per unit held per day   -2 per lost sale")
    print("           -1 per unit wasted (shelf full)   -3 per order placed")
    print(f"Agent:     {label}")
    print(f"Data:      {'unseen TEST period (last 20% of days)' if mode == 'test' else 'TRAINING period'}")
    print(line)


def run_episode(env, actor, seed, episode_no):
    obs, info = env.reset(seed=seed)
    print(f"\n--- Episode {episode_no}: product {info['item_id']}, starting stock {info['stock']:.0f} ---")

    total_reward = sold = lost = 0.0
    orders = 0
    stock_levels = []

    while True:
        action = env.action_space.sample() if actor is None else choose_action(actor, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        sold += info["sold"]
        lost += info["lost"]
        orders += int(info["order"] > 0)
        stock_levels.append(info["stock"])

        if terminated or truncated:
            break

    demand = sold + lost
    result = {
        "product": info["item_id"],
        "days": info["day"],
        "total_reward": total_reward,
        "fill_rate": 100 * sold / demand if demand else 100.0,
        "units_sold": sold,
        "lost_sales": lost,
        "avg_stock": float(np.mean(stock_levels)),
        "orders": orders,
        "ended_early": terminated,
    }

    print(f"\nEpisode {episode_no} result:")
    print(f"  Total reward:   {result['total_reward']:.1f}")
    print(f"  Fill rate:      {result['fill_rate']:.1f}% of customers served")
    print(f"  Units sold:     {result['units_sold']:.0f}   Lost sales: {result['lost_sales']:.0f}")
    print(f"  Average stock:  {result['avg_stock']:.1f} units   Orders placed: {result['orders']}")
    if result["ended_early"]:
        print("  Episode ended early: too many stock-out days in a row.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run a trained StockMind agent")
    parser.add_argument("--algo", default="best",
                        choices=["best", "dqn", "reinforce", "ppo", "a2c", "random"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mode", default="test", choices=["test", "train"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=6, help="days shown per second in the window")
    parser.add_argument("--no-render", action="store_true", help="terminal output only")
    args = parser.parse_args()

    names, overall_best = best_run_names()
    algo = overall_best if args.algo == "best" else args.algo

    if algo == "random":
        actor, label = None, "random agent (no learning)"
    else:
        actor, path = load_agent(algo, names[algo])
        label = f"{algo.upper()} - {names[algo]}  ({path.relative_to(ROOT)})"

    print_intro(label, args.mode)

    render_mode = None if args.no_render else "human"
    env = InventoryEnv(mode=args.mode, render_mode=render_mode)
    env.metadata["render_fps"] = args.fps

    results = []
    for ep in range(1, args.episodes + 1):
        results.append(run_episode(env, actor, seed=args.seed + ep, episode_no=ep))

        if env.renderer is not None:
            ASSETS.mkdir(exist_ok=True)
            shot = ASSETS / f"agent_{algo}_episode{ep}.png"
            env.renderer.save(shot)
            print(f"  Screenshot:     {shot.relative_to(ROOT)}")

    env.close()

    print("\n" + "=" * 78)
    print(f"SUMMARY over {len(results)} episodes ({label.split('  (')[0]})")
    print(f"  Mean reward:    {np.mean([r['total_reward'] for r in results]):.1f}")
    print(f"  Mean fill rate: {np.mean([r['fill_rate'] for r in results]):.1f}%")
    print(f"  Mean stock:     {np.mean([r['avg_stock'] for r in results]):.1f} units")
    print("=" * 78)


if __name__ == "__main__":
    main()