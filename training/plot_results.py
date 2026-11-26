import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 

import matplotlib.pyplot as plt  
import matplotlib.ticker  
import numpy as np  
import pandas as pd 

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "assets" / "plots"

RANDOM_BASELINE = -2100  

COLORS = {
    "DQN": "#257de0",      
    "REINFORCE": "#f47230", 
    "PPO": "#00b894",        
    "A2C": "#f2a600",       
}
TEXT = "#333333"
MUTED = "#777777"
GRID = "#e6e6e6"
BASELINE_GREY = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "axes.axisbelow": True,
})


def k_steps(ax, axis="x"):
    """Show 20000 as 20k on a step axis."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def load_results():
    tables = {
        "DQN": LOGS / "dqn" / "results.csv",
        "REINFORCE": LOGS / "pg" / "results_reinforce.csv",
        "PPO": LOGS / "pg" / "results_ppo.csv",
        "A2C": LOGS / "pg" / "results_a2c.csv",
    }
    frames = []
    for algo, path in tables.items():
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset="name", keep="last")  
        df["algo"] = algo
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_history(name):
    folder = "dqn" if name.startswith("dqn") else "pg"
    with open(LOGS / folder / f"{name}.json") as f:
        return json.load(f)["history"]


def smooth(values, window):
    s = pd.Series(values, dtype=float)
    return s.rolling(window, min_periods=1).mean().to_numpy()


def window_for(n_points):
    return max(1, n_points // 25)


def best_runs(results):
    idx = results.groupby("algo")["eval_test_mean"].idxmax()
    return results.loc[idx].set_index("algo").loc[list(COLORS)]


def save(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved assets/plots/{filename}")


def frames_to_threshold(history, threshold=0.0):
    """First training step where the smoothed training reward reaches the threshold."""
    rewards = smooth(history["episode_reward"], window_for(len(history["episode_reward"])))
    for frame, r in zip(history["frames"], rewards):
        if not np.isnan(r) and r >= threshold:
            return frame
    return np.nan


def fig_reward_curves(best):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(RANDOM_BASELINE, color=BASELINE_GREY, lw=1.5, ls="--")
    ax.text(100_000, RANDOM_BASELINE + 60, "random agent", color=MUTED, fontsize=9, ha="right")
    ax.axhline(0, color=MUTED, lw=0.8)

    for algo, row in best.iterrows():
        h = load_history(row["name"])
        y = smooth(h["episode_reward"], window_for(len(h["frames"])))
        ax.plot(h["frames"], y, color=COLORS[algo], label=f"{algo} ({row['name']})")

    ax.set_title("Training reward of the best run per algorithm")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward (mean of last 20 episodes)")
    ax.legend(loc="lower right", fontsize=8)
    k_steps(ax)
    save(fig, "fig1_reward_curves.png")


def fig_dqn_loss(results):
    dqn = results[results["algo"] == "DQN"]
    picks = ["dqn_01_baseline", "dqn_09_fast_target", "dqn_04_low_gamma"]
    picks = [p for p in picks if p in set(dqn["name"])]
    run_colors = ["#257de0", "#f47230", "#00b894"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for name, color in zip(picks, run_colors):
        h = load_history(name)
        loss = np.array(h["loss"], dtype=float)
        axes[0].plot(h["frames"], smooth(loss, window_for(len(loss))), color=color, label=name)

    eps_picks = ["dqn_01_baseline", "dqn_07_fast_eps_decay", "dqn_08_high_eps_end"]
    for name, color in zip(eps_picks, run_colors):
        if name in set(dqn["name"]):
            h = load_history(name)
            axes[1].plot(h["frames"], h["epsilon"], color=color, label=name)

    axes[0].set_title("DQN loss (TD error, Huber)")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("DQN exploration rate (epsilon)")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Epsilon")
    axes[1].legend(fontsize=8)

    for ax in axes:
        k_steps(ax)
    save(fig, "fig2_dqn_loss.png")


def fig_pg_entropy(best, results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for algo in ["REINFORCE", "PPO", "A2C"]:
        h = load_history(best.loc[algo, "name"])
        y = smooth(h["entropy"], window_for(len(h["entropy"])))
        axes[0].plot(h["frames"], y, color=COLORS[algo], label=f"{algo} ({best.loc[algo, 'name']})")
    axes[0].axhline(np.log(6), color=BASELINE_GREY, lw=1.2, ls="--")
    axes[0].text(1000, np.log(6) + 0.03, "max entropy (uniform over 6 actions)", color=MUTED, fontsize=8)
    axes[0].set_title("Policy entropy, best run per algorithm")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].legend(fontsize=8, loc="lower left")

    ppo_runs = [("ppo_06_no_entropy", "coef 0.00", "#0e3d7e"),
                ("ppo_01_baseline", "coef 0.01", "#257de0"),
                ("ppo_07_high_entropy", "coef 0.05", "#84b9f3")]
    for name, label, color in ppo_runs:
        if name in set(results["name"]):
            h = load_history(name)
            axes[1].plot(h["frames"], h["entropy"], color=color, label=f"PPO entropy {label}", marker="o", ms=3)
    axes[1].set_title("PPO: effect of the entropy coefficient")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].legend(fontsize=8)

    for ax in axes:
        k_steps(ax)
    save(fig, "fig3_pg_entropy.png")


def fig_sweeps(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, algo in zip(axes.flat, COLORS):
        df = results[results["algo"] == algo].sort_values("eval_test_mean")
        labels = [n.split("_", 2)[-1].replace("_", " ") for n in df["name"]]
        ax.barh(labels, df["eval_test_mean"], color=COLORS[algo], height=0.7,
                edgecolor="white", linewidth=2)
        ax.axvline(0, color=MUTED, lw=0.8)
        for i, v in enumerate(df["eval_test_mean"]):
            ax.text(v + (15 if v >= 0 else -15), i, f"{v:.0f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=TEXT)
        ax.set_title(f"{algo}: 10 hyperparameter runs")
        ax.set_xlabel("Test-period reward (greedy policy)")
        ax.grid(axis="y", visible=False)
        lo = min(df["eval_test_mean"].min(), 0)
        hi = max(df["eval_test_mean"].max(), 0)
        ax.set_xlim(lo - 150, hi + 150)

    fig.suptitle(f"Hyperparameter sweeps (random agent ≈ {RANDOM_BASELINE})",
                 fontweight="bold", color=TEXT)
    save(fig, "fig4_hyperparameter_sweeps.png")


def fig_generalization(best):
    metrics = [
        ("eval_train_mean", "Train period (greedy)", "#84b9f3"),
        ("eval_test_mean", "Test period (greedy)", "#257de0"),
        ("eval_test_stochastic_mean", "Test period (stochastic)", "#0e3d7e"),
    ]
    algos = list(best.index)
    x = np.arange(len(algos))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (col, label, color) in enumerate(metrics):
        values = best[col].to_numpy(dtype=float) if col in best else np.full(len(algos), np.nan)
        bars = ax.bar(x + (i - 1) * width, values, width, label=label, color=color,
                      edgecolor="white", linewidth=2)
        for b, v in zip(bars, values):
            label = "n/a" if np.isnan(v) else f"{v:.0f}"
            ax.text(b.get_x() + b.get_width() / 2, (0 if np.isnan(v) else v) + 8, label,
                    ha="center", va="bottom", fontsize=8, color=MUTED if np.isnan(v) else TEXT)

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"{a}\n{best.loc[a, 'name']}" for a in algos], fontsize=8)
    ax.set_ylabel("Mean episode reward (20 episodes)")
    ax.set_title("Generalization: train vs unseen test period (best run per algorithm)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.text(0, -0.30, "DQN has no stochastic policy (it always picks the highest Q-value), so that bar is n/a.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "fig5_generalization.png")


def fig_convergence(best):
    algos = list(best.index)
    steps = [frames_to_threshold(load_history(best.loc[a, "name"])) for a in algos]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(algos))
    ax.barh(y, [0 if np.isnan(s) else s for s in steps],
            color=[COLORS[a] for a in algos], height=0.6, edgecolor="white", linewidth=2)
    for i, s in enumerate(steps):
        text = "never" if np.isnan(s) else f"{s/1000:.0f}k steps"
        ax.text((0 if np.isnan(s) else s) + 1000, i, text, va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, algos)
    ax.invert_yaxis()
    ax.set_xlabel("Environment steps until training reward first reaches 0 (break-even)")
    ax.set_title("Convergence speed (best run per algorithm)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 110_000)
    k_steps(ax)
    save(fig, "fig6_convergence.png")
    return dict(zip(algos, steps))

def main():
    results = load_results()
    best = best_runs(results)

    print("Making figures...")
    fig_reward_curves(best)
    fig_dqn_loss(results)
    fig_pg_entropy(best, results)
    fig_sweeps(results)
    fig_generalization(best)
    convergence = fig_convergence(best)
    
    summary = best[["name", "eval_train_mean", "eval_test_mean", "eval_test_std"]].copy()
    if "eval_test_stochastic_mean" in best:
        summary["eval_test_stochastic_mean"] = best["eval_test_stochastic_mean"]
    summary["steps_to_break_even"] = [convergence[a] for a in summary.index]
    summary.to_csv(LOGS / "summary_best_runs.csv")

    print("\nBest run per algorithm (test period, greedy):")
    print(summary.to_string())
    print("\nSaved logs/summary_best_runs.csv")


if __name__ == "__main__":
    main()