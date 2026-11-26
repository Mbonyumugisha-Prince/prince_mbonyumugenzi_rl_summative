import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.custom_env import InventoryEnv 


@pytest.fixture
def env():
    e = InventoryEnv(mode="train")
    yield e
    e.close()


def test_gymnasium_api_is_valid():
    
    check_env(InventoryEnv(), skip_render_check=True)


def test_spaces(env):
    assert env.action_space.n == 6
    assert env.observation_space.shape == (11,)
    assert list(env.ORDER_OPTIONS) == [0, 5, 10, 20, 40, 80]


def test_reset_gives_valid_start_state(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["day"] == 0
    assert info["on_the_way"] == 0          
    assert info["stock"] >= 0
    assert info["item_id"].startswith("HOBBIES")


def test_observations_stay_in_range_for_a_full_episode(env):
    obs, _ = env.reset(seed=1)
    for _ in range(env.episode_length):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_episode_is_truncated_after_90_days(env):
    env.reset(seed=2)
    steps = 0
    truncated = terminated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(3)
        steps += 1
    assert truncated and steps == 90


def test_orders_arrive_after_3_day_lead_time(env):
    env.reset(seed=3)
    env.step(4)                     
    assert env.pipeline[-1] == 40   
    env.step(0)                     
    env.step(0)                     
    assert env.pipeline[0] == 40    


def test_reward_matches_formula(env):
    env.reset(seed=4)
    _, reward, _, _, info = env.step(2)  
    expected = (
        1.0 * info["sold"]
        - 0.05 * info["stock"]
        - 2.0 * info["lost"]
        - 3.0                            
    )
    assert reward == pytest.approx(expected)


def test_stock_out_streak_ends_episode():
    env = InventoryEnv(mode="train")
    env.reset(seed=5)
    env.stock = 0.0
    terminated = truncated = False
    days = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(0)  
        days += 1

    if terminated:
        assert info["stockout_streak"] >= env.max_stockout_streak
    assert days <= env.episode_length


def test_train_and_test_periods_do_not_overlap():
    train = InventoryEnv(mode="train")
    test = InventoryEnv(mode="test")
    for item in train.product_ids:
        n_train = len(train.products[item]["sales"])
        n_test = len(test.products[item]["sales"])
        assert n_train > n_test > 90 


@pytest.mark.parametrize("folder", ["dqn", "pg"])
def test_saved_models_load(folder):

    files = sorted((ROOT / "models" / folder).glob("*.pt"))
    assert files, f"no models found in models/{folder}"
    for f in files:
        checkpoint = torch.load(f, map_location="cpu")
        assert "state_dict" in checkpoint and "config" in checkpoint


def test_best_agent_beats_random_agent():

    import main

    actor, _ = main.load_agent("ppo", "ppo_08_short_rollout")

    def run(policy, seed):
        e = InventoryEnv(mode="test")
        obs, _ = e.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            action = e.action_space.sample() if policy is None else main.choose_action(policy, obs)
            obs, r, terminated, truncated, _ = e.step(action)
            total += r
            done = terminated or truncated
        return total

    seeds = range(5)
    trained = np.mean([run(actor, s) for s in seeds])
    random_ = np.mean([run(None, s) for s in seeds])
    assert trained > random_ + 1000