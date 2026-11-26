from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

try:
    from .rendering import InventoryRenderer   # when imported as a package (main.py, training)
except ImportError:
    from rendering import InventoryRenderer    # when running this file directly


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "selected_products.csv"


class InventoryEnv(gym.Env):
    """
    StockMind inventory environment.

    Each day the agent decides how many units of one product to order.
    Orders arrive after a delay (lead time). Customer demand comes from
    real Walmart sales (M5 dataset, store CA_1, HOBBIES category).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Action i means: order ORDER_OPTIONS[i] units
    ORDER_OPTIONS = np.array([0, 5, 10, 20, 40, 80])

    def __init__(
        self,
        data_file=DATA_FILE,
        mode="train",
        episode_length=90,
        lead_time=3,
        max_stock=400,
        render_mode=None,
    ):
        super().__init__()

        self.mode = mode
        self.episode_length = episode_length
        self.lead_time = lead_time
        self.max_stock = max_stock
        self.render_mode = render_mode
        self.renderer = None

        # Reward settings (measured in "units of profit")
        self.unit_profit = 1.0        # earned per unit sold
        self.holding_cost = 0.05      # paid per unit kept in stock each day
        self.stockout_penalty = 2.0   # paid per unit of demand we could not meet
        self.order_fixed_cost = 3.0   # paid each time we place an order (truck/admin cost)
        self.waste_cost = 1.0         # paid per unit that does not fit on the shelf

        # Terminal condition: too many stock-out days in a row
        self.max_stockout_streak = 14

        # Used to scale observations into the range [0, 1]
        self.history_days = 28
        self.demand_scale = 50.0

        self.products = self._load_products(data_file)
        self.product_ids = list(self.products)
        self.max_price = max(p["price"].max() for p in self.products.values())

        self.action_space = spaces.Discrete(len(self.ORDER_OPTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

    def _load_products(self, data_file):
        df = pd.read_csv(data_file, parse_dates=["date"])
        df = df.sort_values(["item_id", "date"])

        products = {}

        for item_id, group in df.groupby("item_id"):
            # First 80% of days for training, last 20% for testing
            split = int(len(group) * 0.8)

            if self.mode == "train":
                group = group.iloc[:split]
            else:
                group = group.iloc[split:]

            products[item_id] = {
                "sales": group["sales"].to_numpy(dtype=np.float32),
                "price": group["sell_price"].to_numpy(dtype=np.float32),
                "event": group["event_name_1"].notna().to_numpy(dtype=np.float32),
                "snap": group["snap_CA"].to_numpy(dtype=np.float32),
                "wday": group["wday"].to_numpy(dtype=np.float32),
            }

        return products

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start state: random product, random start day, random starting stock
        self.item_id = str(self.np_random.choice(self.product_ids))
        self.data = self.products[self.item_id]

        last_start = len(self.data["sales"]) - self.episode_length
        self.t = int(self.np_random.integers(self.history_days, last_start))

        recent_avg = self.data["sales"][self.t - self.history_days:self.t].mean()
        max_start_stock = int(recent_avg * self.lead_time * 2) + 1
        self.stock = float(self.np_random.integers(0, max_start_stock))

        # Orders on the way: pipeline[0] arrives tomorrow
        self.pipeline = [0.0] * self.lead_time

        self.day = 0
        self.stockout_streak = 0
        self.last = {"order": 0, "demand": 0.0, "sold": 0.0, "lost": 0.0, "reward": 0.0}

        if self.renderer is not None:
            self.renderer.reset()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        order = int(self.ORDER_OPTIONS[action])

        # 1. Delivery arrives (extra units that do not fit are wasted)
        arrived = self.pipeline.pop(0)
        total = self.stock + arrived
        wasted = max(0.0, total - self.max_stock)
        self.stock = min(total, self.max_stock)

        # 2. Place today's order (arrives after lead_time days)
        self.pipeline.append(float(order))

        # 3. Customers arrive (real demand from the data)
        demand = float(self.data["sales"][self.t])
        sold = min(self.stock, demand)
        lost = demand - sold
        self.stock -= sold

        # 4. Reward
        reward = (
            self.unit_profit * sold
            - self.holding_cost * self.stock
            - self.stockout_penalty * lost
            - self.waste_cost * wasted
            - (self.order_fixed_cost if order > 0 else 0.0)
        )

        # 5. Move to the next day
        self.stockout_streak = self.stockout_streak + 1 if lost > 0 else 0
        self.t += 1
        self.day += 1

        terminated = self.stockout_streak >= self.max_stockout_streak
        truncated = self.day >= self.episode_length

        self.last = {
            "order": order,
            "demand": demand,
            "sold": sold,
            "lost": lost,
            "reward": reward,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self):
        sales = self.data["sales"]
        t = self.t

        obs = np.array(
            [
                self.stock / self.max_stock,                          # 0 stock on hand
                sum(self.pipeline) / self.max_stock,                  # 1 units on the way
                self.pipeline[0] / self.ORDER_OPTIONS.max(),          # 2 arriving tomorrow
                sales[t - 7:t].mean() / self.demand_scale,            # 3 avg sales last 7 days
                sales[t - 28:t].mean() / self.demand_scale,           # 4 avg sales last 28 days
                sales[t - 1] / self.demand_scale,                     # 5 yesterday's sales
                (self.data["wday"][t] - 1) / 6,                       # 6 day of week
                self.data["event"][t],                                # 7 event/holiday today
                self.data["snap"][t],                                 # 8 SNAP day today
                self.data["price"][t] / self.max_price,               # 9 price
                self.day / self.episode_length,                       # 10 episode progress
            ],
            dtype=np.float32,
        )

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        return {
            "item_id": self.item_id,
            "day": self.day,
            "stock": self.stock,
            "on_the_way": sum(self.pipeline),
            "stockout_streak": self.stockout_streak,
            **self.last,
        }

    def render(self):
        i = self._get_info()

        # Terminal output (verbose log)
        print(
            f"Day {i['day']:>3} | {i['item_id']} | order {i['order']:>3} | "
            f"demand {i['demand']:>5.0f} | sold {i['sold']:>5.0f} | "
            f"lost {i['lost']:>4.0f} | stock {i['stock']:>5.0f} | "
            f"reward {i['reward']:>7.2f}"
        )

        # Graphical window (pygame)
        if self.renderer is None:
            self.renderer = InventoryRenderer(
                self.max_stock, self.lead_time, fps=self.metadata["render_fps"]
            )
        self.renderer.draw(i, self.pipeline)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    # Check the environment without opening a window
    check_env(InventoryEnv(), skip_render_check=True)
    print("Environment check passed!\n")

    env = InventoryEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    total_reward = 0.0

    for _ in range(env.episode_length):
        action = env.action_space.sample()  # random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nRandom agent total reward: {total_reward:.2f}")

    # Save the last frame as a picture for the report
    Path(ROOT / "assets").mkdir(exist_ok=True)
    env.renderer.save(ROOT / "assets" / "environment_random_agent.png")
    print("Screenshot saved to assets/environment_random_agent.png")

    env.close()
