import math
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


# Window size
WIDTH, HEIGHT = 1000, 640

# Colours (R, G, B)
BG = (245, 247, 250)
PANEL = (255, 255, 255)
BORDER = (210, 215, 222)
TEXT = (33, 37, 41)
MUTED = (108, 117, 125)
GREEN = (46, 160, 67)
ORANGE = (230, 145, 30)
RED = (214, 57, 57)
BLUE = (52, 120, 246)
GREY = (190, 195, 200)
BOX = (196, 140, 80)
BOX_EDGE = (140, 95, 50)
SHELF = (120, 90, 60)

UNITS_PER_BOX = 10      # one box on the shelf = 10 units
BOXES_PER_ROW = 10
HISTORY_DAYS = 60       # days shown in the chart


class InventoryRenderer:
    """Draws the StockMind inventory simulation with pygame."""

    def __init__(self, max_stock, lead_time, fps=4):
        pygame.init()
        pygame.display.set_caption("StockMind AI - Inventory Simulation")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("arial", 26, bold=True)
        self.font = pygame.font.SysFont("arial", 18)
        self.font_small = pygame.font.SysFont("arial", 14)

        self.max_stock = max_stock
        self.lead_time = lead_time
        self.fps = fps
        self.is_open = True

        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.total_sold = 0.0
        self.total_lost = 0.0
        self.stock_history = []
        self.demand_history = []

    # ------------------------------------------------------------------
    # Main drawing function (called once per day)
    # ------------------------------------------------------------------
    def draw(self, info, pipeline):
        if not self.is_open:
            return

        # Keep the window responsive and allow closing it
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Update running totals (day 0 is the start state, nothing happened yet)
        if info["day"] > 0:
            self.total_reward += info["reward"]
            self.total_sold += info["sold"]
            self.total_lost += info["lost"]
            self.stock_history.append(info["stock"])
            self.demand_history.append(info["demand"])
            self.stock_history = self.stock_history[-HISTORY_DAYS:]
            self.demand_history = self.demand_history[-HISTORY_DAYS:]

        self.screen.fill(BG)

        self._draw_header(info)
        self._draw_shelf(info)
        self._draw_trucks(pipeline)
        self._draw_customers(info)
        self._draw_stats(info)
        self._draw_chart()

        pygame.display.flip()
        self.clock.tick(self.fps)

    def save(self, path):
        """Save the current frame as an image (useful for the report)."""
        pygame.image.save(self.screen, str(path))

    def close(self):
        if self.is_open:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _text(self, text, pos, font=None, color=TEXT):
        surface = (font or self.font).render(str(text), True, color)
        self.screen.blit(surface, pos)

    def _panel(self, rect, title):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=10)
        self._text(title, (rect[0] + 14, rect[1] + 10), self.font, MUTED)

    # ------------------------------------------------------------------
    # Sections of the screen
    # ------------------------------------------------------------------
    def _draw_header(self, info):
        self._text("StockMind AI - Inventory Manager", (20, 18), self.font_big)
        self._text(
            f"Product: {info['item_id']}     Day {info['day']}",
            (600, 24),
            self.font,
            MUTED,
        )

    def _draw_shelf(self, info):
        rect = (20, 70, 300, 330)
        self._panel(rect, "Store shelf")

        stock = info["stock"]
        if stock <= 0:
            color = RED
        elif stock < 20:
            color = ORANGE
        else:
            color = GREEN
        self._text(f"{stock:.0f} / {self.max_stock} units", (34, 100), self.font, color)

        rows = math.ceil(self.max_stock / (UNITS_PER_BOX * BOXES_PER_ROW))
        boxes = math.ceil(stock / UNITS_PER_BOX)
        row_height = 250 // rows
        bottom = 385

        # Shelf planks
        for r in range(rows):
            plank_y = bottom - r * row_height
            pygame.draw.rect(self.screen, SHELF, (35, plank_y, 270, 6))

        # Boxes (each box = 10 units)
        for i in range(boxes):
            r, c = divmod(i, BOXES_PER_ROW)
            x = 40 + c * 26
            y = bottom - r * row_height - 24
            pygame.draw.rect(self.screen, BOX, (x, y, 22, 22), border_radius=3)
            pygame.draw.rect(self.screen, BOX_EDGE, (x, y, 22, 22), width=1, border_radius=3)

    def _draw_trucks(self, pipeline):
        rect = (340, 70, 400, 150)
        self._panel(rect, "Deliveries on the way")

        slot = 380 // max(len(pipeline), 1)

        for k, units in enumerate(pipeline):
            x = 355 + k * slot
            y = 115
            color = BLUE if units > 0 else GREY

            # Cargo box, cab and wheels
            pygame.draw.rect(self.screen, color, (x, y, 62, 34), border_radius=4)
            pygame.draw.rect(self.screen, color, (x + 64, y + 10, 24, 24), border_radius=4)
            pygame.draw.circle(self.screen, TEXT, (x + 14, y + 38), 6)
            pygame.draw.circle(self.screen, TEXT, (x + 74, y + 38), 6)

            days = k + 1
            label = "tomorrow" if days == 1 else f"in {days} days"
            self._text(f"{units:.0f} units", (x, y + 50), self.font, TEXT if units > 0 else MUTED)
            self._text(label, (x, y + 72), self.font_small, MUTED)

    def _draw_customers(self, info):
        rect = (340, 230, 400, 170)
        self._panel(rect, "Customers today")

        demand = int(info["demand"])
        sold = int(info["sold"])
        shown = min(demand, 50)

        for i in range(shown):
            r, c = divmod(i, 25)
            x = 360 + c * 15
            y = 275 + r * 32
            color = GREEN if i < sold else RED
            pygame.draw.circle(self.screen, color, (x, y), 5)                       # head
            pygame.draw.rect(self.screen, color, (x - 5, y + 7, 10, 12), border_radius=3)  # body

        if demand > shown:
            self._text(f"+{demand - shown} more", (640, 340), self.font_small, MUTED)

        self._text(
            f"Demand {demand}   Sold {sold}   Lost {int(info['lost'])}",
            (355, 365),
            self.font,
            RED if info["lost"] > 0 else TEXT,
        )

    def _draw_stats(self, info):
        rect = (760, 70, 220, 330)
        self._panel(rect, "Performance")

        served = self.total_sold + self.total_lost
        fill_rate = 100 * self.total_sold / served if served > 0 else 100.0

        lines = [
            ("Order today", f"{info['order']}", TEXT),
            ("Reward today", f"{info['reward']:.2f}", GREEN if info["reward"] >= 0 else RED),
            ("Total reward", f"{self.total_reward:.1f}", GREEN if self.total_reward >= 0 else RED),
            ("Units sold", f"{self.total_sold:.0f}", TEXT),
            ("Lost sales", f"{self.total_lost:.0f}", RED if self.total_lost > 0 else TEXT),
            ("Fill rate", f"{fill_rate:.1f}%", GREEN if fill_rate >= 95 else ORANGE),
            ("Stock-out streak", f"{info['stockout_streak']}", RED if info["stockout_streak"] > 0 else TEXT),
        ]

        y = 105
        for label, value, color in lines:
            self._text(label, (775, y), self.font_small, MUTED)
            self._text(value, (775, y + 16), self.font, color)
            y += 42

    def _draw_chart(self):
        rect = (20, 420, 960, 200)
        self._panel(rect, f"Last {HISTORY_DAYS} days:  stock (blue line)  vs  demand (orange bars)")

        x0, y0, w, h = 40, 450, 920, 150
        pygame.draw.line(self.screen, BORDER, (x0, y0 + h), (x0 + w, y0 + h))

        if not self.stock_history:
            return

        y_max = max(max(self.stock_history), max(self.demand_history), 1)
        step = w / HISTORY_DAYS

        # Demand bars
        for i, d in enumerate(self.demand_history):
            bar_h = h * d / y_max
            pygame.draw.rect(
                self.screen, ORANGE,
                (x0 + i * step, y0 + h - bar_h, max(step - 2, 1), bar_h),
            )

        # Stock line
        points = [
            (x0 + i * step + step / 2, y0 + h - h * s / y_max)
            for i, s in enumerate(self.stock_history)
        ]
        if len(points) > 1:
            pygame.draw.lines(self.screen, BLUE, False, points, 3)

        self._text(f"max {y_max:.0f}", (x0 + w - 70, y0 - 22), self.font_small, MUTED)
