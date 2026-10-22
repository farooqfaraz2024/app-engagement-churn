"""Synthetic GA4-shaped events with a known churn mechanism, for end-to-end tests.

Each user joins on a random day, uses the app with a personal daily probability,
fades out over the week before quitting, and sometimes uninstalls on the way out.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from appchurn.config import DATASET
from tests.sql_harness import event

JOURNEYS = ["level_start_quickplay", "post_score", "spend_virtual_currency", "ad_reward"]


def synthetic_events(n_users: int = 400, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    start, end = DATASET.data_start, DATASET.data_end
    span = (end - start).days
    events: list[dict] = []
    for i in range(n_users):
        user = f"user_{i:04d}"
        join = start + timedelta(days=int(rng.integers(0, span - 10)))
        lifetime = int(rng.exponential(45)) + 3
        quit_day = join + timedelta(days=lifetime)
        base_p = rng.beta(2, 3)
        platform = rng.choice(["ANDROID", "IOS"], p=[0.7, 0.3])
        country = rng.choice(["Germany", "United States", "India", "Brazil"])
        payer = rng.uniform() < 0.08
        day = join
        while day <= min(quit_day, end):
            days_left = (quit_day - day).days
            p = base_p * (0.35 if days_left < 7 else 1.0)  # fading before leaving
            if rng.uniform() < p:
                kw = dict(first_touch=join, platform=platform, country=country)
                events.append(event(user, day, "session_start", **kw))
                events.append(
                    event(
                        user, day, "user_engagement", engagement_ms=int(rng.gamma(2, 60_000)), **kw
                    )
                )
                for j in JOURNEYS:
                    for _ in range(rng.poisson(1.5 if days_left >= 7 else 0.5)):
                        events.append(event(user, day, j, **kw))
                if payer and rng.uniform() < 0.1:
                    events.append(event(user, day, "in_app_purchase", revenue=1.99, **kw))
            day += timedelta(days=1)
        if quit_day <= end and rng.uniform() < 0.3:
            events.append(event(user, quit_day, "app_remove", first_touch=join))
    return events
