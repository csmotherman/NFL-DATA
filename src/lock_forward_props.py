#!/usr/bin/env python3
"""Lock current player-prop market edges before target-week games settle.

Player projections may be refreshed repeatedly. A trustworthy forward betting
record needs an immutable copy of the quoted sportsbook lines and the model's
decision before results are known. This script creates that copy exactly once.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import nflreadpy as nfl


def target_week_is_clean(season: int, week: int) -> bool:
    schedules = nfl.load_schedules([season]).to_pandas()
    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"].eq("REG")].copy()

    games = schedules[
        schedules["season"].eq(season)
        & schedules["week"].eq(week)
    ].copy()

    if games.empty:
        return False

    home_final = pd.to_numeric(
        games.get("home_score"), errors="coerce"
    ).notna()
    away_final = pd.to_numeric(
        games.get("away_score"), errors="coerce"
    ).notna()
    return not bool((home_final | away_final).any())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edges",
        default="outputs/latest_player_prop_edges.csv",
    )
    parser.add_argument(
        "--archive-dir",
        default="outputs/player_edges_locked",
    )
    args = parser.parse_args()

    edge_path = Path(args.edges)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    if not edge_path.exists() or edge_path.stat().st_size <= 1:
        print("No live player-prop edges to lock.")
        return

    edges = pd.read_csv(edge_path)
    if edges.empty:
        print("No live player-prop edges to lock.")
        return

    season_values = pd.to_numeric(edges["season"], errors="coerce").dropna()
    week_values = pd.to_numeric(edges["week"], errors="coerce").dropna()
    if season_values.empty or week_values.empty:
        raise RuntimeError("Player-prop edges are missing season/week.")

    season = int(season_values.iloc[0])
    week = int(week_values.iloc[0])
    output = archive_dir / f"{season}_week_{week:02d}_player_prop_edges.csv"

    if output.exists():
        print(f"Player-prop ledger already locked: {output}")
        return

    if not target_week_is_clean(season, week):
        print(
            "Not locking player-prop edges because at least one target-week "
            "game already has a final score. This prevents hindsight leakage."
        )
        return

    locked = edges.copy()
    locked["locked_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    locked.to_csv(output, index=False)
    print(f"Locked forward player-prop ledger: {output}")


if __name__ == "__main__":
    main()
