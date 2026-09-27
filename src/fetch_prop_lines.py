#!/usr/bin/env python3
"""
Fetch live NFL player prop lines from PropLine.

Requires:
    PROPLINE_API_KEY GitHub secret / environment variable.

Free tier currently supports 1,000 requests/day. This script pulls one snapshot
per Thursday run and writes normalized lines for the projection pipeline.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from propline import PropLine


SPORT = "football_nfl"

MARKET_MAP = {
    "player_pass_attempts": "pass_attempts",
    "player_pass_completions": "completions",
    "player_pass_yds": "passing_yards",
    "player_rush_attempts": "rush_attempts",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}

PROP_MARKETS = list(MARKET_MAP.keys())


def clean_player_name(value: str) -> str:
    # Prop feeds sometimes append a team tag: "Player Name (DET)".
    name = re.sub(r"\s*\([A-Z]{2,4}\)\s*$", "", str(value)).strip()
    return " ".join(name.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="inputs/player_prop_lines.csv")
    parser.add_argument("--archive-dir", default="outputs/prop_line_snapshots")
    args = parser.parse_args()

    api_key = os.getenv("PROPLINE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("PROPLINE_API_KEY is not set.")

    client = PropLine(api_key)
    events = client.get_events(SPORT)

    rows = []

    for event in events:
        event_id = event.get("id")
        if event_id is None:
            continue

        try:
            odds = client.get_odds(
                SPORT,
                event_id=event_id,
                markets=PROP_MARKETS,
            )
        except Exception as exc:
            print(f"[warning] Could not fetch event {event_id}: {exc}")
            continue

        home = odds.get("home_team") or event.get("home_team")
        away = odds.get("away_team") or event.get("away_team")
        commence = odds.get("commence_time") or event.get("commence_time")

        for book in odds.get("bookmakers", []) or []:
            book_key = book.get("key") or book.get("title") or "unknown"

            for market in book.get("markets", []) or []:
                source_market = market.get("key")
                normalized_market = MARKET_MAP.get(source_market)
                if normalized_market is None:
                    continue

                grouped = {}

                for outcome in market.get("outcomes", []) or []:
                    direction = str(outcome.get("name", "")).strip().lower()
                    if direction not in {"over", "under"}:
                        continue

                    point = outcome.get("point")
                    description = outcome.get("description")
                    if point is None or not description:
                        continue

                    player_name = clean_player_name(description)
                    source_player_id = outcome.get("player_id")

                    key = (
                        player_name,
                        float(point),
                        str(source_player_id or ""),
                    )
                    rec = grouped.setdefault(
                        key,
                        {
                            "player_name": player_name,
                            "source_player_id": source_player_id,
                            "market": normalized_market,
                            "line": float(point),
                            "book": book_key,
                            "price_over": None,
                            "price_under": None,
                            "event_id": event_id,
                            "home_team": home,
                            "away_team": away,
                            "commence_time": commence,
                            "last_update": market.get("last_update"),
                        },
                    )
                    rec[f"price_{direction}"] = outcome.get("price")

                rows.extend(grouped.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if df.empty:
        # Keep a valid schema so downstream code does not break.
        df = pd.DataFrame(columns=[
            "player_name", "source_player_id", "market", "line", "book",
            "price_over", "price_under", "event_id", "home_team",
            "away_team", "commence_time", "last_update",
        ])
    else:
        df = df.drop_duplicates(
            subset=["player_name", "market", "line", "book", "event_id"],
            keep="last",
        )
        df = df.sort_values(
            ["commence_time", "player_name", "market", "book"],
            na_position="last",
        )

    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} live prop quote rows to {output}")

    # Preserve the exact market the model saw at Thursday execution time.
    # These snapshots become our clean forward prop-line backtest archive.
    if not df.empty and args.archive_dir:
        archive_dir = Path(args.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = archive_dir / f"nfl_props_{stamp}.csv"
        df.to_csv(archive_path, index=False)
        print(f"Archived market snapshot to {archive_path}")


if __name__ == "__main__":
    main()
