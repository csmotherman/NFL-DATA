#!/usr/bin/env python3
"""
Grade Forward Player-Prop History
=================================

Grades only prop quotes/projections that this repository actually archived.
No synthetic historical sportsbook lines are invented.

Outputs:
- outputs/prop_bet_history.csv
- outputs/prop_forward_report.json
- outputs/validated_prop_rules.json

A strategy is promoted to VALIDATED only when:
- it has enough forward bets and distinct weeks;
- realized ROI is positive;
- a one-sided 95% lower confidence bound on mean unit profit is > 0.

This is intentionally strict.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

from player_props import odds_profit_per_unit


MARKET_TARGET = {
    "pass_attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "rush_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
}

EV_THRESHOLDS = [0.00, 0.025, 0.05, 0.075, 0.10, 0.15]
MIN_FORWARD_BETS = 75
MIN_DISTINCT_WEEKS = 6
MIN_MODEL_PROBABILITY = 0.55
MIN_ROI = 0.03


def normalize_name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_edge_archive(edge_dir: Path) -> pd.DataFrame:
    frames = []

    for path in sorted(edge_dir.glob("*_player_prop_edges.csv")):
        if path.stat().st_size <= 1:
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue

        if df.empty:
            continue

        # Backward-compatible filename fallback.
        match = re.search(r"(\d{4})_week_(\d+)", path.name)
        if "season" not in df.columns and match:
            df["season"] = int(match.group(1))
        if "week" not in df.columns and match:
            df["week"] = int(match.group(2))

        df["archive_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["model_ev_per_unit"] = pd.to_numeric(
        out.get("model_ev_per_unit", np.nan),
        errors="coerce",
    )
    out["model_lean_probability"] = pd.to_numeric(
        out.get("model_lean_probability", np.nan),
        errors="coerce",
    )
    out["probability_edge"] = pd.to_numeric(
        out.get("probability_edge", np.nan),
        errors="coerce",
    )
    out["player_key"] = out["player_name"].map(normalize_name)

    return out


def select_executable_quotes(edges: pd.DataFrame) -> pd.DataFrame:
    """
    One quote per player/market/week.

    This approximates real line shopping: among simultaneously archived quotes,
    keep the highest model-EV quote that was not already marked unavailable.
    """
    if edges.empty:
        return pd.DataFrame()

    required = {
        "model_ev_per_unit",
        "model_lean_probability",
        "line",
    }
    if not required.issubset(edges.columns):
        return pd.DataFrame()

    x = edges.copy()

    if "availability_ok" in x.columns:
        ok = x["availability_ok"]
        if ok.dtype != bool:
            ok = ok.astype(str).str.lower().isin(["true", "1", "yes"])
        x = x[ok].copy()

    x = x[
        x["model_ev_per_unit"].notna()
        & x["model_lean_probability"].notna()
        & x["line"].notna()
    ].copy()

    if x.empty:
        return x

    x = x.sort_values(
        ["model_ev_per_unit", "probability_edge"],
        ascending=False,
        na_position="last",
    )

    return x.drop_duplicates(
        subset=["season", "week", "player_key", "market"],
        keep="first",
    ).copy()


def load_actuals(seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()

    stats = nfl.load_player_stats(
        seasons=seasons,
        summary_level="week",
    ).to_pandas()

    if "season_type" in stats.columns:
        stats = stats[stats["season_type"].eq("REG")].copy()

    if "player_display_name" not in stats.columns:
        stats["player_display_name"] = stats.get(
            "player_name",
            stats.get("player_id", ""),
        )

    stats["player_key"] = stats["player_display_name"].map(normalize_name)

    rows = []
    for market, target in MARKET_TARGET.items():
        if target not in stats.columns:
            continue

        part = stats[
            ["season", "week", "player_key", "player_display_name", target]
        ].copy()
        part["market"] = market
        part["actual"] = pd.to_numeric(part[target], errors="coerce")
        rows.append(
            part.drop(columns=[target])
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grade_quotes(edges: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or actuals.empty:
        return pd.DataFrame()

    g = edges.merge(
        actuals,
        on=["season", "week", "player_key", "market"],
        how="inner",
        suffixes=("", "_actual"),
    )

    g = g[g["actual"].notna()].copy()
    if g.empty:
        return g

    over = g["lean"].astype(str).str.upper().eq("OVER")
    under = g["lean"].astype(str).str.upper().eq("UNDER")

    g["result"] = np.select(
        [
            over & (g["actual"] > g["line"]),
            under & (g["actual"] < g["line"]),
            g["actual"] == g["line"],
        ],
        ["WIN", "WIN", "PUSH"],
        default="LOSS",
    )

    if "lean_price" not in g.columns:
        g["lean_price"] = np.where(
            over,
            pd.to_numeric(g.get("price_over", np.nan), errors="coerce"),
            pd.to_numeric(g.get("price_under", np.nan), errors="coerce"),
        )

    g["win_profit_per_unit"] = g["lean_price"].apply(
        odds_profit_per_unit
    )

    g["unit_profit"] = np.select(
        [
            g["result"].eq("WIN"),
            g["result"].eq("LOSS"),
        ],
        [
            g["win_profit_per_unit"],
            -1.0,
        ],
        default=0.0,
    )

    return g


def summarize_strategy(g: pd.DataFrame) -> dict:
    settled = g[g["result"].isin(["WIN", "LOSS"])].copy()

    if settled.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "roi": None,
            "roi_lcb_95_one_sided": None,
            "weeks": 0,
        }

    profits = pd.to_numeric(settled["unit_profit"], errors="coerce").dropna()
    wins = int(settled["result"].eq("WIN").sum())
    losses = int(settled["result"].eq("LOSS").sum())
    bets = wins + losses
    roi = float(profits.mean()) if len(profits) else np.nan

    if len(profits) > 1:
        se = float(profits.std(ddof=1) / math.sqrt(len(profits)))
        lcb = roi - 1.645 * se
    else:
        lcb = np.nan

    weeks = int(
        settled[["season", "week"]].drop_duplicates().shape[0]
    )

    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": None if bets == 0 else round(wins / bets, 4),
        "roi": None if not np.isfinite(roi) else round(roi, 4),
        "roi_lcb_95_one_sided": (
            None if not np.isfinite(lcb) else round(float(lcb), 4)
        ),
        "weeks": weeks,
    }


def evaluate_rules(history: pd.DataFrame) -> tuple[dict, dict]:
    report = {}
    validated_rules = {}

    markets = sorted(history["market"].dropna().unique())

    for market in markets:
        market_rows = history[history["market"].eq(market)].copy()
        market_report = {}

        for threshold in EV_THRESHOLDS:
            subset = market_rows[
                (market_rows["model_ev_per_unit"] >= threshold)
                & (
                    market_rows["model_lean_probability"]
                    >= MIN_MODEL_PROBABILITY
                )
            ].copy()

            stats = summarize_strategy(subset)
            stats["ev_threshold"] = threshold

            validated = (
                stats["bets"] >= MIN_FORWARD_BETS
                and stats["weeks"] >= MIN_DISTINCT_WEEKS
                and stats["roi"] is not None
                and stats["roi"] >= MIN_ROI
                and stats["roi_lcb_95_one_sided"] is not None
                and stats["roi_lcb_95_one_sided"] > 0
            )
            stats["validated"] = bool(validated)
            market_report[str(threshold)] = stats

        report[market] = market_report

        valid = [
            x for x in market_report.values()
            if x["validated"]
        ]

        if valid:
            # Prefer the validated rule with strongest lower confidence bound;
            # break ties with sample size.
            best = max(
                valid,
                key=lambda x: (
                    x["roi_lcb_95_one_sided"],
                    x["bets"],
                ),
            )
            validated_rules[market] = {
                "status": "VALIDATED",
                "min_model_ev_per_unit": best["ev_threshold"],
                "min_model_probability": MIN_MODEL_PROBABILITY,
                "forward_bets": best["bets"],
                "forward_weeks": best["weeks"],
                "forward_roi": best["roi"],
                "forward_roi_lcb_95_one_sided": (
                    best["roi_lcb_95_one_sided"]
                ),
            }
        else:
            validated_rules[market] = {
                "status": "WATCH",
                "reason": (
                    "No forward rule yet clears the sample-size and "
                    "positive lower-confidence-bound requirements."
                ),
            }

    return report, validated_rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-dir",
        default="outputs/player_edges",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
    )
    args = parser.parse_args()

    edge_dir = Path(args.edge_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = load_edge_archive(edge_dir)
    selected = select_executable_quotes(edges)

    if selected.empty:
        payload = {
            "status": "waiting_for_forward_market_history",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "archived_quotes": int(len(edges)),
            "selected_quotes": 0,
            "message": (
                "No executable archived prop quotes are available yet. "
                "Configure live prop odds or add manual lines."
            ),
        }
        with (output_dir / "prop_forward_report.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(payload, f, indent=2)

        with (output_dir / "validated_prop_rules.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({}, f, indent=2)

        pd.DataFrame().to_csv(
            output_dir / "prop_bet_history.csv",
            index=False,
        )
        print(payload["message"])
        return

    seasons = sorted(
        int(x) for x in selected["season"].dropna().unique()
    )
    actuals = load_actuals(seasons)
    history = grade_quotes(selected, actuals)

    history.to_csv(
        output_dir / "prop_bet_history.csv",
        index=False,
    )

    if history.empty:
        strategy_report = {}
        rules = {}
    else:
        strategy_report, rules = evaluate_rules(history)

    payload = {
        "status": "ok" if not history.empty else "no_settled_quotes_yet",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "archived_quotes": int(len(edges)),
        "selected_quotes": int(len(selected)),
        "settled_quotes": int(len(history)),
        "minimum_forward_bets_for_validation": MIN_FORWARD_BETS,
        "minimum_distinct_weeks_for_validation": MIN_DISTINCT_WEEKS,
        "minimum_model_probability": MIN_MODEL_PROBABILITY,
        "minimum_roi_for_validation": MIN_ROI,
        "strategies": strategy_report,
    }

    with (output_dir / "prop_forward_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    with (output_dir / "validated_prop_rules.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(rules, f, indent=2, allow_nan=False)

    print("\nFORWARD PROP GRADING COMPLETE")
    print("=" * 72)
    print(f"Archived quote rows: {len(edges):,}")
    print(f"One best quote/player-market: {len(selected):,}")
    print(f"Settled forward quotes: {len(history):,}")

    validated = [
        (market, rule)
        for market, rule in rules.items()
        if rule.get("status") == "VALIDATED"
    ]

    if not validated:
        print("Validated prop strategies: 0")
    else:
        for market, rule in validated:
            print(
                f"{market}: VALIDATED | "
                f"EV >= {rule['min_model_ev_per_unit']:.3f} | "
                f"ROI {rule['forward_roi']:.1%} | "
                f"{rule['forward_bets']} bets"
            )


if __name__ == "__main__":
    main()
