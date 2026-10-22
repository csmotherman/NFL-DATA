#!/usr/bin/env python3
"""Historical Week-6 frozen walk-forward evaluation of active-game ROS PPR estimates.

All predictions use only Weeks 1-6 of the validation year plus PRIOR seasons.
Actual Weeks 7-17 are used solely to score predictions, never as features.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fantasy_leaderboard import load_current_season
from fantasy_trade_market import build, future_schedule, load_skill_history


def validate(seasons=(2024, 2025), output_dir="outputs"):
    folds = []
    pooled = []
    for season in seasons:
        complete = load_current_season(season)
        before = complete[complete["week"].between(1, 6)].copy()
        after = complete[complete["week"].between(7, 17)].copy()
        history = load_skill_history(season)
        pred = build(before, future_schedule(season, 6, 17), 6, history)
        after = after[after["position"].isin(("RB", "WR", "TE"))]
        actual = after.groupby(["player_id", "position"], as_index=False).agg(
            played_ros_games=("game_id", "nunique"),
            actual_ros_ppg=("fantasy_points_ppr", "mean"),
        )
        joined = pred[pred["position"].isin(("RB", "WR", "TE"))].merge(
            actual, on=["player_id", "position"], how="inner", validate="one_to_one"
        )
        # Evaluate per game PLAYED because the model does not forecast injury absences.
        joined = joined[(joined["played_ros_games"] >= 3) & (joined["games"] >= 3)]
        if joined.empty:
            continue
        y = joined["actual_ros_ppg"].astype(float)
        model = joined["projected_ros_ppr_pg"].astype(float)
        season_pg = joined["ppr_pg"].astype(float)
        for pos, part in joined.groupby("position"):
            t = part["actual_ros_ppg"].astype(float)
            p = part["projected_ros_ppr_pg"].astype(float)
            b = part["ppr_pg"].astype(float)
            folds.append({
                "season": season, "position": pos, "players": len(part),
                "model_mae": round(float((p-t).abs().mean()), 4),
                "season_ppr_baseline_mae": round(float((b-t).abs().mean()), 4),
                "model_bias": round(float((p-t).mean()), 4),
                "model_spearman": round(float(p.corr(t, method="spearman")), 4) if len(part)>2 else None,
            })
        pooled.append(joined[["season", "position", "player_id", "player_name", "actual_ros_ppg", "projected_ros_ppr_pg", "ppr_pg"]].assign(season=season))
    if not pooled:
        raise RuntimeError("No valid historical holdout sample")
    all_rows = pd.concat(pooled, ignore_index=True)
    err = (all_rows["projected_ros_ppr_pg"] - all_rows["actual_ros_ppg"]).abs()
    base_err = (all_rows["ppr_pg"] - all_rows["actual_ros_ppg"]).abs()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": folds,
        "total_players_season_rows": len(all_rows),
        "model_mae": round(float(err.mean()), 4),
        "season_ppr_baseline_mae": round(float(base_err.mean()), 4),
        "beats_season_baseline": bool(err.mean() < base_err.mean()),
        "method": "Week-6 cutoff; prior years only as skill priors; compare Weeks 7-17 realized per active game",
        "limitations": [
            "Selection conditions on playing at least three ROS games, excluding worst injury outcomes",
            "Not true player-season time-series cross-validation beyond these two holdouts",
            "No team-by-team injury, roster, coach or QB rotation features",
            "Season averages are a demanding but useful naïve baseline",
            "Do not use current-season ROS rankings as calibrated forecasts if model fails baseline",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fantasy_trade_validation.json").write_text(json.dumps(report, indent=2))
    all_rows.to_csv(out / "fantasy_trade_validation_rows.csv", index=False)
    print("FANTASY TRADE VALIDATION", report["total_players_season_rows"],
          "model MAE", report["model_mae"], "naive MAE", report["season_ppr_baseline_mae"],
          "beats baseline", report["beats_season_baseline"])
    return report


if __name__ == "__main__":
    validate()
