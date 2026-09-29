#!/usr/bin/env python3
"""
Fantasy ROS Model
=================

A leakage-safe rest-of-season PPR model for RBs and WRs.

The current-season snapshot is frozen after Week 3. Every forecast for
Weeks 4-16 uses only:
- player usage/production from Weeks 1-3,
- team production from Weeks 1-3,
- opponent fantasy production allowed from Weeks 1-3,
- the known Weeks 4-16 schedule.

Historical training rows are built the exact same way for prior seasons.
No Weeks 4-16 player or defense results are used as features for that season.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl_edge import BlendRegressor, configure_cache


START_SEASON = 2018
SNAPSHOT_END_WEEK = 3
FORECAST_START_WEEK = 4
FORECAST_END_WEEK = 16
VALIDATION_SEASONS = [2022, 2023, 2024, 2025]
POSITIONS = ("RB", "WR")
DEFENSE_SHRINK_GAMES = 3.0

BOX_COLUMNS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_epa",
]

FEATURES = [
    "games_w1_3",
    "ppr_pg_w1_3",
    "ppr_sd_w1_3",
    "opportunities_pg_w1_3",
    "carries_pg_w1_3",
    "rushing_yards_pg_w1_3",
    "rushing_tds_pg_w1_3",
    "targets_pg_w1_3",
    "receptions_pg_w1_3",
    "receiving_yards_pg_w1_3",
    "receiving_tds_pg_w1_3",
    "receiving_air_yards_pg_w1_3",
    "receiving_yac_pg_w1_3",
    "rush_share_w1_3",
    "target_share_w1_3",
    "air_yards_share_w1_3",
    "rush_ypc_w1_3",
    "rec_ypt_w1_3",
    "catch_rate_w1_3",
    "w3_ppr",
    "ppr_trend",
    "w3_opportunities",
    "team_points_pg_w1_3",
    "team_carries_pg_w1_3",
    "team_targets_pg_w1_3",
    "team_pass_share_proxy_w1_3",
    "matchup_index",
    "opp_pos_ppr_allowed_pg",
    "opp_pos_carries_allowed_pg",
    "opp_pos_targets_allowed_pg",
    "opp_pos_rush_yards_allowed_pg",
    "opp_pos_rec_yards_allowed_pg",
    "opp_pos_rush_tds_allowed_pg",
    "opp_pos_rec_tds_allowed_pg",
    "week",
    "is_home",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    n = numeric(num)
    d = numeric(den)
    out = pd.Series(np.nan, index=n.index, dtype=float)
    mask = d.abs() > 1e-12
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def load_inputs(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading weekly player data: {min(seasons)}-{max(seasons)}")
    players = nfl.load_player_stats(
        seasons=seasons,
        summary_level="week",
    ).to_pandas()

    print("Loading schedules...")
    schedules = nfl.load_schedules(seasons).to_pandas()

    if "season_type" in players.columns:
        players = players[players["season_type"].eq("REG")].copy()
    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"].eq("REG")].copy()

    players["season"] = numeric(players["season"]).astype("Int64")
    players["week"] = numeric(players["week"]).astype("Int64")
    schedules["season"] = numeric(schedules["season"]).astype("Int64")
    schedules["week"] = numeric(schedules["week"]).astype("Int64")

    if "position" not in players.columns:
        raise RuntimeError("Weekly player data is missing position.")
    if "game_id" not in players.columns:
        raise RuntimeError("Weekly player data is missing game_id.")

    players["position"] = (
        players["position"].fillna("").astype(str).str.upper()
    )

    schedule_cols = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team", "home_score", "away_score",
    ]
    keep = [c for c in schedule_cols if c in schedules.columns]
    sched_small = schedules[keep].drop_duplicates("game_id").copy()

    extra = [c for c in ["gameday", "home_team", "away_team"] if c not in players.columns]
    if extra:
        merge_cols = ["game_id"] + extra
        players = players.merge(
            sched_small[merge_cols],
            on="game_id",
            how="left",
        )

    if "opponent_team" not in players.columns:
        players["opponent_team"] = np.where(
            players["team"].eq(players["home_team"]),
            players["away_team"],
            players["home_team"],
        )

    for column in BOX_COLUMNS:
        if column not in players.columns:
            players[column] = 0.0
        players[column] = numeric(players[column]).fillna(0.0)

    if "fantasy_points_ppr" in players.columns:
        players["ppr"] = numeric(players["fantasy_points_ppr"]).fillna(0.0)
    else:
        fumbles_lost = (
            numeric(players["rushing_fumbles_lost"]).fillna(0.0)
            if "rushing_fumbles_lost" in players.columns
            else pd.Series(0.0, index=players.index)
        )
        players["ppr"] = (
            players["receptions"]
            + 0.1 * (players["rushing_yards"] + players["receiving_yards"])
            + 6.0 * (players["rushing_tds"] + players["receiving_tds"])
            - 2.0 * fumbles_lost
        )

    # Usage shares are based only on same-game team opportunities.
    team_game = (
        players.groupby(["game_id", "team"], as_index=False)
        .agg(
            team_carries=("carries", "sum"),
            team_targets=("targets", "sum"),
            team_air_yards=("receiving_air_yards", "sum"),
        )
    )
    players = players.merge(team_game, on=["game_id", "team"], how="left")
    players["rush_share"] = safe_div(players["carries"], players["team_carries"])
    players["target_share"] = safe_div(players["targets"], players["team_targets"])
    players["air_yards_share"] = safe_div(
        players["receiving_air_yards"],
        players["team_air_yards"],
    )
    players["opportunities"] = players["carries"] + players["targets"]

    if "player_display_name" not in players.columns:
        players["player_display_name"] = players.get(
            "player_name",
            players["player_id"],
        )

    schedules["gameday"] = pd.to_datetime(
        schedules.get("gameday"),
        errors="coerce",
    )
    return players, schedules


def team_schedule(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    s = schedules[
        schedules["season"].eq(season)
        & schedules["week"].between(1, FORECAST_END_WEEK)
    ].copy()

    home = s[["game_id", "week", "gameday", "home_team", "away_team"]].copy()
    home = home.rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home["is_home"] = 1

    away = s[["game_id", "week", "gameday", "home_team", "away_team"]].copy()
    away = away.rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away["is_home"] = 0

    return pd.concat([home, away], ignore_index=True)


def team_first3_context(
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    p = players[
        players["season"].eq(season)
        & players["week"].between(1, SNAPSHOT_END_WEEK)
    ].copy()

    usage = (
        p.groupby(["team", "game_id"], as_index=False)
        .agg(
            team_carries=("carries", "sum"),
            team_targets=("targets", "sum"),
        )
    )

    s = schedules[
        schedules["season"].eq(season)
        & schedules["week"].between(1, SNAPSHOT_END_WEEK)
    ].copy()

    score_rows = []
    for _, row in s.iterrows():
        home_score = pd.to_numeric(row.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(row.get("away_score"), errors="coerce")
        score_rows.append(
            {
                "team": row.get("home_team"),
                "game_id": row.get("game_id"),
                "points": home_score,
            }
        )
        score_rows.append(
            {
                "team": row.get("away_team"),
                "game_id": row.get("game_id"),
                "points": away_score,
            }
        )
    scores = pd.DataFrame(score_rows)

    game = usage.merge(scores, on=["team", "game_id"], how="left")
    team = (
        game.groupby("team", as_index=False)
        .agg(
            team_points_pg_w1_3=("points", "mean"),
            team_carries_pg_w1_3=("team_carries", "mean"),
            team_targets_pg_w1_3=("team_targets", "mean"),
        )
    )
    denom = (
        team["team_targets_pg_w1_3"]
        + team["team_carries_pg_w1_3"]
    )
    team["team_pass_share_proxy_w1_3"] = safe_div(
        team["team_targets_pg_w1_3"],
        denom,
    )
    return team


def player_snapshot(
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    p = players[
        players["season"].eq(season)
        & players["week"].between(1, SNAPSHOT_END_WEEK)
        & players["position"].isin(POSITIONS)
    ].copy()

    if p.empty:
        return p

    p = p.sort_values(["player_id", "week", "game_id"])
    latest = (
        p.groupby("player_id", as_index=False)
        .tail(1)[
            ["player_id", "player_display_name", "position", "team"]
        ]
        .rename(columns={"player_display_name": "player_name"})
    )

    agg = (
        p.groupby("player_id", as_index=False)
        .agg(
            games_w1_3=("game_id", "nunique"),
            ppr_total_w1_3=("ppr", "sum"),
            ppr_pg_w1_3=("ppr", "mean"),
            ppr_sd_w1_3=("ppr", "std"),
            opportunities_pg_w1_3=("opportunities", "mean"),
            carries_pg_w1_3=("carries", "mean"),
            rushing_yards_pg_w1_3=("rushing_yards", "mean"),
            rushing_tds_pg_w1_3=("rushing_tds", "mean"),
            targets_pg_w1_3=("targets", "mean"),
            receptions_pg_w1_3=("receptions", "mean"),
            receiving_yards_pg_w1_3=("receiving_yards", "mean"),
            receiving_tds_pg_w1_3=("receiving_tds", "mean"),
            receiving_air_yards_pg_w1_3=("receiving_air_yards", "mean"),
            receiving_yac_pg_w1_3=("receiving_yards_after_catch", "mean"),
            rush_share_w1_3=("rush_share", "mean"),
            target_share_w1_3=("target_share", "mean"),
            air_yards_share_w1_3=("air_yards_share", "mean"),
        )
    )

    sums = (
        p.groupby("player_id", as_index=False)
        .agg(
            carries_sum=("carries", "sum"),
            rushing_yards_sum=("rushing_yards", "sum"),
            targets_sum=("targets", "sum"),
            receptions_sum=("receptions", "sum"),
            receiving_yards_sum=("receiving_yards", "sum"),
        )
    )
    sums["rush_ypc_w1_3"] = safe_div(
        sums["rushing_yards_sum"],
        sums["carries_sum"],
    )
    sums["rec_ypt_w1_3"] = safe_div(
        sums["receiving_yards_sum"],
        sums["targets_sum"],
    )
    sums["catch_rate_w1_3"] = safe_div(
        sums["receptions_sum"],
        sums["targets_sum"],
    )
    sums = sums[
        [
            "player_id",
            "rush_ypc_w1_3",
            "rec_ypt_w1_3",
            "catch_rate_w1_3",
        ]
    ]

    week_one = (
        p[p["week"].eq(1)]
        .groupby("player_id")["ppr"]
        .sum()
    )
    week_three = (
        p[p["week"].eq(3)]
        .groupby("player_id")["ppr"]
        .sum()
    )
    week_three_opp = (
        p[p["week"].eq(3)]
        .groupby("player_id")["opportunities"]
        .sum()
    )

    snapshot = latest.merge(agg, on="player_id", how="left")
    snapshot = snapshot.merge(sums, on="player_id", how="left")
    snapshot["w3_ppr"] = snapshot["player_id"].map(week_three).fillna(0.0)
    snapshot["w3_opportunities"] = (
        snapshot["player_id"].map(week_three_opp).fillna(0.0)
    )
    w1 = snapshot["player_id"].map(week_one).fillna(0.0)
    snapshot["ppr_trend"] = (snapshot["w3_ppr"] - w1) / 2.0
    snapshot["ppr_sd_w1_3"] = snapshot["ppr_sd_w1_3"].fillna(0.0)

    team = team_first3_context(players, schedules, season)
    snapshot = snapshot.merge(team, on="team", how="left")
    snapshot["season"] = season

    # Fantasy-relevant pool. We keep emerging players but remove players with
    # virtually no Week 1-3 role.
    relevant = (
        (snapshot["position"].eq("RB") & (snapshot["opportunities_pg_w1_3"] >= 2.0))
        | (snapshot["position"].eq("WR") & (snapshot["targets_pg_w1_3"] >= 1.5))
    )
    snapshot = snapshot[
        snapshot["games_w1_3"].ge(2) & relevant
    ].copy()

    return snapshot


def defense_profiles(
    players: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    p = players[
        players["season"].eq(season)
        & players["week"].between(1, SNAPSHOT_END_WEEK)
        & players["position"].isin(POSITIONS)
    ].copy()

    if p.empty:
        return pd.DataFrame()

    game_pos = (
        p.groupby(["opponent_team", "position", "game_id"], as_index=False)
        .agg(
            pos_ppr=("ppr", "sum"),
            pos_carries=("carries", "sum"),
            pos_targets=("targets", "sum"),
            pos_rush_yards=("rushing_yards", "sum"),
            pos_rec_yards=("receiving_yards", "sum"),
            pos_rush_tds=("rushing_tds", "sum"),
            pos_rec_tds=("receiving_tds", "sum"),
        )
        .rename(columns={"opponent_team": "defense"})
    )

    profile = (
        game_pos.groupby(["defense", "position"], as_index=False)
        .agg(
            defense_games=("game_id", "nunique"),
            obs_ppr=("pos_ppr", "mean"),
            obs_carries=("pos_carries", "mean"),
            obs_targets=("pos_targets", "mean"),
            obs_rush_yards=("pos_rush_yards", "mean"),
            obs_rec_yards=("pos_rec_yards", "mean"),
            obs_rush_tds=("pos_rush_tds", "mean"),
            obs_rec_tds=("pos_rec_tds", "mean"),
        )
    )

    output = []
    for position, rows in profile.groupby("position"):
        league = {
            column: rows[column].mean()
            for column in [
                "obs_ppr", "obs_carries", "obs_targets",
                "obs_rush_yards", "obs_rec_yards",
                "obs_rush_tds", "obs_rec_tds",
            ]
        }

        rows = rows.copy()
        weight = (
            rows["defense_games"]
            / (rows["defense_games"] + DEFENSE_SHRINK_GAMES)
        )

        mapping = {
            "obs_ppr": "opp_pos_ppr_allowed_pg",
            "obs_carries": "opp_pos_carries_allowed_pg",
            "obs_targets": "opp_pos_targets_allowed_pg",
            "obs_rush_yards": "opp_pos_rush_yards_allowed_pg",
            "obs_rec_yards": "opp_pos_rec_yards_allowed_pg",
            "obs_rush_tds": "opp_pos_rush_tds_allowed_pg",
            "obs_rec_tds": "opp_pos_rec_tds_allowed_pg",
        }
        for raw, final in mapping.items():
            rows[final] = (
                weight * rows[raw]
                + (1.0 - weight) * league[raw]
            )

        league_ppr = float(league["obs_ppr"]) if pd.notna(league["obs_ppr"]) else 0.0
        rows["matchup_index"] = np.where(
            league_ppr > 0,
            100.0 * rows["opp_pos_ppr_allowed_pg"] / league_ppr,
            100.0,
        )
        output.append(
            rows[
                [
                    "defense", "position", "matchup_index",
                    *mapping.values(),
                ]
            ]
        )

    return pd.concat(output, ignore_index=True)


def schedule_rows_for_snapshot(
    snapshot: pd.DataFrame,
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    include_target: bool,
) -> pd.DataFrame:
    future = team_schedule(schedules, season)
    future = future[
        future["week"].between(
            FORECAST_START_WEEK,
            FORECAST_END_WEEK,
        )
    ].copy()

    profiles = defense_profiles(players, season)
    if profiles.empty or future.empty or snapshot.empty:
        return pd.DataFrame()

    rows = snapshot.merge(future, on="team", how="inner")
    rows = rows.merge(
        profiles,
        left_on=["opponent", "position"],
        right_on=["defense", "position"],
        how="left",
    )

    # A missing defense/position row means no usable Week 1-3 sample.
    # Neutral league-average matchup is safer than dropping the game.
    rows["matchup_index"] = numeric(rows["matchup_index"]).fillna(100.0)
    matchup_cols = [
        "opp_pos_ppr_allowed_pg",
        "opp_pos_carries_allowed_pg",
        "opp_pos_targets_allowed_pg",
        "opp_pos_rush_yards_allowed_pg",
        "opp_pos_rec_yards_allowed_pg",
        "opp_pos_rush_tds_allowed_pg",
        "opp_pos_rec_tds_allowed_pg",
    ]
    for column in matchup_cols:
        rows[column] = numeric(rows[column])
        rows[column] = rows.groupby("position")[column].transform(
            lambda s: s.fillna(s.median())
        )

    if include_target:
        actual = players[
            players["season"].eq(season)
            & players["week"].between(
                FORECAST_START_WEEK,
                FORECAST_END_WEEK,
            )
        ].copy()
        actual = (
            actual.groupby(["player_id", "week"], as_index=False)
            .agg(actual_ppr=("ppr", "sum"))
        )
        rows = rows.merge(
            actual,
            on=["player_id", "week"],
            how="left",
        )
        rows["actual_ppr"] = rows["actual_ppr"].fillna(0.0)

    return rows


def historical_dataset(
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    seasons: list[int],
) -> pd.DataFrame:
    parts = []
    for season in seasons:
        print(f"Building Week-3 snapshot rows for {season}...")
        snapshot = player_snapshot(players, schedules, season)
        rows = schedule_rows_for_snapshot(
            snapshot,
            players,
            schedules,
            season,
            include_target=True,
        )
        if not rows.empty:
            parts.append(rows)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def model_for_position() -> BlendRegressor:
    return BlendRegressor()


def walk_forward_validation(history: pd.DataFrame) -> dict:
    report = {}

    for position in POSITIONS:
        pos_rows = history[history["position"].eq(position)].copy()
        folds = []
        all_actual = []
        all_pred = []
        all_baseline = []

        for season in VALIDATION_SEASONS:
            train = pos_rows[pos_rows["season"] < season].copy()
            test = pos_rows[pos_rows["season"].eq(season)].copy()
            if len(train) < 500 or test.empty:
                continue

            model = model_for_position()
            model.fit(train[FEATURES], train["actual_ppr"])
            pred = np.maximum(0.0, model.predict(test[FEATURES]))
            baseline = np.maximum(
                0.0,
                numeric(test["ppr_pg_w1_3"]).fillna(0.0).to_numpy(),
            )
            actual = numeric(test["actual_ppr"]).fillna(0.0).to_numpy()

            model_mae = mean_absolute_error(actual, pred)
            baseline_mae = mean_absolute_error(actual, baseline)

            weekly_corr = pd.Series(pred).corr(
                pd.Series(actual),
                method="spearman",
            )

            player_eval = pd.DataFrame(
                {
                    "player_id": test["player_id"].to_numpy(),
                    "actual": actual,
                    "prediction": pred,
                    "baseline": baseline,
                }
            )
            player_totals = (
                player_eval.groupby("player_id", as_index=False)
                .agg(
                    actual=("actual", "sum"),
                    prediction=("prediction", "sum"),
                    baseline=("baseline", "sum"),
                )
            )
            ros_corr = player_totals["prediction"].corr(
                player_totals["actual"],
                method="spearman",
            )

            folds.append(
                {
                    "season": int(season),
                    "rows": int(len(test)),
                    "players": int(test["player_id"].nunique()),
                    "model_mae": round(float(model_mae), 3),
                    "baseline_mae": round(float(baseline_mae), 3),
                    "weekly_rank_corr": (
                        None if pd.isna(weekly_corr)
                        else round(float(weekly_corr), 4)
                    ),
                    "ros_rank_corr": (
                        None if pd.isna(ros_corr)
                        else round(float(ros_corr), 4)
                    ),
                }
            )
            all_actual.extend(actual.tolist())
            all_pred.extend(pred.tolist())
            all_baseline.extend(baseline.tolist())

        if all_actual:
            report[position] = {
                "status": "ok",
                "folds": folds,
                "rows": len(all_actual),
                "model_mae": round(
                    float(mean_absolute_error(all_actual, all_pred)),
                    3,
                ),
                "baseline_mae": round(
                    float(mean_absolute_error(all_actual, all_baseline)),
                    3,
                ),
                "model_beats_baseline": (
                    mean_absolute_error(all_actual, all_pred)
                    < mean_absolute_error(all_actual, all_baseline)
                ),
            }
        else:
            report[position] = {
                "status": "insufficient_history",
                "folds": folds,
            }

    return report


def project_current(
    history: pd.DataFrame,
    players: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = player_snapshot(players, schedules, season)
    future = schedule_rows_for_snapshot(
        snapshot,
        players,
        schedules,
        season,
        include_target=False,
    )
    if future.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    projected_parts = []
    for position in POSITIONS:
        train = history[history["position"].eq(position)].copy()
        test = future[future["position"].eq(position)].copy()
        if train.empty or test.empty:
            continue

        model = model_for_position()
        model.fit(train[FEATURES], train["actual_ppr"])
        test["projected_ppr"] = np.maximum(
            0.0,
            model.predict(test[FEATURES]),
        )
        projected_parts.append(test)

    weekly = pd.concat(projected_parts, ignore_index=True)
    weekly = weekly.sort_values(
        ["position", "week", "projected_ppr"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Schedule difficulty is team/position specific and is intentionally
    # separate from player talent/usage.
    sos = (
        weekly.groupby(["team", "position"], as_index=False)
        .agg(
            ros_games=("week", "nunique"),
            avg_matchup_index=("matchup_index", "mean"),
            easiest_matchup_index=("matchup_index", "max"),
            toughest_matchup_index=("matchup_index", "min"),
            easy_games=("matchup_index", lambda s: int((s >= 105).sum())),
            tough_games=("matchup_index", lambda s: int((s <= 95).sum())),
        )
    )
    sos["sos_rank"] = (
        sos.groupby("position")["avg_matchup_index"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    schedule_text = (
        weekly[
            ["team", "position", "week", "opponent", "is_home", "matchup_index"]
        ]
        .drop_duplicates()
        .sort_values(["team", "position", "week"])
        .groupby(["team", "position"])
        .apply(
            lambda g: " | ".join(
                (
                    f"W{int(row.week)} "
                    f"{'vs' if int(row.is_home) else '@'} "
                    f"{row.opponent} "
                    f"({row.matchup_index:.0f})"
                )
                for row in g.itertuples()
            ),
            include_groups=False,
        )
        .rename("schedule")
        .reset_index()
    )
    sos = sos.merge(schedule_text, on=["team", "position"], how="left")

    rankings = (
        weekly.groupby(
            [
                "player_id", "player_name", "position", "team",
                "ppr_pg_w1_3", "opportunities_pg_w1_3",
                "carries_pg_w1_3", "targets_pg_w1_3",
            ],
            as_index=False,
        )
        .agg(
            projected_ppr_w4_16=("projected_ppr", "sum"),
            projected_ppr_pg_w4_16=("projected_ppr", "mean"),
            ros_games=("week", "nunique"),
            avg_matchup_index=("matchup_index", "mean"),
        )
    )
    rankings["fantasy_rank"] = (
        rankings.groupby("position")["projected_ppr_w4_16"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    rankings["sos_rank"] = (
        rankings.groupby("position")["avg_matchup_index"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    best_week = (
        weekly.sort_values(
            ["player_id", "projected_ppr"],
            ascending=[True, False],
        )
        .groupby("player_id", as_index=False)
        .first()[
            ["player_id", "week", "opponent", "projected_ppr"]
        ]
        .rename(
            columns={
                "week": "best_projected_week",
                "opponent": "best_projected_opponent",
                "projected_ppr": "best_projected_ppr",
            }
        )
    )
    rankings = rankings.merge(best_week, on="player_id", how="left")
    rankings = rankings.sort_values(
        ["position", "fantasy_rank"]
    ).reset_index(drop=True)

    weekly_keep = [
        "season", "week", "player_id", "player_name", "position",
        "team", "opponent", "is_home", "matchup_index",
        "projected_ppr",
    ]
    weekly = weekly[[c for c in weekly_keep if c in weekly.columns]].copy()

    return rankings, sos, weekly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-season", type=int, default=START_SEASON)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cache-dir", default=".nfl_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_cache(Path(args.cache_dir))

    historical_seasons = list(range(args.start_season, args.season))
    seasons = historical_seasons + [args.season]

    players, schedules = load_inputs(seasons)
    history = historical_dataset(
        players,
        schedules,
        historical_seasons,
    )
    if history.empty:
        raise RuntimeError("No historical fantasy rows were built.")

    validation = walk_forward_validation(history)
    rankings, sos, weekly = project_current(
        history,
        players,
        schedules,
        args.season,
    )
    if rankings.empty:
        raise RuntimeError("No current fantasy projections were generated.")

    rankings.to_csv(
        output_dir / "fantasy_ros_rankings.csv",
        index=False,
    )
    sos.to_csv(
        output_dir / "fantasy_ros_sos.csv",
        index=False,
    )
    weekly.to_csv(
        output_dir / "fantasy_ros_weekly.csv",
        index=False,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "season": args.season,
        "snapshot_weeks": [1, SNAPSHOT_END_WEEK],
        "forecast_weeks": [
            FORECAST_START_WEEK,
            FORECAST_END_WEEK,
        ],
        "training_start_season": args.start_season,
        "validation": validation,
        "sos_definition": (
            "Weeks 4-16 opponents ranked by RB/WR PPR allowed in Weeks 1-3, "
            "shrunk 50% toward the position league average after a three-game sample. "
            "Higher matchup index = easier fantasy matchup."
        ),
        "notes": [
            "Current-season player/team/defense features stop after Week 3.",
            "Historical training seasons are built using the same Week-3 snapshot rule.",
            "Future schedule is known information; future game results are never model features.",
            "Weekly targets are PPR points, with zero points when a snapshot player does not record stats in a scheduled team game.",
            "No route-run or snap-count feature is currently available in the repository.",
            "Projections assume the player's Week-3 team/role is the best available representation of rest-of-season opportunity.",
        ],
    }
    with (output_dir / "fantasy_ros_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("\nFANTASY ROS MODEL COMPLETE")
    print("=" * 72)
    for position in POSITIONS:
        top = rankings[rankings["position"].eq(position)].head(12)
        print(f"\nTop {position}s, Weeks 4-16:")
        print(
            top[
                [
                    "fantasy_rank", "player_name", "team",
                    "projected_ppr_pg_w4_16",
                    "projected_ppr_w4_16", "sos_rank",
                    "avg_matchup_index",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
