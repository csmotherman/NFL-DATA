#!/usr/bin/env python3
"""
NFL Player Props
================

Leakage-safe player prop projection pipeline using free nflverse data.

Markets projected:
- QB pass attempts
- QB completions
- QB passing yards
- rushing attempts
- rushing yards
- receptions
- receiving yards

Core ideas:
- player rolling usage + efficiency
- player opportunity shares
- team pace/pass tendency
- opponent pass/run defense
- game spread/total context
- walk-forward historical validation
- automatic model-vs-baseline blending
- optional prop-line ingestion for edge calculations

This produces projections and research edges. It does not claim that an
edge is profitable until actual historical sportsbook prop lines are
available for a true market backtest.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import mean_absolute_error

from nfl_edge import (
    BlendRegressor,
    RAW_FEATURES,
    add_pregame_rolling,
    build_team_game_stats,
    configure_cache,
    infer_target_week,
    latest_team_state,
    load_data,
    normalize_schedule,
)


START_SEASON = 2018
PLAYER_SHORT_WINDOW = 3
PLAYER_LONG_WINDOW = 6
MIN_PLAYER_HISTORY = 2
WALK_FORWARD_SEASONS = [2023, 2024, 2025]

PROP_CONFIG = {
    "pass_attempts": {
        "target": "attempts",
        "baseline_feature": "attempts",
        "positions": {"QB"},
        "min_usage_feature": "p6_attempts",
        "min_usage": 12.0,
    },
    "completions": {
        "target": "completions",
        "baseline_feature": "completions",
        "positions": {"QB"},
        "min_usage_feature": "p6_attempts",
        "min_usage": 12.0,
    },
    "passing_yards": {
        "target": "passing_yards",
        "baseline_feature": "passing_yards",
        "positions": {"QB"},
        "min_usage_feature": "p6_attempts",
        "min_usage": 12.0,
    },
    "rush_attempts": {
        "target": "carries",
        "baseline_feature": "carries",
        "positions": {"QB", "RB", "FB", "WR"},
        "min_usage_feature": "p6_carries",
        "min_usage": 2.0,
    },
    "rushing_yards": {
        "target": "rushing_yards",
        "baseline_feature": "rushing_yards",
        "positions": {"QB", "RB", "FB", "WR"},
        "min_usage_feature": "p6_carries",
        "min_usage": 2.0,
    },
    "receptions": {
        "target": "receptions",
        "baseline_feature": "receptions",
        "positions": {"RB", "FB", "WR", "TE"},
        "min_usage_feature": "p6_targets",
        "min_usage": 1.5,
    },
    "receiving_yards": {
        "target": "receiving_yards",
        "baseline_feature": "receiving_yards",
        "positions": {"RB", "FB", "WR", "TE"},
        "min_usage_feature": "p6_targets",
        "min_usage": 1.5,
    },
}

PLAYER_RAW = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_epa",
    "passing_cpoe",
    "carries",
    "rushing_yards",
    "rushing_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_epa",
    "pass_attempt_share",
    "rush_share",
    "target_share",
    "air_yards_share",
    "pass_ypa",
    "rush_ypc",
    "rec_ypt",
    "catch_rate",
]

TEAM_FEATURES = [
    "team_pre_off_pass_rate",
    "team_pre_off_plays",
    "team_pre_off_pass_epa_db",
    "team_pre_off_rush_epa_att",
    "team_pre_off_pass_yd_db",
    "team_pre_off_rush_yd_att",
    "opp_pre_def_pass_epa_db_allowed",
    "opp_pre_def_rush_epa_att_allowed",
    "opp_pre_def_pass_yd_db_allowed",
    "opp_pre_def_rush_yd_att_allowed",
    "opp_pre_def_completion_rate_allowed",
    "opp_pre_def_sack_rate_allowed",
    "opp_pre_def_pass_rate_faced",
    "opp_pre_def_plays_faced",
]

POSITION_FEATURES = [
    "is_qb",
    "is_rb",
    "is_wr",
    "is_te",
]

CONTEXT_FEATURES = [
    "week",
    "is_home",
    "rest_days",
    "team_spread",
    "game_total",
    "team_implied_points",
]

PLAYER_FEATURES = (
    [f"p3_{x}" for x in PLAYER_RAW]
    + [f"p6_{x}" for x in PLAYER_RAW]
    + [f"p6sd_{x}" for x in PLAYER_RAW]
    + [f"season_{x}" for x in PLAYER_RAW]
    + TEAM_FEATURES
    + POSITION_FEATURES
    + CONTEXT_FEATURES
)


def series(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def safe_div(num, den) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=n.index, dtype=float)
    mask = d.abs() > 1e-12
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def load_player_data(seasons: list[int]) -> pd.DataFrame:
    print(f"Loading player stats: {min(seasons)}-{max(seasons)}")
    p = nfl.load_player_stats(
        seasons=seasons,
        summary_level="week",
    ).to_pandas()

    if "season_type" in p.columns:
        p = p[p["season_type"].eq("REG")].copy()

    required = ["player_id", "season", "week", "team"]
    missing = [c for c in required if c not in p.columns]
    if missing:
        raise RuntimeError(f"Player stats missing required columns: {missing}")

    if "game_id" not in p.columns:
        raise RuntimeError("Player stats missing game_id; cannot create leakage-safe rows.")

    return p


def add_schedule_context(
    p: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "game_id",
        "gameday",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
        "spread_line",
        "total_line",
    ]
    s = schedules[[c for c in cols if c in schedules.columns]].copy()
    x = p.merge(s, on="game_id", how="left")

    x["gameday"] = pd.to_datetime(x["gameday"], errors="coerce")
    x["is_home"] = (x["team"] == x["home_team"]).astype(int)

    x["rest_days"] = np.where(
        x["is_home"].eq(1),
        pd.to_numeric(x["home_rest"], errors="coerce"),
        pd.to_numeric(x["away_rest"], errors="coerce"),
    )

    spread = pd.to_numeric(x["spread_line"], errors="coerce")
    total = pd.to_numeric(x["total_line"], errors="coerce")

    # nflverse schedule spread_line is used by the game model as home margin.
    x["team_spread"] = np.where(x["is_home"].eq(1), spread, -spread)
    x["game_total"] = total
    x["team_implied_points"] = (total + x["team_spread"]) / 2.0

    return x


def add_player_usage_features(p: pd.DataFrame) -> pd.DataFrame:
    x = p.copy()

    # Ensure core box-score columns exist.
    for c in [
        "attempts",
        "completions",
        "passing_yards",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_epa",
        "passing_cpoe",
        "carries",
        "rushing_yards",
        "rushing_epa",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_epa",
    ]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # Team usage denominators built directly from player stats.
    team_usage = (
        x.groupby(["game_id", "team"], as_index=False)
        .agg(
            team_attempts=("attempts", "sum"),
            team_carries=("carries", "sum"),
            team_targets=("targets", "sum"),
            team_receiving_air_yards=("receiving_air_yards", "sum"),
        )
    )

    x = x.merge(team_usage, on=["game_id", "team"], how="left")

    x["pass_attempt_share"] = safe_div(x["attempts"], x["team_attempts"])
    x["rush_share"] = safe_div(x["carries"], x["team_carries"])
    x["target_share"] = safe_div(x["targets"], x["team_targets"])
    x["air_yards_share"] = safe_div(
        x["receiving_air_yards"],
        x["team_receiving_air_yards"],
    )

    x["pass_ypa"] = safe_div(x["passing_yards"], x["attempts"])
    x["rush_ypc"] = safe_div(x["rushing_yards"], x["carries"])
    x["rec_ypt"] = safe_div(x["receiving_yards"], x["targets"])
    x["catch_rate"] = safe_div(x["receptions"], x["targets"])

    if "player_display_name" not in x.columns:
        x["player_display_name"] = x.get("player_name", x["player_id"])

    if "position" not in x.columns:
        x["position"] = "UNK"

    x["position"] = x["position"].fillna("UNK").astype(str).str.upper()

    return x


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    pos = x["position"].fillna("UNK").astype(str).str.upper()
    x["is_qb"] = pos.eq("QB").astype(int)
    x["is_rb"] = pos.isin(["RB", "FB"]).astype(int)
    x["is_wr"] = pos.eq("WR").astype(int)
    x["is_te"] = pos.eq("TE").astype(int)
    return x


def add_player_pregame_features(p: pd.DataFrame) -> pd.DataFrame:
    x = p.sort_values(
        ["player_id", "gameday", "season", "week", "game_id"]
    ).copy()

    for feature in PLAYER_RAW:
        if feature not in x.columns:
            x[feature] = np.nan

        x[f"p3_{feature}"] = (
            x.groupby("player_id", group_keys=False)[feature]
            .transform(
                lambda s: s.shift(1).rolling(
                    PLAYER_SHORT_WINDOW,
                    min_periods=1,
                ).mean()
            )
        )

        x[f"p6_{feature}"] = (
            x.groupby("player_id", group_keys=False)[feature]
            .transform(
                lambda s: s.shift(1).rolling(
                    PLAYER_LONG_WINDOW,
                    min_periods=MIN_PLAYER_HISTORY,
                ).mean()
            )
        )

        x[f"p6sd_{feature}"] = (
            x.groupby("player_id", group_keys=False)[feature]
            .transform(
                lambda s: s.shift(1).rolling(
                    PLAYER_LONG_WINDOW,
                    min_periods=MIN_PLAYER_HISTORY,
                ).std()
            )
        )

        # Season-to-date, shifted so the current game never leaks.
        x[f"season_{feature}"] = (
            x.groupby(["player_id", "season"], group_keys=False)[feature]
            .transform(
                lambda s: s.shift(1).expanding(min_periods=1).mean()
            )
        )

    x["player_games_before"] = x.groupby("player_id").cumcount()
    x = add_position_features(x)
    return x


def latest_player_state(p: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = []

    for player_id, g in p.sort_values("gameday").groupby("player_id"):
        current = g[g["season"].eq(season)]
        if current.empty:
            continue

        latest = current.iloc[-1]
        career = g[g["gameday"] <= latest["gameday"]]
        short = career.tail(PLAYER_SHORT_WINDOW)
        long = career.tail(PLAYER_LONG_WINDOW)

        row = {
            "player_id": player_id,
            "player_name": latest.get(
                "player_display_name",
                latest.get("player_name", player_id),
            ),
            "position": str(latest.get("position", "UNK")).upper(),
            "team": latest["team"],
            "games_current_season": int(len(current)),
            "games_long_window": int(len(long)),
        }

        for feature in PLAYER_RAW:
            row[f"p3_{feature}"] = pd.to_numeric(
                short[feature], errors="coerce"
            ).mean()
            row[f"p6_{feature}"] = pd.to_numeric(
                long[feature], errors="coerce"
            ).mean()
            row[f"p6sd_{feature}"] = pd.to_numeric(
                long[feature], errors="coerce"
            ).std()
            row[f"season_{feature}"] = pd.to_numeric(
                current[feature], errors="coerce"
            ).mean()

        rows.append(row)

    state = pd.DataFrame(rows)
    if state.empty:
        return state
    state = add_position_features(state)
    return state


def historical_team_context(
    team_pregame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_cols = {
        "pre_off_pass_rate": "team_pre_off_pass_rate",
        "pre_off_plays": "team_pre_off_plays",
        "pre_off_pass_epa_db": "team_pre_off_pass_epa_db",
        "pre_off_rush_epa_att": "team_pre_off_rush_epa_att",
        "pre_off_pass_yd_db": "team_pre_off_pass_yd_db",
        "pre_off_rush_yd_att": "team_pre_off_rush_yd_att",
    }

    opp_cols = {
        "pre_def_pass_epa_db_allowed": "opp_pre_def_pass_epa_db_allowed",
        "pre_def_rush_epa_att_allowed": "opp_pre_def_rush_epa_att_allowed",
        "pre_def_pass_yd_db_allowed": "opp_pre_def_pass_yd_db_allowed",
        "pre_def_rush_yd_att_allowed": "opp_pre_def_rush_yd_att_allowed",
        "pre_def_completion_rate_allowed": "opp_pre_def_completion_rate_allowed",
        "pre_def_sack_rate_allowed": "opp_pre_def_sack_rate_allowed",
        "pre_def_pass_rate_faced": "opp_pre_def_pass_rate_faced",
        "pre_def_plays_faced": "opp_pre_def_plays_faced",
    }

    team_keep = ["game_id", "team"] + [
        c for c in team_cols if c in team_pregame.columns
    ]
    team_ctx = team_pregame[team_keep].copy().rename(columns=team_cols)

    opp_keep = ["game_id", "team"] + [
        c for c in opp_cols if c in team_pregame.columns
    ]
    opp_ctx = team_pregame[opp_keep].copy().rename(
        columns={"team": "opponent_team", **opp_cols}
    )

    return team_ctx, opp_ctx


def add_team_context_to_history(
    players: pd.DataFrame,
    team_pregame: pd.DataFrame,
) -> pd.DataFrame:
    team_ctx, opp_ctx = historical_team_context(team_pregame)
    x = players.merge(team_ctx, on=["game_id", "team"], how="left")
    x = x.merge(
        opp_ctx,
        on=["game_id", "opponent_team"],
        how="left",
    )

    for c in PLAYER_FEATURES:
        if c not in x.columns:
            x[c] = np.nan

    return x


def current_team_context(
    team_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_cols = {
        "pre_off_pass_rate": "team_pre_off_pass_rate",
        "pre_off_plays": "team_pre_off_plays",
        "pre_off_pass_epa_db": "team_pre_off_pass_epa_db",
        "pre_off_rush_epa_att": "team_pre_off_rush_epa_att",
        "pre_off_pass_yd_db": "team_pre_off_pass_yd_db",
        "pre_off_rush_yd_att": "team_pre_off_rush_yd_att",
    }
    opp_cols = {
        "pre_def_pass_epa_db_allowed": "opp_pre_def_pass_epa_db_allowed",
        "pre_def_rush_epa_att_allowed": "opp_pre_def_rush_epa_att_allowed",
        "pre_def_pass_yd_db_allowed": "opp_pre_def_pass_yd_db_allowed",
        "pre_def_rush_yd_att_allowed": "opp_pre_def_rush_yd_att_allowed",
        "pre_def_completion_rate_allowed": "opp_pre_def_completion_rate_allowed",
        "pre_def_sack_rate_allowed": "opp_pre_def_sack_rate_allowed",
        "pre_def_pass_rate_faced": "opp_pre_def_pass_rate_faced",
        "pre_def_plays_faced": "opp_pre_def_plays_faced",
    }

    a = team_state[
        ["team"] + [c for c in team_cols if c in team_state.columns]
    ].copy().rename(columns=team_cols)

    b = team_state[
        ["team"] + [c for c in opp_cols if c in team_state.columns]
    ].copy().rename(columns={"team": "opponent_team", **opp_cols})

    return a, b


def add_upcoming_context(
    player_state: pd.DataFrame,
    team_state: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    games = schedules[
        schedules["season"].eq(season)
        & schedules["week"].eq(week)
    ].copy()

    if games.empty:
        raise RuntimeError(f"No upcoming games found for {season} Week {week}")

    # Team -> opponent mapping for target week.
    home = games[
        [
            "game_id", "week", "gameday",
            "home_team", "away_team",
            "home_rest", "spread_line", "total_line",
        ]
    ].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent_team",
        "home_rest": "rest_days",
    })
    home["is_home"] = 1
    home["team_spread"] = home["spread_line"]

    away = games[
        [
            "game_id", "week", "gameday",
            "away_team", "home_team",
            "away_rest", "spread_line", "total_line",
        ]
    ].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent_team",
        "away_rest": "rest_days",
    })
    away["is_home"] = 0
    away["team_spread"] = -away["spread_line"]

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["game_total"] = team_games["total_line"]
    team_games["team_implied_points"] = (
        team_games["game_total"] + team_games["team_spread"]
    ) / 2.0

    x = player_state.merge(team_games, on="team", how="inner")

    team_ctx, opp_ctx = current_team_context(team_state)
    x = x.merge(team_ctx, on="team", how="left")
    x = x.merge(opp_ctx, on="opponent_team", how="left")

    for c in PLAYER_FEATURES:
        if c not in x.columns:
            x[c] = np.nan

    return x


def baseline_prediction(df: pd.DataFrame, feature: str) -> np.ndarray:
    short = pd.to_numeric(df[f"p3_{feature}"], errors="coerce")
    long = pd.to_numeric(df[f"p6_{feature}"], errors="coerce")

    base = 0.60 * short + 0.40 * long
    base = base.where(base.notna(), short)
    base = base.where(base.notna(), long)

    return base.to_numpy(dtype=float)


def eligible_rows(
    df: pd.DataFrame,
    market: str,
    require_target: bool = True,
) -> pd.DataFrame:
    cfg = PROP_CONFIG[market]
    x = df.copy()

    if "position" not in x.columns:
        x["position"] = "UNK"

    mask = x["position"].astype(str).str.upper().isin(cfg["positions"])
    usage = pd.to_numeric(
        x.get(cfg["min_usage_feature"], np.nan),
        errors="coerce",
    )
    mask &= usage >= cfg["min_usage"]

    if require_target:
        target = cfg["target"]
        mask &= pd.to_numeric(x[target], errors="coerce").notna()

    # Need at least some real player history.
    if "player_games_before" in x.columns:
        mask &= x["player_games_before"] >= MIN_PLAYER_HISTORY
    elif "games_long_window" in x.columns:
        mask &= x["games_long_window"] >= MIN_PLAYER_HISTORY

    return x[mask].copy()


def optimize_blend_weight(
    actual: np.ndarray,
    model_pred: np.ndarray,
    baseline: np.ndarray,
) -> tuple[float, float]:
    valid = (
        np.isfinite(actual)
        & np.isfinite(model_pred)
        & np.isfinite(baseline)
    )
    if valid.sum() == 0:
        return 0.5, np.nan

    y = actual[valid]
    m = model_pred[valid]
    b = baseline[valid]

    best_w = 0.0
    best_mae = float("inf")

    for w in np.linspace(0, 1, 21):
        pred = w * m + (1 - w) * b
        mae = mean_absolute_error(y, pred)
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)

    return best_w, float(best_mae)


def walk_forward_prop(
    data: pd.DataFrame,
    market: str,
) -> tuple[dict, pd.DataFrame]:
    cfg = PROP_CONFIG[market]
    target = cfg["target"]
    rows = []

    available_seasons = sorted(
        int(s) for s in data["season"].dropna().unique()
    )

    for test_season in WALK_FORWARD_SEASONS:
        if test_season not in available_seasons:
            continue

        train = eligible_rows(
            data[data["season"] < test_season],
            market,
            require_target=True,
        )
        test = eligible_rows(
            data[data["season"] == test_season],
            market,
            require_target=True,
        )

        if len(train) < 300 or len(test) < 40:
            continue

        model = BlendRegressor().fit(
            train[PLAYER_FEATURES],
            train[target],
        )
        model_pred = model.predict(test[PLAYER_FEATURES])
        baseline = baseline_prediction(test, cfg["baseline_feature"])

        fold = test[
            [
                "season", "week", "game_id",
                "player_id", "player_display_name",
                "position", "team", "opponent_team",
            ]
        ].copy()
        fold["market"] = market
        fold["actual"] = pd.to_numeric(test[target], errors="coerce").to_numpy()
        fold["model_pred"] = model_pred
        fold["baseline_pred"] = baseline
        rows.append(fold)

    if not rows:
        return {
            "status": "not_enough_walk_forward_data",
            "market": market,
        }, pd.DataFrame()

    oof = pd.concat(rows, ignore_index=True)
    valid = (
        oof["actual"].notna()
        & np.isfinite(oof["model_pred"])
        & np.isfinite(oof["baseline_pred"])
    )
    oof = oof[valid].copy()

    weight, blended_mae = optimize_blend_weight(
        oof["actual"].to_numpy(dtype=float),
        oof["model_pred"].to_numpy(dtype=float),
        oof["baseline_pred"].to_numpy(dtype=float),
    )
    oof["blended_pred"] = (
        weight * oof["model_pred"]
        + (1 - weight) * oof["baseline_pred"]
    )
    oof["residual"] = oof["actual"] - oof["blended_pred"]

    by_season = {}
    for season, g in oof.groupby("season"):
        by_season[str(int(season))] = {
            "games": int(len(g)),
            "model_mae": round(
                float(mean_absolute_error(g["actual"], g["model_pred"])),
                3,
            ),
            "baseline_mae": round(
                float(mean_absolute_error(g["actual"], g["baseline_pred"])),
                3,
            ),
            "blended_mae": round(
                float(mean_absolute_error(g["actual"], g["blended_pred"])),
                3,
            ),
        }

    model_mae = mean_absolute_error(oof["actual"], oof["model_pred"])
    baseline_mae = mean_absolute_error(oof["actual"], oof["baseline_pred"])
    residual_sd = float(oof["residual"].std(ddof=1))

    report = {
        "status": "ok",
        "market": market,
        "oof_rows": int(len(oof)),
        "seasons": by_season,
        "model_mae": round(float(model_mae), 3),
        "baseline_mae": round(float(baseline_mae), 3),
        "optimal_model_weight": round(float(weight), 2),
        "blended_mae": round(float(blended_mae), 3),
        "residual_sd": round(residual_sd, 3),
        "model_beats_baseline": bool(model_mae < baseline_mae),
    }

    return report, oof


def train_full_prop_model(
    data: pd.DataFrame,
    market: str,
) -> BlendRegressor:
    cfg = PROP_CONFIG[market]
    train = eligible_rows(data, market, require_target=True)
    if len(train) < 300:
        raise RuntimeError(
            f"Not enough training rows for {market}: {len(train)}"
        )

    model = BlendRegressor().fit(
        train[PLAYER_FEATURES],
        train[cfg["target"]],
    )
    return model


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def build_prop_projections(
    history: pd.DataFrame,
    upcoming: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    projections = []
    reports = {}
    all_oof = []

    for market, cfg in PROP_CONFIG.items():
        print(f"Training player prop model: {market}")
        report, oof = walk_forward_prop(history, market)
        reports[market] = report

        if not oof.empty:
            all_oof.append(oof)

        model = train_full_prop_model(history, market)
        current = eligible_rows(
            upcoming,
            market,
            require_target=False,
        )

        if current.empty:
            continue

        model_pred = model.predict(current[PLAYER_FEATURES])
        baseline = baseline_prediction(current, cfg["baseline_feature"])

        weight = report.get("optimal_model_weight", 0.5)
        residual_sd = report.get("residual_sd", np.nan)

        projection = (
            weight * model_pred
            + (1 - weight) * baseline
        )
        projection = np.maximum(projection, 0)

        out = current[
            [
                "season" if "season" in current.columns else "week",
            ]
        ].copy() if False else pd.DataFrame(index=current.index)

        out["week"] = current["week"].to_numpy()
        out["game_id"] = current["game_id"].to_numpy()
        out["player_id"] = current["player_id"].to_numpy()
        out["player_name"] = current["player_name"].to_numpy()
        out["position"] = current["position"].to_numpy()
        out["team"] = current["team"].to_numpy()
        out["opponent"] = current["opponent_team"].to_numpy()
        out["market"] = market
        out["projection"] = projection
        out["model_projection"] = model_pred
        out["recent_baseline"] = baseline
        out["model_weight"] = weight
        out["historical_residual_sd"] = residual_sd
        out["games_current_season"] = current["games_current_season"].to_numpy()
        out["team_spread"] = current["team_spread"].to_numpy()
        out["game_total"] = current["game_total"].to_numpy()
        out["team_implied_points"] = current["team_implied_points"].to_numpy()

        # Most relevant usage/matchup context by market.
        if market in {"pass_attempts", "completions", "passing_yards"}:
            out["recent_volume"] = current["p6_attempts"].to_numpy()
            out["usage_share"] = current["p6_pass_attempt_share"].to_numpy()
            out["opp_matchup_eff"] = current[
                "opp_pre_def_pass_epa_db_allowed"
            ].to_numpy()
            out["opp_matchup_yards"] = current[
                "opp_pre_def_pass_yd_db_allowed"
            ].to_numpy()
        elif market in {"rush_attempts", "rushing_yards"}:
            out["recent_volume"] = current["p6_carries"].to_numpy()
            out["usage_share"] = current["p6_rush_share"].to_numpy()
            out["opp_matchup_eff"] = current[
                "opp_pre_def_rush_epa_att_allowed"
            ].to_numpy()
            out["opp_matchup_yards"] = current[
                "opp_pre_def_rush_yd_att_allowed"
            ].to_numpy()
        else:
            out["recent_volume"] = current["p6_targets"].to_numpy()
            out["usage_share"] = current["p6_target_share"].to_numpy()
            out["opp_matchup_eff"] = current[
                "opp_pre_def_pass_epa_db_allowed"
            ].to_numpy()
            out["opp_matchup_yards"] = current[
                "opp_pre_def_pass_yd_db_allowed"
            ].to_numpy()

        projections.append(out.reset_index(drop=True))

    props = (
        pd.concat(projections, ignore_index=True)
        if projections
        else pd.DataFrame()
    )

    oof_all = (
        pd.concat(all_oof, ignore_index=True)
        if all_oof
        else pd.DataFrame()
    )

    return props, reports, oof_all


def read_prop_lines(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "player_prop_lines.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    lines = pd.read_csv(path)
    required = {"market", "line"}
    if not required.issubset(lines.columns):
        raise RuntimeError(
            "inputs/player_prop_lines.csv must contain at least market,line "
            "and either player_id or player_name."
        )

    if "player_id" not in lines.columns and "player_name" not in lines.columns:
        raise RuntimeError(
            "Prop lines file must contain player_id or player_name."
        )

    lines["market"] = lines["market"].astype(str).str.strip().str.lower()
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    return lines


def attach_market_lines(
    projections: pd.DataFrame,
    lines: pd.DataFrame,
) -> pd.DataFrame:
    if projections.empty or lines.empty:
        return pd.DataFrame()

    if "player_id" in lines.columns:
        join_cols = ["player_id", "market"]
    else:
        join_cols = ["player_name", "market"]

    edges = projections.merge(lines, on=join_cols, how="inner")

    if edges.empty:
        return edges

    edges["edge"] = edges["projection"] - edges["line"]
    edges["edge_pct_of_line"] = np.where(
        edges["line"].abs() > 1e-9,
        edges["edge"] / edges["line"],
        np.nan,
    )
    edges["edge_sigma"] = np.where(
        edges["historical_residual_sd"] > 1e-9,
        edges["edge"] / edges["historical_residual_sd"],
        np.nan,
    )

    # Normal-error approximation only. This is not a sportsbook-calibrated
    # win probability until historical prop line archives are available.
    edges["approx_p_over"] = edges["edge_sigma"].apply(
        lambda z: normal_cdf(z) if pd.notna(z) else np.nan
    )
    edges["approx_p_under"] = 1 - edges["approx_p_over"]
    edges["lean"] = np.where(edges["edge"] >= 0, "OVER", "UNDER")
    edges["approx_lean_probability"] = np.where(
        edges["lean"].eq("OVER"),
        edges["approx_p_over"],
        edges["approx_p_under"],
    )

    return edges.sort_values(
        ["approx_lean_probability", "edge_sigma"],
        ascending=False,
        na_position="last",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--start-season", type=int, default=START_SEASON)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--cache-dir", default=".nfl_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    configure_cache(Path(args.cache_dir))

    today = date.today()
    auto_season = today.year if today.month >= 7 else today.year - 1
    season = args.season or auto_season
    seasons = list(range(args.start_season, season + 1))

    team_stats, schedules = load_data(seasons)
    schedules = normalize_schedule(schedules)

    player_stats = load_player_data(seasons)
    player_stats = add_schedule_context(player_stats, schedules)
    player_stats = add_player_usage_features(player_stats)

    # Team historical context uses the same leakage-safe machinery as the
    # game model.
    team_games = build_team_game_stats(team_stats, schedules)
    team_games = team_games[
        team_games["points_for"].notna()
        & team_games["points_against"].notna()
    ].copy()
    team_pregame = add_pregame_rolling(team_games)
    team_state = latest_team_state(team_games)

    player_history = add_player_pregame_features(player_stats)
    player_history = add_team_context_to_history(
        player_history,
        team_pregame,
    )

    target_week = infer_target_week(schedules, season, args.week)

    p_state = latest_player_state(player_stats, season)
    upcoming = add_upcoming_context(
        p_state,
        team_state,
        schedules,
        season,
        target_week,
    )

    # add_upcoming_context does not naturally carry season because it joins
    # one target week only.
    upcoming["season"] = season

    props, reports, oof = build_prop_projections(
        player_history,
        upcoming,
    )

    props_path = output_dir / "latest_player_props.csv"
    props.to_csv(props_path, index=False)

    if not oof.empty:
        oof.to_csv(
            output_dir / "player_prop_walkforward_oof.csv",
            index=False,
        )

    report_payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "season": season,
        "target_week": target_week,
        "training_start_season": args.start_season,
        "short_window_games": PLAYER_SHORT_WINDOW,
        "long_window_games": PLAYER_LONG_WINDOW,
        "markets": reports,
        "notes": [
            "Every historical rolling player feature is shifted before the target game.",
            "Current projections blend ML and recent-player baselines using walk-forward out-of-sample MAE.",
            "Prop-line probabilities are normal-error approximations, not calibrated sportsbook win probabilities.",
            "True betting ROI requires an archive of historical prop lines and prices.",
        ],
    }

    with (output_dir / "player_prop_model_report.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report_payload, f, indent=2, allow_nan=False)

    lines = read_prop_lines(input_dir)
    edges = attach_market_lines(props, lines)

    edge_path = output_dir / "latest_player_prop_edges.csv"
    edges.to_csv(edge_path, index=False)

    print("\nPLAYER PROP PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Season/week: {season} / {target_week}")
    print(f"Player projections: {len(props):,}")
    print(f"Market-line edges: {len(edges):,}")

    print("\nWalk-forward validation:")
    for market, report in reports.items():
        if report.get("status") != "ok":
            print(f"- {market}: {report.get('status')}")
            continue
        print(
            f"- {market}: blended MAE {report['blended_mae']} | "
            f"baseline {report['baseline_mae']} | "
            f"ML weight {report['optimal_model_weight']:.0%}"
        )

    if not edges.empty:
        show = [
            "player_name", "team", "opponent", "market",
            "projection", "line", "edge", "edge_sigma",
            "lean", "approx_lean_probability",
        ]
        print("\nTop current prop discrepancies:")
        print(edges[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
