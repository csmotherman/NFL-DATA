#!/usr/bin/env python3
"""
NFL Edge
========

Free-data NFL betting research pipeline.

Data:
- nflverse team stats via nflreadpy
- nflverse schedules / market lines via nflreadpy

Design principles:
- no game predicts itself;
- rolling features are shifted for historical training rows;
- sportsbook lines are NOT model inputs;
- market disagreement is evaluated only after predictions;
- latest team state uses completed games only;
- early-season/small-sample uncertainty remains visible.

This is a research screener, not a guarantee of profitability.
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
from nflreadpy.config import update_config

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_strategy import (
    fit_final_market_model,
    predict_current_market,
    spread_bet_audit_for_season,
    total_bet_audit_for_season,
    walk_forward_strategy_validation,
)


ROLLING_GAMES = 8
MIN_ROLLING_GAMES = 3
START_SEASON = 2014
RANDOM_STATE = 42
MARKET_WALK_FORWARD_SEASONS = [2022, 2023, 2024, 2025]
ADJUSTED_EPA_RIDGE_ALPHA = 2.5

RAW_FEATURES = [
    "off_pass_epa_db",
    "off_rush_epa_att",
    "off_pass_yd_db",
    "off_rush_yd_att",
    "off_completion_rate",
    "off_cpoe",
    "off_sack_rate",
    "off_pass_fd_rate",
    "off_rush_fd_rate",
    "off_turnover_rate",
    "off_pass_rate",
    "off_plays",
    "points_for",
    "points_against",
    "margin",
    "def_pass_epa_db_allowed",
    "def_rush_epa_att_allowed",
    "def_pass_yd_db_allowed",
    "def_rush_yd_att_allowed",
    "def_completion_rate_allowed",
    "def_cpoe_allowed",
    "def_sack_rate_allowed",
    "def_pass_fd_rate_allowed",
    "def_rush_fd_rate_allowed",
    "def_turnover_rate_forced",
    "def_pass_rate_faced",
    "def_plays_faced",
]

MODEL_STATE_FEATURES = [f"pre_{c}" for c in RAW_FEATURES]


def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den.abs() > 1e-12, num / den, np.nan)


def col(df: pd.DataFrame, names: Iterable[str], default=0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def configure_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    update_config(
        cache_mode="filesystem",
        cache_dir=cache_dir,
        cache_duration=1800,
        verbose=True,
        timeout=60,
        user_agent="NFL-DATA/1.0",
    )


def load_data(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading team stats: {min(seasons)}-{max(seasons)}")
    stats = nfl.load_team_stats(seasons=seasons).to_pandas()

    print("Loading schedules...")
    schedules = nfl.load_schedules(seasons).to_pandas()

    if "season_type" in stats.columns:
        stats = stats[stats["season_type"].eq("REG")].copy()

    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"].eq("REG")].copy()

    return stats, schedules


def normalize_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    s = schedules.copy()
    s["gameday"] = pd.to_datetime(s["gameday"], errors="coerce")
    for c in [
        "home_score", "away_score", "spread_line", "total_line",
        "home_rest", "away_rest", "away_moneyline", "home_moneyline",
        "away_spread_odds", "home_spread_odds", "over_odds", "under_odds",
    ]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
        else:
            s[c] = np.nan

    if "result" not in s.columns:
        s["result"] = s["home_score"] - s["away_score"]
    else:
        s["result"] = pd.to_numeric(s["result"], errors="coerce")

    if "total" not in s.columns:
        s["total"] = s["home_score"] + s["away_score"]
    else:
        s["total"] = pd.to_numeric(s["total"], errors="coerce")

    if "location" not in s.columns:
        s["location"] = "Home"

    return s


def add_schedule_scores(stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    home = schedules[["game_id", "home_team", "away_team", "home_score", "away_score", "gameday", "week"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "sched_opponent",
        "home_score": "points_for",
        "away_score": "points_against",
    })

    away = schedules[["game_id", "away_team", "home_team", "away_score", "home_score", "gameday", "week"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "sched_opponent",
        "away_score": "points_for",
        "home_score": "points_against",
    })

    score_map = pd.concat([home, away], ignore_index=True)
    score_map["margin"] = score_map["points_for"] - score_map["points_against"]

    out = stats.merge(
        score_map,
        on=["game_id", "team"],
        how="left",
        suffixes=("", "_sched"),
    )

    if "week_sched" in out.columns:
        out["week"] = out["week"].fillna(out["week_sched"])
        out = out.drop(columns=["week_sched"])

    return out


def build_team_game_stats(stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    x = add_schedule_scores(stats.copy(), schedules)

    attempts = col(x, ["attempts"])
    completions = col(x, ["completions"])
    pass_yards = col(x, ["passing_yards"])
    pass_epa = col(x, ["passing_epa"])
    cpoe = col(x, ["passing_cpoe", "cpoe"], np.nan)
    sacks = col(x, ["sacks_suffered", "sacks"])
    pass_fd = col(x, ["passing_first_downs"])
    ints = col(x, ["passing_interceptions", "interceptions"])

    carries = col(x, ["carries"])
    rush_yards = col(x, ["rushing_yards"])
    rush_epa = col(x, ["rushing_epa"])
    rush_fd = col(x, ["rushing_first_downs"])
    rush_fumbles_lost = col(x, ["rushing_fumbles_lost"])
    sack_fumbles_lost = col(x, ["sack_fumbles_lost"])

    dropbacks = attempts + sacks
    plays = dropbacks + carries
    turnovers = ints + rush_fumbles_lost + sack_fumbles_lost

    x["off_pass_epa_db"] = safe_div(pass_epa, dropbacks)
    x["off_rush_epa_att"] = safe_div(rush_epa, carries)
    x["off_pass_yd_db"] = safe_div(pass_yards, dropbacks)
    x["off_rush_yd_att"] = safe_div(rush_yards, carries)
    x["off_completion_rate"] = safe_div(completions, attempts)
    x["off_cpoe"] = cpoe
    x["off_sack_rate"] = safe_div(sacks, dropbacks)
    x["off_pass_fd_rate"] = safe_div(pass_fd, dropbacks)
    x["off_rush_fd_rate"] = safe_div(rush_fd, carries)
    x["off_turnover_rate"] = safe_div(turnovers, plays)
    x["off_pass_rate"] = safe_div(dropbacks, plays)
    x["off_plays"] = plays

    keep = [
        "season", "week", "game_id", "gameday", "team", "opponent_team",
        "points_for", "points_against", "margin",
        "off_pass_epa_db", "off_rush_epa_att", "off_pass_yd_db",
        "off_rush_yd_att", "off_completion_rate", "off_cpoe",
        "off_sack_rate", "off_pass_fd_rate", "off_rush_fd_rate",
        "off_turnover_rate", "off_pass_rate", "off_plays",
    ]
    keep = [c for c in keep if c in x.columns]
    offense = x[keep].copy()

    opponent_metrics = [
        "off_pass_epa_db", "off_rush_epa_att", "off_pass_yd_db",
        "off_rush_yd_att", "off_completion_rate", "off_cpoe",
        "off_sack_rate", "off_pass_fd_rate", "off_rush_fd_rate",
        "off_turnover_rate", "off_pass_rate", "off_plays",
    ]

    opp = offense[["game_id", "team"] + opponent_metrics].copy()
    opp = opp.rename(columns={
        "team": "opponent_team",
        "off_pass_epa_db": "def_pass_epa_db_allowed",
        "off_rush_epa_att": "def_rush_epa_att_allowed",
        "off_pass_yd_db": "def_pass_yd_db_allowed",
        "off_rush_yd_att": "def_rush_yd_att_allowed",
        "off_completion_rate": "def_completion_rate_allowed",
        "off_cpoe": "def_cpoe_allowed",
        "off_sack_rate": "def_sack_rate_allowed",
        "off_pass_fd_rate": "def_pass_fd_rate_allowed",
        "off_rush_fd_rate": "def_rush_fd_rate_allowed",
        "off_turnover_rate": "def_turnover_rate_forced",
        "off_pass_rate": "def_pass_rate_faced",
        "off_plays": "def_plays_faced",
    })

    tg = offense.merge(opp, on=["game_id", "opponent_team"], how="left")
    tg["gameday"] = pd.to_datetime(tg["gameday"], errors="coerce")
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    return tg


def add_pregame_rolling(team_games: pd.DataFrame) -> pd.DataFrame:
    out = team_games.copy()

    for feature in RAW_FEATURES:
        if feature not in out.columns:
            out[feature] = np.nan

        out[f"pre_{feature}"] = (
            out.groupby("team", group_keys=False)[feature]
            .transform(
                lambda s: s.shift(1).rolling(
                    ROLLING_GAMES,
                    min_periods=MIN_ROLLING_GAMES,
                ).mean()
            )
        )

    out["pre_games_available"] = (
        out.groupby("team").cumcount().clip(upper=ROLLING_GAMES)
    )

    return out


def latest_team_state(team_games: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for team, g in team_games.sort_values("gameday").groupby("team"):
        recent = g.tail(ROLLING_GAMES)
        row = {"team": team, "games_in_window": len(recent)}
        for feature in RAW_FEATURES:
            row[f"pre_{feature}"] = pd.to_numeric(
                recent[feature], errors="coerce"
            ).mean()
        rows.append(row)

    state = pd.DataFrame(rows)

    # Z scores for interpretable matchup edges.
    z_cols = [
        "pre_off_pass_epa_db",
        "pre_off_rush_epa_att",
        "pre_off_pass_yd_db",
        "pre_off_rush_yd_att",
        "pre_def_pass_epa_db_allowed",
        "pre_def_rush_epa_att_allowed",
        "pre_def_pass_yd_db_allowed",
        "pre_def_rush_yd_att_allowed",
        "pre_off_pass_rate",
        "pre_off_plays",
        "pre_def_plays_faced",
    ]
    for c in z_cols:
        if c not in state.columns:
            continue
        sd = state[c].std()
        state[f"z_{c}"] = (
            (state[c] - state[c].mean()) / sd
            if pd.notna(sd) and sd > 0
            else 0.0
        )

    return state


def _weighted_metric_average(
    frame: pd.DataFrame,
    value_col: str,
    weight_col: str,
) -> float:
    """Play-weighted average for one EPA component."""
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    valid = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _fit_adjusted_epa_component(
    games: pd.DataFrame,
    teams: list[str],
    value_col: str,
    weight_col: str,
    alpha: float = ADJUSTED_EPA_RIDGE_ALPHA,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Estimate schedule-adjusted EPA with a regularized two-way model.

    Each team-game observation is decomposed into an offense effect and the
    opposing defense effect. Play counts are used as weights, and ridge
    shrinkage keeps early-season estimates from overreacting to tiny samples.

    Returns:
      offense: expected EPA versus an average defense
      defense: expected EPA allowed versus an average offense
      league_average: play-weighted league EPA for the component
    """
    sample = games[
        ["team", "opponent_team", value_col, weight_col]
    ].copy()
    sample[value_col] = pd.to_numeric(sample[value_col], errors="coerce")
    sample[weight_col] = pd.to_numeric(sample[weight_col], errors="coerce")
    sample = sample[
        sample["team"].notna()
        & sample["opponent_team"].notna()
        & sample[value_col].notna()
        & sample[weight_col].notna()
        & np.isfinite(sample[value_col])
        & np.isfinite(sample[weight_col])
        & (sample[weight_col] > 0)
    ].copy()

    if sample.empty:
        empty = {team: np.nan for team in teams}
        return empty, empty.copy(), np.nan

    y = sample[value_col].to_numpy(dtype=float)
    raw_weights = sample[weight_col].to_numpy(dtype=float)
    league_average = float(np.average(y, weights=raw_weights))

    # Normalize weights so alpha remains interpretable as the season grows.
    weights = raw_weights / raw_weights.mean()
    team_to_col = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    X = np.zeros((len(sample), n_teams * 2), dtype=float)

    for row_idx, (offense, defense) in enumerate(
        zip(sample["team"], sample["opponent_team"])
    ):
        X[row_idx, team_to_col[str(offense)]] = 1.0
        X[row_idx, n_teams + team_to_col[str(defense)]] = 1.0

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(
        X,
        y - league_average,
        sample_weight=weights,
    )

    offense = {
        team: league_average + float(model.coef_[team_to_col[team]])
        for team in teams
    }
    defense = {
        team: league_average
        + float(model.coef_[n_teams + team_to_col[team]])
        for team in teams
    }
    return offense, defense, league_average


def build_adjusted_epa_table(
    team_games: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Build current-season opponent-adjusted EPA for every team.

    EPA/pass uses dropbacks (attempts + sacks); EPA/rush uses carries.
    EPA/play combines those two components. Defensive values are EPA allowed,
    so lower is better on defense.
    """
    current = team_games[
        pd.to_numeric(team_games["season"], errors="coerce").eq(season)
    ].copy()

    if current.empty:
        return pd.DataFrame(columns=[
            "season", "team", "games",
            "raw_off_epa_per_play", "adj_off_epa_per_play",
            "raw_off_epa_per_pass", "adj_off_epa_per_pass",
            "raw_off_epa_per_rush", "adj_off_epa_per_rush",
            "raw_def_epa_per_play_allowed", "adj_def_epa_per_play_allowed",
            "raw_def_epa_per_pass_allowed", "adj_def_epa_per_pass_allowed",
            "raw_def_epa_per_rush_allowed", "adj_def_epa_per_rush_allowed",
        ])

    current["epa_plays"] = pd.to_numeric(
        current["off_plays"], errors="coerce"
    )
    current["epa_pass_plays"] = (
        pd.to_numeric(current["off_pass_rate"], errors="coerce")
        * current["epa_plays"]
    )
    current["epa_rush_plays"] = (
        current["epa_plays"] - current["epa_pass_plays"]
    )
    current["off_epa_per_play"] = safe_div(
        (
            pd.to_numeric(current["off_pass_epa_db"], errors="coerce")
            * current["epa_pass_plays"]
        )
        + (
            pd.to_numeric(current["off_rush_epa_att"], errors="coerce")
            * current["epa_rush_plays"]
        ),
        current["epa_plays"],
    )

    teams = sorted(
        {
            str(team)
            for team in pd.concat(
                [current["team"], current["opponent_team"]],
                ignore_index=True,
            ).dropna()
        }
    )

    adj_off_play, adj_def_play, league_epa_play = _fit_adjusted_epa_component(
        current, teams, "off_epa_per_play", "epa_plays"
    )
    adj_off_pass, adj_def_pass, league_epa_pass = _fit_adjusted_epa_component(
        current, teams, "off_pass_epa_db", "epa_pass_plays"
    )
    adj_off_rush, adj_def_rush, league_epa_rush = _fit_adjusted_epa_component(
        current, teams, "off_rush_epa_att", "epa_rush_plays"
    )

    # Flip the offense perspective so raw defensive numbers use the exact same
    # underlying observations and play weights as the adjusted model.
    defense_games = current[
        [
            "team", "opponent_team",
            "off_epa_per_play", "off_pass_epa_db", "off_rush_epa_att",
            "epa_plays", "epa_pass_plays", "epa_rush_plays",
        ]
    ].rename(columns={
        "team": "offense_team",
        "opponent_team": "team",
        "off_epa_per_play": "def_epa_per_play_allowed",
        "off_pass_epa_db": "def_epa_per_pass_allowed",
        "off_rush_epa_att": "def_epa_per_rush_allowed",
        "epa_plays": "def_plays",
        "epa_pass_plays": "def_pass_plays",
        "epa_rush_plays": "def_rush_plays",
    })

    rows = []
    for team in teams:
        offense_rows = current[current["team"].astype(str).eq(team)]
        defense_rows = defense_games[
            defense_games["team"].astype(str).eq(team)
        ]

        rows.append({
            "season": season,
            "team": team,
            "games": int(len(offense_rows)),
            "raw_off_epa_per_play": _weighted_metric_average(
                offense_rows, "off_epa_per_play", "epa_plays"
            ),
            "adj_off_epa_per_play": adj_off_play.get(team, league_epa_play),
            "raw_off_epa_per_pass": _weighted_metric_average(
                offense_rows, "off_pass_epa_db", "epa_pass_plays"
            ),
            "adj_off_epa_per_pass": adj_off_pass.get(team, league_epa_pass),
            "raw_off_epa_per_rush": _weighted_metric_average(
                offense_rows, "off_rush_epa_att", "epa_rush_plays"
            ),
            "adj_off_epa_per_rush": adj_off_rush.get(team, league_epa_rush),
            "raw_def_epa_per_play_allowed": _weighted_metric_average(
                defense_rows, "def_epa_per_play_allowed", "def_plays"
            ),
            "adj_def_epa_per_play_allowed": adj_def_play.get(
                team, league_epa_play
            ),
            "raw_def_epa_per_pass_allowed": _weighted_metric_average(
                defense_rows, "def_epa_per_pass_allowed", "def_pass_plays"
            ),
            "adj_def_epa_per_pass_allowed": adj_def_pass.get(
                team, league_epa_pass
            ),
            "raw_def_epa_per_rush_allowed": _weighted_metric_average(
                defense_rows, "def_epa_per_rush_allowed", "def_rush_plays"
            ),
            "adj_def_epa_per_rush_allowed": adj_def_rush.get(
                team, league_epa_rush
            ),
        })

    out = pd.DataFrame(rows)

    rank_specs = [
        ("adj_off_epa_per_play", "off_epa_play_rank", False),
        ("adj_off_epa_per_pass", "off_epa_pass_rank", False),
        ("adj_off_epa_per_rush", "off_epa_rush_rank", False),
        ("adj_def_epa_per_play_allowed", "def_epa_play_rank", True),
        ("adj_def_epa_per_pass_allowed", "def_epa_pass_rank", True),
        ("adj_def_epa_per_rush_allowed", "def_epa_rush_rank", True),
    ]
    for value_col, rank_col, ascending in rank_specs:
        out[rank_col] = (
            pd.to_numeric(out[value_col], errors="coerce")
            .rank(method="min", ascending=ascending)
            .astype("Int64")
        )

    return out.sort_values(
        ["adj_off_epa_per_play", "team"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_historical_dataset(
    schedules: pd.DataFrame,
    pregame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    completed = schedules[
        schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].copy()

    home_state = pregame[
        ["game_id", "team", "pre_games_available"] + MODEL_STATE_FEATURES
    ].copy()
    home_state = home_state.rename(
        columns={
            "team": "home_team_key",
            "pre_games_available": "home_pre_games",
            **{c: f"home_{c}" for c in MODEL_STATE_FEATURES},
        }
    )

    away_state = pregame[
        ["game_id", "team", "pre_games_available"] + MODEL_STATE_FEATURES
    ].copy()
    away_state = away_state.rename(
        columns={
            "team": "away_team_key",
            "pre_games_available": "away_pre_games",
            **{c: f"away_{c}" for c in MODEL_STATE_FEATURES},
        }
    )

    d = completed.merge(
        home_state,
        left_on=["game_id", "home_team"],
        right_on=["game_id", "home_team_key"],
        how="left",
    ).merge(
        away_state,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "away_team_key"],
        how="left",
    )

    d["neutral"] = d["location"].astype(str).str.lower().eq("neutral").astype(int)
    d["rest_diff"] = d["home_rest"] - d["away_rest"]
    d["target_margin"] = d["home_score"] - d["away_score"]
    d["target_total"] = d["home_score"] + d["away_score"]

    # Require at least three previous games for both teams.
    d = d[
        (d["home_pre_games"] >= MIN_ROLLING_GAMES)
        & (d["away_pre_games"] >= MIN_ROLLING_GAMES)
    ].copy()

    feature_cols = (
        [f"home_{c}" for c in MODEL_STATE_FEATURES]
        + [f"away_{c}" for c in MODEL_STATE_FEATURES]
        + ["week", "rest_diff", "neutral"]
    )

    return d, feature_cols


class BlendRegressor:
    """Stable blend of linear Ridge and nonlinear HistGradientBoosting."""

    def __init__(self):
        self.ridge = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
        ])
        self.hgb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=6.0,
                random_state=RANDOM_STATE,
            )),
        ])

    def fit(self, X, y):
        self.ridge.fit(X, y)
        self.hgb.fit(X, y)
        return self

    def predict(self, X):
        return 0.45 * self.ridge.predict(X) + 0.55 * self.hgb.predict(X)


def ats_backtest(
    actual: pd.Series,
    prediction: np.ndarray,
    market: pd.Series,
    thresholds=(1.5, 2.5, 3.5, 4.5),
) -> dict:
    a = pd.to_numeric(actual, errors="coerce").to_numpy()
    m = pd.to_numeric(market, errors="coerce").to_numpy()
    p = np.asarray(prediction)

    out = {}
    valid = np.isfinite(a) & np.isfinite(m) & np.isfinite(p)

    for threshold in thresholds:
        edge = p - m
        bet = valid & (np.abs(edge) >= threshold)
        cover_delta = a - m

        decisions = np.sign(edge[bet])
        outcomes = np.sign(cover_delta[bet])

        pushes = outcomes == 0
        decisions = decisions[~pushes]
        outcomes = outcomes[~pushes]

        wins = int(np.sum(decisions == outcomes))
        losses = int(np.sum(decisions != outcomes))
        bets = wins + losses
        win_rate = wins / bets if bets else np.nan

        # One unit risked per bet at -110; win returns +0.9091.
        roi = ((wins * (100 / 110)) - losses) / bets if bets else np.nan

        out[str(threshold)] = {
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "win_rate": None if pd.isna(win_rate) else round(float(win_rate), 4),
            "roi_at_minus_110": None if pd.isna(roi) else round(float(roi), 4),
        }

    return out


def totals_backtest(
    actual: pd.Series,
    prediction: np.ndarray,
    market: pd.Series,
    thresholds=(1.5, 2.5, 3.5, 4.5),
) -> dict:
    return ats_backtest(actual, prediction, market, thresholds)


def _aggregate_walk_forward_thresholds(
    folds: list[dict],
    thresholds=(1.5, 2.5, 3.5, 4.5),
) -> tuple[dict, float | None]:
    """
    Aggregate sequential out-of-sample betting results and only validate a
    threshold when it survives multiple seasons with enough volume.
    """
    summary = {}

    for threshold in thresholds:
        key = str(threshold)
        season_rows = []
        wins = losses = 0

        for fold in folds:
            row = fold.get("threshold_backtest", {}).get(key)
            if not row or row.get("bets", 0) == 0:
                continue
            season_rows.append({
                "season": fold["season"],
                **row,
            })
            wins += int(row["wins"])
            losses += int(row["losses"])

        bets = wins + losses
        win_rate = wins / bets if bets else np.nan
        roi = ((wins * (100 / 110)) - losses) / bets if bets else np.nan
        profitable_seasons = sum(
            1 for r in season_rows
            if r.get("roi_at_minus_110") is not None
            and r["roi_at_minus_110"] > 0
        )
        seasons_with_bets = len(season_rows)

        # Deliberately strict. A threshold is not called validated because of
        # one hot season or a tiny sample.
        stable = (
            seasons_with_bets >= 3
            and bets >= 100
            and profitable_seasons >= math.ceil(seasons_with_bets * 0.60)
            and pd.notna(roi)
            and roi > 0
            and win_rate > (110 / 210)  # break-even at -110
        )

        summary[key] = {
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "win_rate": None if pd.isna(win_rate) else round(float(win_rate), 4),
            "roi_at_minus_110": None if pd.isna(roi) else round(float(roi), 4),
            "seasons_with_bets": seasons_with_bets,
            "profitable_seasons": profitable_seasons,
            "stable": bool(stable),
            "by_season": season_rows,
        }

    stable_thresholds = [
        float(k) for k, v in summary.items() if v["stable"]
    ]

    if not stable_thresholds:
        return summary, None

    # Among stable thresholds, favor the strongest ROI adjusted for sample size.
    best = max(
        stable_thresholds,
        key=lambda t: (
            (summary[str(t)]["roi_at_minus_110"] or -999)
            * math.sqrt(max(summary[str(t)]["bets"], 1))
        ),
    )
    return summary, float(best)


def walk_forward_market_validation(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    current_season: int,
) -> dict:
    """
    Rolling-origin validation: train only on seasons before each test season.

    This is much harder to fool than one holdout and is the gatekeeper for
    promoting a current market discrepancy from WATCH to VALIDATED.
    """
    market_features = feature_cols + ["spread_line", "total_line"]
    available = {
        int(s) for s in dataset["season"].dropna().unique()
        if int(s) < current_season
    }

    output = {}

    for kind in ["spread", "total"]:
        line_col = "spread_line" if kind == "spread" else "total_line"
        target_col = "target_margin" if kind == "spread" else "target_total"
        residual_col = f"{kind}_residual"

        folds = []
        all_rows = []

        for test_season in MARKET_WALK_FORWARD_SEASONS:
            if test_season not in available:
                continue

            train = dataset[
                (dataset["season"] < test_season)
                & dataset[line_col].notna()
            ].copy()
            test = dataset[
                (dataset["season"] == test_season)
                & dataset[line_col].notna()
            ].copy()

            if len(train) < 500 or len(test) < 100:
                continue

            train[residual_col] = train[target_col] - train[line_col]
            model = BlendRegressor().fit(
                train[market_features],
                train[residual_col],
            )

            residual_pred = model.predict(test[market_features])
            adjusted = test[line_col].to_numpy(dtype=float) + residual_pred

            fold_backtest = ats_backtest(
                test[target_col],
                adjusted,
                test[line_col],
            )

            folds.append({
                "season": int(test_season),
                "games": int(len(test)),
                "market_mae": round(float(mean_absolute_error(
                    test[target_col], test[line_col]
                )), 3),
                "adjusted_mae": round(float(mean_absolute_error(
                    test[target_col], adjusted
                )), 3),
                "threshold_backtest": fold_backtest,
            })

            fold_rows = pd.DataFrame({
                "season": test_season,
                "actual": test[target_col].to_numpy(dtype=float),
                "market": test[line_col].to_numpy(dtype=float),
                "adjusted": adjusted,
            })
            all_rows.append(fold_rows)

        if not all_rows:
            output[kind] = {
                "status": "not_enough_walk_forward_data",
                "validated_threshold": None,
            }
            continue

        oof = pd.concat(all_rows, ignore_index=True)
        threshold_summary, validated_threshold = (
            _aggregate_walk_forward_thresholds(folds)
        )

        output[kind] = {
            "status": "ok",
            "seasons": folds,
            "oof_games": int(len(oof)),
            "market_mae": round(float(mean_absolute_error(
                oof["actual"], oof["market"]
            )), 3),
            "adjusted_mae": round(float(mean_absolute_error(
                oof["actual"], oof["adjusted"]
            )), 3),
            "thresholds": threshold_summary,
            "validated_threshold": validated_threshold,
        }

    return output


def evaluate_holdout(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    current_season: int,
) -> dict:
    """
    Evaluate two layers on the newest completed historical season:

    1) independent football model (never sees market lines);
    2) market-residual model, trained to predict where actual outcomes
       deviate from the spread/total.

    The residual layer is the betting layer. It answers a more honest
    question than "what score do we predict?": where has the market
    historically been wrong conditional on these football features?
    """
    available = sorted(
        int(x) for x in dataset["season"].dropna().unique()
        if int(x) < current_season
    )
    if len(available) < 2:
        return {"status": "not_enough_historical_seasons"}

    holdout = available[-1]
    train = dataset[dataset["season"] < holdout].copy()
    test = dataset[dataset["season"] == holdout].copy()

    if len(train) < 300 or len(test) < 100:
        return {
            "status": "not_enough_games",
            "holdout_season": holdout,
            "train_games": len(train),
            "test_games": len(test),
        }

    # Layer 1: independent football projections.
    margin_model = BlendRegressor().fit(
        train[feature_cols],
        train["target_margin"],
    )
    total_model = BlendRegressor().fit(
        train[feature_cols],
        train["target_total"],
    )

    independent_margin = margin_model.predict(test[feature_cols])
    independent_total = total_model.predict(test[feature_cols])

    winner_accuracy = np.mean(
        np.sign(independent_margin) == np.sign(test["target_margin"].to_numpy())
    )

    report = {
        "status": "ok",
        "holdout_season": holdout,
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "independent_model": {
            "margin_mae": round(float(mean_absolute_error(test["target_margin"], independent_margin)), 3),
            "total_mae": round(float(mean_absolute_error(test["target_total"], independent_total)), 3),
            "winner_accuracy": round(float(winner_accuracy), 4),
        },
    }

    # Layer 2: market residual models.
    market_features = feature_cols + ["spread_line", "total_line"]

    spread_train = train[train["spread_line"].notna()].copy()
    spread_test = test[test["spread_line"].notna()].copy()

    if len(spread_train) >= 300 and len(spread_test) >= 50:
        spread_train["spread_residual"] = (
            spread_train["target_margin"] - spread_train["spread_line"]
        )
        spread_resid_model = BlendRegressor().fit(
            spread_train[market_features],
            spread_train["spread_residual"],
        )
        spread_resid_pred = spread_resid_model.predict(spread_test[market_features])
        adjusted_margin = spread_test["spread_line"].to_numpy() + spread_resid_pred

        report["spread_market_model"] = {
            "games": int(len(spread_test)),
            "market_margin_mae": round(float(mean_absolute_error(
                spread_test["target_margin"],
                spread_test["spread_line"],
            )), 3),
            "adjusted_margin_mae": round(float(mean_absolute_error(
                spread_test["target_margin"],
                adjusted_margin,
            )), 3),
            "threshold_backtest": ats_backtest(
                spread_test["target_margin"],
                adjusted_margin,
                spread_test["spread_line"],
            ),
        }

    total_train = train[train["total_line"].notna()].copy()
    total_test = test[test["total_line"].notna()].copy()

    if len(total_train) >= 300 and len(total_test) >= 50:
        total_train["total_residual"] = (
            total_train["target_total"] - total_train["total_line"]
        )
        total_resid_model = BlendRegressor().fit(
            total_train[market_features],
            total_train["total_residual"],
        )
        total_resid_pred = total_resid_model.predict(total_test[market_features])
        adjusted_total = total_test["total_line"].to_numpy() + total_resid_pred

        report["total_market_model"] = {
            "games": int(len(total_test)),
            "market_total_mae": round(float(mean_absolute_error(
                total_test["target_total"],
                total_test["total_line"],
            )), 3),
            "adjusted_total_mae": round(float(mean_absolute_error(
                total_test["target_total"],
                adjusted_total,
            )), 3),
            "threshold_backtest": totals_backtest(
                total_test["target_total"],
                adjusted_total,
                total_test["total_line"],
            ),
        }

    return report


def fit_market_residual_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
):
    market_features = feature_cols + ["spread_line", "total_line"]

    spread_rows = dataset[dataset["spread_line"].notna()].copy()
    spread_rows["spread_residual"] = (
        spread_rows["target_margin"] - spread_rows["spread_line"]
    )
    spread_model = BlendRegressor().fit(
        spread_rows[market_features],
        spread_rows["spread_residual"],
    )

    total_rows = dataset[dataset["total_line"].notna()].copy()
    total_rows["total_residual"] = (
        total_rows["target_total"] - total_rows["total_line"]
    )
    total_model = BlendRegressor().fit(
        total_rows[market_features],
        total_rows["total_residual"],
    )

    return spread_model, total_model, market_features


def infer_target_week(
    schedules: pd.DataFrame,
    season: int,
    explicit_week: int | None,
) -> int:
    if explicit_week is not None:
        return explicit_week

    s = schedules[schedules["season"].eq(season)].copy()
    today = pd.Timestamp(date.today())

    future = s[s["gameday"].notna() & (s["gameday"] >= today)]
    if not future.empty:
        return int(future.sort_values("gameday").iloc[0]["week"])

    return int(s["week"].max())


def state_for_side(state: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = state.copy()
    out = out.rename(columns={
        "team": f"{prefix}_team_key",
        "games_in_window": f"{prefix}_games_in_window",
        **{
            c: f"{prefix}_{c}"
            for c in state.columns
            if c not in {"team", "games_in_window"}
        },
    })
    return out


def build_upcoming_features(
    schedules: pd.DataFrame,
    state: pd.DataFrame,
    season: int,
    week: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    games = schedules[
        schedules["season"].eq(season)
        & schedules["week"].eq(week)
    ].copy()

    if games.empty:
        raise RuntimeError(f"No games found for season={season}, week={week}")

    home = state_for_side(state, "home")
    away = state_for_side(state, "away")

    g = games.merge(
        home,
        left_on="home_team",
        right_on="home_team_key",
        how="left",
    ).merge(
        away,
        left_on="away_team",
        right_on="away_team_key",
        how="left",
    )

    g["neutral"] = g["location"].astype(str).str.lower().eq("neutral").astype(int)
    g["rest_diff"] = g["home_rest"] - g["away_rest"]

    for c in feature_cols:
        if c not in g.columns:
            g[c] = np.nan

    return g


def matchup_metrics(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    # Positive = favorable to the offense.
    g["home_pass_matchup"] = (
        g.get("home_z_pre_off_pass_epa_db", np.nan)
        + g.get("away_z_pre_def_pass_epa_db_allowed", np.nan)
    )
    g["away_pass_matchup"] = (
        g.get("away_z_pre_off_pass_epa_db", np.nan)
        + g.get("home_z_pre_def_pass_epa_db_allowed", np.nan)
    )
    g["home_run_matchup"] = (
        g.get("home_z_pre_off_rush_epa_att", np.nan)
        + g.get("away_z_pre_def_rush_epa_att_allowed", np.nan)
    )
    g["away_run_matchup"] = (
        g.get("away_z_pre_off_rush_epa_att", np.nan)
        + g.get("home_z_pre_def_rush_epa_att_allowed", np.nan)
    )

    # Rough volume projection: blend offense's recent tendency with
    # the opponent defense's recent plays faced.
    g["home_proj_plays"] = g[
        ["home_pre_off_plays", "away_pre_def_plays_faced"]
    ].mean(axis=1)
    g["away_proj_plays"] = g[
        ["away_pre_off_plays", "home_pre_def_plays_faced"]
    ].mean(axis=1)

    g["home_proj_pass_rate"] = g[
        ["home_pre_off_pass_rate", "away_pre_def_pass_rate_faced"]
    ].mean(axis=1)
    g["away_proj_pass_rate"] = g[
        ["away_pre_off_pass_rate", "home_pre_def_pass_rate_faced"]
    ].mean(axis=1)

    g["home_proj_dropbacks"] = g["home_proj_plays"] * g["home_proj_pass_rate"]
    g["away_proj_dropbacks"] = g["away_proj_plays"] * g["away_proj_pass_rate"]
    g["home_proj_rushes"] = g["home_proj_plays"] * (1 - g["home_proj_pass_rate"])
    g["away_proj_rushes"] = g["away_proj_plays"] * (1 - g["away_proj_pass_rate"])

    return g


def build_candidates(
    predictions: pd.DataFrame,
    spread_edge_threshold: float,
    total_edge_threshold: float,
    validated_spread_threshold: float | None = None,
    validated_total_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Production candidate board.

    Betting qualification comes from the nested market-strategy engine.
    Point-space disagreement is retained only as descriptive context and does
    not override a NO BET decision.
    """
    rows = []

    for _, r in predictions.iterrows():
        reasons = []
        statuses = []

        spread_status = str(r.get("spread_status", "NO BET")).upper()
        if spread_status in {"WATCH", "CAUTION", "VALIDATED"}:
            statuses.append(spread_status)
            reasons.append(
                f"{spread_status} SPREAD: {r.get('spread_pick', '')} | "
                f"model p={float(r.get('spread_probability', np.nan)):.1%} | "
                f"prob edge={float(r.get('spread_probability_edge', np.nan)):.1%} | "
                f"EV={float(r.get('spread_expected_value', np.nan)):.1%}"
            )

        total_status = str(r.get("total_status", "NO BET")).upper()
        if total_status in {"WATCH", "CAUTION", "VALIDATED"}:
            statuses.append(total_status)
            reasons.append(
                f"{total_status} TOTAL: {r.get('total_pick', '')} | "
                f"model p={float(r.get('total_probability', np.nan)):.1%} | "
                f"prob edge={float(r.get('total_probability_edge', np.nan)):.1%} | "
                f"EV={float(r.get('total_expected_value', np.nan)):.1%}"
            )

        if not reasons:
            continue

        promotion_status = (
            "VALIDATED" if "VALIDATED" in statuses
            else ("CAUTION" if "CAUTION" in statuses else "WATCH")
        )

        rows.append({
            "season": r["season"],
            "week": r["week"],
            "game_id": r["game_id"],
            "away_team": r["away_team"],
            "home_team": r["home_team"],
            "model_home_margin": r["model_home_margin"],
            "market_home_margin": r.get("spread_line", np.nan),
            "spread_edge": r.get("spread_edge", np.nan),
            "spread_candidate_side": r.get("spread_candidate_side", ""),
            "spread_pick": r.get("spread_pick", ""),
            "spread_probability": r.get("spread_probability", np.nan),
            "spread_probability_edge": r.get(
                "spread_probability_edge", np.nan
            ),
            "spread_expected_value": r.get(
                "spread_expected_value", np.nan
            ),
            "spread_status": spread_status,
            "model_total": r["model_total"],
            "market_total": r.get("total_line", np.nan),
            "total_edge": r.get("total_edge", np.nan),
            "total_pick": r.get("total_pick", ""),
            "total_probability": r.get("total_probability", np.nan),
            "total_probability_edge": r.get(
                "total_probability_edge", np.nan
            ),
            "total_expected_value": r.get(
                "total_expected_value", np.nan
            ),
            "total_status": total_status,
            "promotion_status": promotion_status,
            "reason": " | ".join(reasons),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "season", "week", "game_id", "away_team", "home_team",
            "model_home_margin", "market_home_margin", "spread_edge",
            "spread_candidate_side", "spread_pick", "spread_probability", "spread_probability_edge",
            "spread_expected_value", "spread_status",
            "model_total", "market_total", "total_edge", "total_pick",
            "total_probability", "total_probability_edge",
            "total_expected_value", "total_status",
            "promotion_status", "reason",
        ])

    out = pd.DataFrame(rows)
    out["max_probability_edge"] = out[
        ["spread_probability_edge", "total_probability_edge"]
    ].max(axis=1)
    return out.sort_values(
        ["promotion_status", "max_probability_edge"],
        ascending=[True, False],
        na_position="last",
    )

def serialize_report(report: dict, output_dir: Path) -> None:
    path = output_dir / "model_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--start-season", type=int, default=START_SEASON)
    parser.add_argument("--spread-edge", type=float, default=3.0)
    parser.add_argument("--total-edge", type=float, default=3.0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cache-dir", default=".nfl_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_cache(Path(args.cache_dir))

    # NFL season naming generally changes in late summer.
    today = date.today()
    auto_season = today.year if today.month >= 7 else today.year - 1
    season = args.season or auto_season

    seasons = list(range(args.start_season, season + 1))
    stats, schedules = load_data(seasons)
    schedules = normalize_schedule(schedules)

    # Only keep completed team-stat rows for feature state.
    stats = stats[stats["game_id"].notna()].copy()
    team_games = build_team_game_stats(stats, schedules)

    # Guard against accidental partial/future rows.
    team_games = team_games[
        team_games["points_for"].notna()
        & team_games["points_against"].notna()
    ].copy()

    pregame = add_pregame_rolling(team_games)
    current_state = latest_team_state(team_games)
    adjusted_epa = build_adjusted_epa_table(team_games, season)

    dataset, feature_cols = build_historical_dataset(schedules, pregame)

    # Keep the independent football holdout as a sanity check, but make the
    # betting decision layer explicitly market-first. Strategy selection is
    # nested walk-forward: signs, sides and thresholds are chosen only from
    # prior out-of-sample seasons before each untouched test season.
    report = evaluate_holdout(dataset, feature_cols, current_season=season)
    market_validation = walk_forward_strategy_validation(
        dataset,
        feature_cols,
        current_season=season,
    )
    report["market_strategy_validation"] = market_validation
    # Preserve this key for the website while changing the underlying method
    # from residual-MAE thresholds to probability/price strategy validation.
    report["walk_forward_market_validation"] = market_validation

    # Explicit no-leakage audit of the full 2025 ATS season. The classifier
    # for 2025 is trained only on seasons before 2025, while the betting policy
    # is frozen from discovery seasons through 2021.
    audit_2025, audit_2025_strategy = spread_bet_audit_for_season(
        dataset,
        feature_cols,
        target_season=2025,
        discovery_end_season=2021,
    )
    audit_2025.to_csv(
        output_dir / "ats_2025_no_leakage_bets.csv",
        index=False,
    )
    report["ats_2025_no_leakage_audit"] = {
        "strategy": audit_2025_strategy,
        "bets": int(len(audit_2025)),
        "wins": int((audit_2025.get("result") == "WIN").sum()) if len(audit_2025) else 0,
        "losses": int((audit_2025.get("result") == "LOSS").sum()) if len(audit_2025) else 0,
        "pushes": int((audit_2025.get("result") == "PUSH").sum()) if len(audit_2025) else 0,
        "profit_units": round(float(audit_2025.get("unit_profit", pd.Series(dtype=float)).sum()), 3),
        "method": "season-2025 classifier fit on seasons <2025; strategy frozen through 2021; outcomes graded only after recommendations",
    }

    total_audit_2025, total_audit_2025_strategy = total_bet_audit_for_season(
        dataset,
        feature_cols,
        target_season=2025,
        discovery_end_season=2021,
    )
    total_audit_2025.to_csv(
        output_dir / "totals_2025_no_leakage_bets.csv",
        index=False,
    )
    total_profit_2025 = float(
        total_audit_2025.get("unit_profit", pd.Series(dtype=float)).sum()
    ) if len(total_audit_2025) else 0.0
    report["totals_2025_no_leakage_audit"] = {
        "strategy": total_audit_2025_strategy,
        "bets": int(len(total_audit_2025)),
        "wins": int((total_audit_2025.get("result") == "WIN").sum()) if len(total_audit_2025) else 0,
        "losses": int((total_audit_2025.get("result") == "LOSS").sum()) if len(total_audit_2025) else 0,
        "pushes": int((total_audit_2025.get("result") == "PUSH").sum()) if len(total_audit_2025) else 0,
        "profit_units": round(total_profit_2025, 3),
        "roi": round(total_profit_2025 / len(total_audit_2025), 4) if len(total_audit_2025) else None,
        "method": "season-2025 total classifier fit on seasons <2025; strategy frozen through 2021; outcomes graded only after recommendations",
    }

    validated_spread_threshold = (
        market_validation.get("spread", {}).get("validated_threshold")
    )
    validated_total_threshold = (
        market_validation.get("total", {}).get("validated_threshold")
    )

    # Final independent football models use all completed games.
    margin_model = BlendRegressor().fit(
        dataset[feature_cols],
        dataset["target_margin"],
    )
    total_model = BlendRegressor().fit(
        dataset[feature_cols],
        dataset["target_total"],
    )

    # Betting layer: directly model cover / over probabilities rather than
    # minimizing score error. These are the models used for wagering decisions.
    spread_market_model = fit_final_market_model(
        dataset, feature_cols, "spread"
    )
    total_market_model = fit_final_market_model(
        dataset, feature_cols, "total"
    )

    target_week = infer_target_week(schedules, season, args.week)

    upcoming = build_upcoming_features(
        schedules,
        current_state,
        season,
        target_week,
        feature_cols,
    )
    upcoming = matchup_metrics(upcoming)

    # Independent football projection: useful as a sanity check.
    upcoming["independent_home_margin"] = margin_model.predict(upcoming[feature_cols])
    upcoming["independent_total"] = total_model.predict(upcoming[feature_cols])

    # Direct market probabilities. The selected sign/direction/threshold is
    # learned from prior OOF seasons; if it does not survive the outer
    # walk-forward test the row is WATCH/NO BET rather than being forced.
    upcoming = predict_current_market(
        spread_market_model,
        upcoming,
        feature_cols,
        "spread",
        market_validation.get("spread", {}),
    )
    upcoming = predict_current_market(
        total_market_model,
        upcoming,
        feature_cols,
        "total",
        market_validation.get("total", {}),
    )

    # Keep point-space disagreement for interpretation only. These values no
    # longer determine the bet direction.
    upcoming["spread_edge"] = np.where(
        upcoming["spread_line"].notna(),
        upcoming["independent_home_margin"] - upcoming["spread_line"],
        np.nan,
    )
    upcoming["total_edge"] = np.where(
        upcoming["total_line"].notna(),
        upcoming["independent_total"] - upcoming["total_line"],
        np.nan,
    )
    upcoming["model_home_margin"] = upcoming["independent_home_margin"]
    upcoming["model_total"] = upcoming["independent_total"]

    upcoming["model_favorite"] = np.where(
        upcoming["model_home_margin"] > 0,
        upcoming["home_team"],
        upcoming["away_team"],
    )
    upcoming["model_total_lean"] = upcoming.get(
        "total_candidate_side",
        upcoming["total_pick"],
    )

    prediction_cols = [
        "season", "week", "gameday", "game_id",
        "away_team", "home_team",
        "independent_home_margin", "model_home_margin", "spread_line", "spread_edge",
        "spread_candidate_side", "spread_pick", "spread_probability", "spread_probability_edge",
        "spread_expected_value", "spread_market_price", "spread_status",
        "independent_total", "model_total", "total_line", "total_edge",
        "total_candidate_side", "total_pick", "total_probability", "total_probability_edge",
        "total_expected_value", "total_market_price", "total_status",
        "model_favorite", "model_total_lean",
        "home_games_in_window", "away_games_in_window",
    ]
    prediction_cols = [c for c in prediction_cols if c in upcoming.columns]

    matchup_cols = [
        "season", "week", "gameday", "game_id",
        "away_team", "home_team",
        "home_pass_matchup", "away_pass_matchup",
        "home_run_matchup", "away_run_matchup",
        "home_proj_plays", "away_proj_plays",
        "home_proj_dropbacks", "away_proj_dropbacks",
        "home_proj_rushes", "away_proj_rushes",
        "home_pre_off_pass_epa_db", "away_pre_off_pass_epa_db",
        "home_pre_off_rush_epa_att", "away_pre_off_rush_epa_att",
        "home_pre_def_pass_epa_db_allowed", "away_pre_def_pass_epa_db_allowed",
        "home_pre_def_rush_epa_att_allowed", "away_pre_def_rush_epa_att_allowed",
    ]
    matchup_cols = [c for c in matchup_cols if c in upcoming.columns]

    predictions = upcoming[prediction_cols].copy()
    matchups = upcoming[matchup_cols].copy()
    candidates = build_candidates(
        upcoming,
        spread_edge_threshold=args.spread_edge,
        total_edge_threshold=args.total_edge,
        validated_spread_threshold=validated_spread_threshold,
        validated_total_threshold=validated_total_threshold,
    )

    predictions.to_csv(output_dir / "latest_predictions.csv", index=False)
    matchups.to_csv(output_dir / "latest_matchups.csv", index=False)
    candidates.to_csv(output_dir / "latest_candidates.csv", index=False)
    current_state.to_csv(output_dir / "team_state.csv", index=False)
    adjusted_epa.to_csv(output_dir / "team_stats.csv", index=False)

    report.update({
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "training_start_season": args.start_season,
        "current_season": season,
        "target_week": target_week,
        "rolling_games": ROLLING_GAMES,
        "min_rolling_games": MIN_ROLLING_GAMES,
        "adjusted_epa_ridge_alpha": ADJUSTED_EPA_RIDGE_ALPHA,
        "adjusted_epa_method": "current-season play-weighted additive offense/defense ridge model; defense is EPA allowed and lower is better",
        "historical_training_games_final_fit": int(len(dataset)),
        "spread_watch_threshold": args.spread_edge,
        "total_watch_threshold": args.total_edge,
        "validated_spread_threshold": validated_spread_threshold,
        "validated_total_threshold": validated_total_threshold,
        "candidate_count": int(len(candidates)),
        "notes": [
            "Independent football projections remain a descriptive sanity check only.",
            "Spread bets directly model home-cover probability; total bets directly model over probability.",
            "Bet selection uses quoted price break-even probability, not raw prediction error.",
            "Sign orientation, one-sided/two-sided rules, and probability-edge thresholds are selected only on prior out-of-sample seasons.",
            "The outer walk-forward seasons grade the entire strategy-selection process on untouched future seasons.",
            "If the strategy does not survive the outer validation, production does not label it VALIDATED.",
            "Historical football features remain shifted before each game.",
        ],
    })
    serialize_report(report, output_dir)

    print("\nNFL EDGE COMPLETE")
    print("=" * 72)
    print(f"Season/week: {season} / {target_week}")
    print(f"Training rows: {len(dataset):,}")
    print(f"Upcoming games: {len(upcoming)}")
    print(f"Candidates/watches: {len(candidates)}")

    if report.get("status") == "ok":
        print(f"Holdout season: {report['holdout_season']}")
        independent = report.get("independent_model", {})
        print(f"Independent margin MAE: {independent.get('margin_mae')}")
        print(f"Independent total MAE: {independent.get('total_mae')}")
        if independent.get("winner_accuracy") is not None:
            print(f"Winner accuracy: {independent['winner_accuracy']:.1%}")

        spread_report = report.get("spread_market_model", {})
        if spread_report:
            print(
                "Spread market MAE -> adjusted MAE: "
                f"{spread_report.get('market_margin_mae')} -> "
                f"{spread_report.get('adjusted_margin_mae')}"
            )

        total_report = report.get("total_market_model", {})
        if total_report:
            print(
                "Total market MAE -> adjusted MAE: "
                f"{total_report.get('market_total_mae')} -> "
                f"{total_report.get('adjusted_total_mae')}"
            )

    if not candidates.empty:
        print("\nTop candidates:")
        print(
            candidates[
                ["away_team", "home_team", "spread_edge", "total_edge", "reason"]
            ].head(12).to_string(index=False)
        )


if __name__ == "__main__":
    main()
