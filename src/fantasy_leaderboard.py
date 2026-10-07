#!/usr/bin/env python3
"""
Opponent-adjusted fantasy points per game leaderboard.

Builds current-season full-PPR leaderboards for QB, RB, WR and TE. The
opponent adjustment is estimated at the team-position-game level with a
shrunk two-way iterative model:

    positional fantasy output
      = league average + team positional offense + opponent positional defense

The offense/defense effects are solved recursively across the season schedule
until convergence. Defensive effects therefore account for the quality of the
position groups a defense has faced instead of using raw fantasy points
allowed. Each player-game is normalized to a league-average opponent, and the
leaderboard reports raw and adjusted fantasy points per game.

Scoring uses nflverse fantasy_points_ppr:
- 1 point per reception
- 1 point per 10 rushing/receiving yards
- 1 point per 25 passing yards
- 6 points per rushing/receiving/special-teams TD
- 4 points per passing TD
- -2 per passing INT or lost fumble
- 2 points per two-point conversion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd


POSITIONS = ("QB", "RB", "WR", "TE")
PRIOR_GAMES = 3.0
MAX_ITERATIONS = 100
TOLERANCE = 1e-8
MIN_FACTOR = 0.80
MAX_FACTOR = 1.25


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_position(series: pd.Series) -> pd.Series:
    position = series.fillna("").astype(str).str.upper()
    return position.replace({"FB": "RB"})


def fantasy_points_ppr(players: pd.DataFrame) -> pd.Series:
    if "fantasy_points_ppr" in players.columns:
        return numeric(players["fantasy_points_ppr"]).fillna(0.0)

    def col(name: str) -> pd.Series:
        if name not in players.columns:
            return pd.Series(0.0, index=players.index)
        return numeric(players[name]).fillna(0.0)

    standard = (
        col("passing_yards") / 25.0
        + 4.0 * col("passing_tds")
        - 2.0 * col("passing_interceptions")
        + (col("rushing_yards") + col("receiving_yards")) / 10.0
        + 6.0 * (col("rushing_tds") + col("receiving_tds") + col("special_teams_tds"))
        + 2.0 * (
            col("passing_2pt_conversions")
            + col("rushing_2pt_conversions")
            + col("receiving_2pt_conversions")
        )
        - 2.0 * (
            col("sack_fumbles_lost")
            + col("rushing_fumbles_lost")
            + col("receiving_fumbles_lost")
        )
    )
    return standard + col("receptions")


def load_current_season(season: int) -> pd.DataFrame:
    print(f"Loading weekly player stats for {season}...")
    players = nfl.load_player_stats(
        seasons=season,
        summary_level="week",
    ).to_pandas()

    if "season_type" in players.columns:
        players = players[players["season_type"].eq("REG")].copy()

    players["season"] = numeric(players["season"]).astype("Int64")
    players["week"] = numeric(players["week"]).astype("Int64")
    players = players[players["season"].eq(season)].copy()

    required = ["player_id", "week", "team", "opponent_team", "position"]
    missing = [column for column in required if column not in players.columns]
    if missing:
        raise RuntimeError(f"Player stats missing required columns: {missing}")

    if "game_id" not in players.columns:
        players["game_id"] = (
            players["season"].astype(str)
            + "_"
            + players["week"].astype(str).str.zfill(2)
            + "_"
            + players["team"].astype(str)
            + "_"
            + players["opponent_team"].astype(str)
        )

    players["position"] = normalize_position(players["position"])
    players = players[players["position"].isin(POSITIONS)].copy()
    players["fantasy_points_ppr"] = fantasy_points_ppr(players)

    if "player_display_name" in players.columns:
        fallback = (
            players["player_name"]
            if "player_name" in players.columns
            else players["player_id"]
        )
        players["player_name"] = players["player_display_name"].fillna(fallback)
    elif "player_name" not in players.columns:
        players["player_name"] = players["player_id"]

    players = players[
        players["team"].notna()
        & players["opponent_team"].notna()
        & players["game_id"].notna()
    ].copy()
    return players


def center_effects(
    effects: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    if not effects:
        return effects
    total_weight = sum(counts.get(key, 0) for key in effects)
    if total_weight <= 0:
        return effects
    mean = sum(
        effects[key] * counts.get(key, 0)
        for key in effects
    ) / total_weight
    return {key: value - mean for key, value in effects.items()}


def fit_position_graph(
    rows: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """
    Solve team-position offense and opponent-position defense effects.

    Positive defense effect means the defense allows more PPR than average
    after controlling for the quality of opposing position groups.
    """
    league_mean = float(rows["position_points"].mean())
    if not np.isfinite(league_mean) or league_mean <= 0:
        return 0.0, {}

    teams = sorted(rows["team"].astype(str).unique())
    defenses = sorted(rows["opponent_team"].astype(str).unique())
    offense = {team: 0.0 for team in teams}
    defense = {team: 0.0 for team in defenses}

    offense_counts = (
        rows.groupby("team")["game_id"].nunique().astype(int).to_dict()
    )
    defense_counts = (
        rows.groupby("opponent_team")["game_id"].nunique().astype(int).to_dict()
    )

    for _ in range(MAX_ITERATIONS):
        old_offense = offense.copy()
        old_defense = defense.copy()

        new_offense: dict[str, float] = {}
        for team, group in rows.groupby("team"):
            residual = (
                group["position_points"].to_numpy(dtype=float)
                - league_mean
                - group["opponent_team"].astype(str).map(defense).fillna(0.0).to_numpy()
            )
            games = max(1, int(group["game_id"].nunique()))
            shrink = games / (games + PRIOR_GAMES)
            new_offense[str(team)] = shrink * float(np.mean(residual))
        offense = center_effects(new_offense, offense_counts)

        new_defense: dict[str, float] = {}
        for opponent, group in rows.groupby("opponent_team"):
            residual = (
                group["position_points"].to_numpy(dtype=float)
                - league_mean
                - group["team"].astype(str).map(offense).fillna(0.0).to_numpy()
            )
            games = max(1, int(group["game_id"].nunique()))
            shrink = games / (games + PRIOR_GAMES)
            new_defense[str(opponent)] = shrink * float(np.mean(residual))
        defense = center_effects(new_defense, defense_counts)

        delta = 0.0
        for key in set(old_offense) | set(offense):
            delta = max(
                delta,
                abs(offense.get(key, 0.0) - old_offense.get(key, 0.0)),
            )
        for key in set(old_defense) | set(defense):
            delta = max(
                delta,
                abs(defense.get(key, 0.0) - old_defense.get(key, 0.0)),
            )

        if delta < TOLERANCE:
            break

    return league_mean, defense


def build_leaderboard(
    players: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame()

    through_week = int(players["week"].max())
    team_position = (
        players.groupby(
            ["game_id", "week", "team", "opponent_team", "position"],
            as_index=False,
        )
        .agg(position_points=("fantasy_points_ppr", "sum"))
    )

    factor_frames = []
    for position in POSITIONS:
        rows = team_position[team_position["position"].eq(position)].copy()
        if rows.empty:
            continue

        league_mean, defense = fit_position_graph(rows)
        if league_mean <= 0:
            continue

        rows["defense_expected_allowed"] = (
            league_mean
            + rows["opponent_team"].astype(str).map(defense).fillna(0.0)
        )
        floor = league_mean * MIN_FACTOR
        ceiling = league_mean / MIN_FACTOR
        rows["defense_expected_allowed"] = rows[
            "defense_expected_allowed"
        ].clip(lower=floor, upper=ceiling)

        rows["opponent_factor"] = (
            league_mean / rows["defense_expected_allowed"]
        ).clip(lower=MIN_FACTOR, upper=MAX_FACTOR)

        factor_frames.append(
            rows[["game_id", "team", "position", "opponent_factor"]]
        )

    if not factor_frames:
        return pd.DataFrame()

    factors = pd.concat(factor_frames, ignore_index=True)
    adjusted = players.merge(
        factors,
        on=["game_id", "team", "position"],
        how="left",
    )
    adjusted["opponent_factor"] = numeric(
        adjusted["opponent_factor"]
    ).fillna(1.0)
    adjusted["adjusted_points"] = (
        adjusted["fantasy_points_ppr"] * adjusted["opponent_factor"]
    )

    adjusted = adjusted.sort_values(["player_id", "week", "game_id"])
    latest_team = (
        adjusted.groupby(["player_id", "position"], as_index=False)
        .tail(1)[["player_id", "position", "team", "player_name"]]
    )

    leaderboard = (
        adjusted.groupby(["player_id", "position"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            raw_fppg=("fantasy_points_ppr", "mean"),
            adjusted_fppg=("adjusted_points", "mean"),
            avg_opponent_factor=("opponent_factor", "mean"),
        )
        .merge(
            latest_team,
            on=["player_id", "position"],
            how="left",
        )
    )

    min_games = 1 if through_week <= 1 else 2
    leaderboard = leaderboard[
        leaderboard["games"].ge(min_games)
        & leaderboard["raw_fppg"].gt(0)
    ].copy()

    leaderboard["opponent_adjustment_pct"] = np.where(
        leaderboard["raw_fppg"].abs() > 1e-12,
        leaderboard["adjusted_fppg"] / leaderboard["raw_fppg"] - 1.0,
        0.0,
    )

    leaderboard["season"] = season
    leaderboard["through_week"] = through_week
    leaderboard = leaderboard.sort_values(
        ["position", "adjusted_fppg", "raw_fppg", "player_name"],
        ascending=[True, False, False, True],
    ).copy()
    leaderboard["rank"] = leaderboard.groupby("position").cumcount() + 1

    columns = [
        "season",
        "through_week",
        "position",
        "rank",
        "player_id",
        "player_name",
        "team",
        "games",
        "raw_fppg",
        "opponent_adjustment_pct",
        "avg_opponent_factor",
        "adjusted_fppg",
    ]
    return leaderboard[columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fantasy_leaderboard.csv"),
    )
    args = parser.parse_args()

    players = load_current_season(args.season)
    leaderboard = build_leaderboard(players, args.season)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.output, index=False)

    through_week = (
        int(leaderboard["through_week"].max())
        if not leaderboard.empty
        else "n/a"
    )
    print(
        f"Wrote {len(leaderboard):,} fantasy leaderboard rows "
        f"through Week {through_week}: {args.output}"
    )


if __name__ == "__main__":
    main()
