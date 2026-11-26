#!/usr/bin/env python3
"""Fantasy SOS from opponent-adjusted defensive EPA and the NFL schedule.

Uses nflreadpy's current season team stats and schedule. High EPA allowed is
an easier matchup for the opposing offense. Pass EPA applies to QB/WR/TE,
rush EPA to RB. Recent EPA is opponent-adjusted with time-decayed weights,
not raw EPA from the last few opponents. Schedule screen, not PPR forecast.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

POSITIONS = {"QB": "pass", "WR": "pass", "TE": "pass", "RB": "rush"}
DEFAULT_HALF_LIFE = 4.0
DEFAULT_RECENT_WEIGHT = .35
RIDGE_ALPHA = 2.5  # Match nfl_edge.py


def last_fully_completed_week(schedule: pd.DataFrame) -> int:
    """Avoid contaminating ratings with a partially played NFL week."""
    games = schedule.copy()
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    finished = games["home_score"].notna() & games["away_score"].notna()
    complete = games.assign(finished=finished).groupby("week")["finished"].all()
    weeks = [int(w) for w, ok in complete.items() if ok and pd.notna(w)]
    if not weeks:
        raise ValueError("No fully completed regular-season weeks in API schedule")
    last = 0
    for week in sorted(weeks):
        if week != last + 1:
            break
        last = week
    if not last:
        raise ValueError("Week 1 is not complete")
    return last



def last_fully_ingested_week(schedule: pd.DataFrame, team_games: pd.DataFrame) -> int:
    """Only use weeks with scores AND two team-game EPA rows for every game."""
    completed = last_fully_completed_week(schedule)
    if not {"game_id", "week", "team"}.issubset(team_games.columns):
        raise ValueError("Team-game stats missing game_id, week or team")
    sides = team_games.groupby(["week", "game_id"])["team"].nunique()
    last = 0
    for week in range(1, completed + 1):
        expected = schedule[pd.to_numeric(schedule["week"], errors="coerce").eq(week)]["game_id"]
        if expected.empty or any(sides.get((week, game_id), 0) != 2 for game_id in expected):
            break
        last = week
    if not last:
        raise ValueError("No complete week of NFL team EPA stats ingested")
    return last

def fit_defense_adjusted(
    games: pd.DataFrame,
    teams: list[str],
    value_col: str,
    weight_col: str,
    through_week: int,
    half_life: float | None = None,
) -> dict[str, float]:
    """Same offense/opponent-defense weighted ridge design as nfl_edge.py."""
    sample = games[["team", "opponent_team", "week", value_col, weight_col]].copy()
    for field in ("week", value_col, weight_col):
        sample[field] = pd.to_numeric(sample[field], errors="coerce")
    sample = sample.dropna(subset=["team", "opponent_team", "week", value_col, weight_col])
    sample = sample[
        (sample["week"] <= through_week)
        & (sample[weight_col] > 0)
        & np.isfinite(sample[value_col])
        & np.isfinite(sample[weight_col])
    ]
    if sample.empty:
        raise ValueError(f"No valid completed-game EPA data for {value_col}")
    # pandas 3 can return a read-only NumPy view; decay must modify a copy.
    weights = sample[weight_col].to_numpy(dtype=float, copy=True)
    if half_life is not None:
        if half_life <= 0:
            raise ValueError("Recent half-life must be positive")
        weeks_ago = through_week - sample["week"].to_numpy(dtype=float)
        weights *= 0.5 ** (weeks_ago / half_life)
    values = sample[value_col].to_numpy(dtype=float)
    league_mean = float(np.average(values, weights=weights))
    indexes = {name: idx for idx, name in enumerate(teams)}
    X = np.zeros((len(sample), 2 * len(teams)))
    for i, (off, opp) in enumerate(zip(sample["team"], sample["opponent_team"])):
        X[i, indexes[str(off)]] = 1.
        X[i, len(teams) + indexes[str(opp)]] = 1.
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    model.fit(X, values - league_mean, sample_weight=weights / weights.mean())
    return {
        team: league_mean + float(model.coef_[len(teams) + idx])
        for team, idx in indexes.items()
    }


def calculate_defenses(
    team_games: pd.DataFrame,
    teams: list[str],
    through_week: int,
    half_life: float = DEFAULT_HALF_LIFE,
    recent_weight: float = DEFAULT_RECENT_WEIGHT,
) -> pd.DataFrame:
    if not 0 <= recent_weight <= 1:
        raise ValueError("recent_weight must be between 0 and 1")
    games = team_games[pd.to_numeric(team_games["week"], errors="coerce") <= through_week].copy()
    games = games[games["points_for"].notna() & games["points_against"].notna()]
    games["pass_plays"] = (
        pd.to_numeric(games["off_plays"], errors="coerce")
        * pd.to_numeric(games["off_pass_rate"], errors="coerce")
    )
    games["rush_plays"] = pd.to_numeric(games["off_plays"], errors="coerce") - games["pass_plays"]
    output = pd.DataFrame({"team": teams})
    for split, value, weight in [
        ("pass", "off_pass_epa_db", "pass_plays"),
        ("rush", "off_rush_epa_att", "rush_plays"),
    ]:
        overall = fit_defense_adjusted(games, teams, value, weight, through_week)
        recent = fit_defense_adjusted(games, teams, value, weight, through_week, half_life)
        output[f"overall_def_{split}_epa_allowed"] = output["team"].map(overall)
        output[f"recent_def_{split}_epa_allowed"] = output["team"].map(recent)
        output[f"blended_def_{split}_epa_allowed"] = (
            (1 - recent_weight) * output[f"overall_def_{split}_epa_allowed"]
            + recent_weight * output[f"recent_def_{split}_epa_allowed"]
        )
        output[f"{split}_defense_ease_rank"] = (
            output[f"blended_def_{split}_epa_allowed"]
            .rank(ascending=False, method="min").astype(int)
        )
    return output


def build_schedules(
    schedules: pd.DataFrame,
    defense: pd.DataFrame,
    through_week: int,
    start_week: int = 11,
    playoff_start: int = 14,
    end_week: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (1 <= start_week <= playoff_start <= end_week <= 18):
        raise ValueError("Invalid schedule windows")
    if start_week <= through_week:
        raise ValueError("Cannot forecast completed weeks; raise --start-week")
    sched = schedules.copy()
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched = sched[sched["week"].between(start_week, end_week)]
    home = sched[["week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = sched[["week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    home["site"] = "vs"
    away["site"] = "@"
    games = pd.concat([home, away], ignore_index=True)
    if games.duplicated(["week", "team"]).any():
        raise ValueError("Duplicate team/week in nflreadpy schedule")
    missing = (set(games["team"]) | set(games["opponent"])) - set(defense["team"])
    if missing:
        raise ValueError(f"Missing defensive ratings: {sorted(missing)}")
    weekly = pd.concat(
        [games.assign(position=pos, split=split) for pos, split in POSITIONS.items()],
        ignore_index=True,
    )
    weekly = weekly.merge(
        defense, left_on="opponent", right_on="team",
        suffixes=("", "_defense"), validate="many_to_one",
    )
    for field in ("overall", "recent", "blended"):
        weekly[f"opponent_{field}_adj_def_epa_allowed"] = np.where(
            weekly["split"].eq("pass"),
            weekly[f"{field}_def_pass_epa_allowed"],
            weekly[f"{field}_def_rush_epa_allowed"],
        )
    weekly["opponent_defense_ease_rank"] = np.where(
        weekly["split"].eq("pass"),
        weekly["pass_defense_ease_rank"],
        weekly["rush_defense_ease_rank"],
    ).astype(int)
    weekly = weekly[
        ["week", "team", "position", "site", "opponent",
         "opponent_defense_ease_rank",
         "opponent_overall_adj_def_epa_allowed",
         "opponent_recent_adj_def_epa_allowed",
         "opponent_blended_adj_def_epa_allowed"]
    ].sort_values(["position", "team", "week"]).reset_index(drop=True)

    rows = []
    windows = {
        "pre_playoffs": (start_week, playoff_start - 1),
        "rest_of_season": (start_week, end_week),
        "playoffs": (playoff_start, end_week),
    }
    for pos in POSITIONS:
        for team in sorted(defense["team"]):
            played = weekly[(weekly["team"].eq(team)) & weekly["position"].eq(pos)]
            for window, (first, last) in windows.items():
                subset = played[played["week"].between(first, last)]
                if subset.empty:
                    continue
                weeks_played = set(subset["week"].astype(int))
                byes = [w for w in range(first, last + 1) if w not in weeks_played]
                rows.append({
                    "team": team, "position": pos, "window": window,
                    "start_week": first, "end_week": last,
                    "games": len(subset),
                    "bye_weeks": ",".join(str(w) for w in byes),
                    "avg_overall_adj_def_epa_allowed": subset[
                        "opponent_overall_adj_def_epa_allowed"].mean(),
                    "avg_recent_adj_def_epa_allowed": subset[
                        "opponent_recent_adj_def_epa_allowed"].mean(),
                    "avg_blended_adj_def_epa_allowed": subset[
                        "opponent_blended_adj_def_epa_allowed"].mean(),
                    "matchups": " | ".join(
                        f"W{int(r.week)} {r.site} {r.opponent} (#{int(r.opponent_defense_ease_rank)})"
                        for r in subset.itertuples(index=False)
                    ),
                })
    rankings = pd.DataFrame(rows)
    if rankings.empty:
        raise ValueError("No future regular-season opponents")
    rankings["ease_rank_matchups_only"] = (
        rankings.groupby(["window", "position"])["avg_blended_adj_def_epa_allowed"]
        .rank(method="min", ascending=False).astype(int)
    )
    # Do not let a 3-game playoff schedule rank above 4 playable games.
    rankings = rankings.sort_values(
        ["window", "position", "games", "avg_blended_adj_def_epa_allowed", "team"],
        ascending=[True, True, False, False, True],
    )
    rankings["rank_including_byes"] = rankings.groupby(
        ["window", "position"]).cumcount() + 1
    return weekly, rankings.reset_index(drop=True)


def load_live_data(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl
    from schedule_loader import load_nflverse_schedules
    from nfl_edge import build_team_game_stats, normalize_schedule
    stats = nfl.load_team_stats(seasons=[season]).to_pandas()
    schedules = load_nflverse_schedules([season]).to_pandas()
    if "season_type" in stats:
        stats = stats[stats["season_type"].eq("REG")].copy()
    if "game_type" in schedules:
        schedules = schedules[schedules["game_type"].eq("REG")].copy()
    schedules = normalize_schedule(schedules)
    stats = stats[stats["game_id"].notna()].copy()
    return build_team_game_stats(stats, schedules), schedules


def main() -> None:
    parser = argparse.ArgumentParser(description="Opponent-adjusted EPA NFL fantasy SOS")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-week", type=int, default=11)
    parser.add_argument("--playoff-start", type=int, default=14)
    parser.add_argument("--end-week", type=int, default=17)
    parser.add_argument("--through-week", type=int, default=None)
    parser.add_argument("--recent-half-life", type=float, default=DEFAULT_HALF_LIFE)
    parser.add_argument("--recent-weight", type=float, default=DEFAULT_RECENT_WEIGHT)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    games, schedule = load_live_data(args.season)
    completed = last_fully_completed_week(schedule)
    ingested = last_fully_ingested_week(schedule, games)
    asof = args.through_week if args.through_week is not None else ingested
    if asof > ingested:
        raise ValueError(f"Week {asof} is missing complete EPA stats; latest usable is {ingested}")
    start = max(args.start_week, asof + 1)
    if start > args.end_week:
        raise ValueError("No remaining games in specified date range")
    playoff_start = max(start, args.playoff_start)
    teams = sorted(set(schedule["home_team"]).union(schedule["away_team"]))
    defense = calculate_defenses(games, teams, asof, args.recent_half_life, args.recent_weight)
    weekly, team = build_schedules(schedule, defense, asof, start, playoff_start, args.end_week)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    defense.to_csv(out / "fantasy_epa_defense.csv", index=False)
    weekly.to_csv(out / "fantasy_epa_sos_weekly.csv", index=False)
    team.to_csv(out / "fantasy_epa_sos_team.csv", index=False)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": args.season, "last_fully_completed_week": completed,
        "last_fully_ingested_epa_week": ingested,
        "through_week": asof, "schedule_start_week": start,
        "playoff_start_week": playoff_start, "end_week": args.end_week,
        "source": "nflreadpy.load_team_stats and nflreadpy.load_schedules (nflverse)",
        "method": "play-weighted two-way ridge offense/defense EPA, recency-weighted refit",
        "ridge_alpha": RIDGE_ALPHA, "recent_half_life_weeks": args.recent_half_life,
        "overall_weight": 1 - args.recent_weight, "recent_weight": args.recent_weight,
        "interpretation": "Higher defensive EPA allowed is easier for opposing fantasy QB/WR/TE (pass), RB (rush)",
        "caveats": [
            "Team-level opponent screen, not player-specific fantasy projections",
            "Does not model usage, injuries, depth charts, red-zone work, target share, or fantasy-scoring rules",
            "A bye has no matchup; rank_including_byes sorts playable-game count first",
            "No leakage from unfinished games; recent weighted EPA is still opponent adjusted",
        ],
    }
    (out / "fantasy_epa_sos_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for pos in POSITIONS:
        print(f"\n{pos} playoff SOS (Weeks {playoff_start}-{args.end_week}, includes byes)")
        print(team[
            (team["position"].eq(pos)) & team["window"].eq("playoffs")
        ][["rank_including_byes", "team", "games", "bye_weeks",
           "avg_blended_adj_def_epa_allowed", "matchups"]].head(12).to_string(index=False))
    print(f"Saved defensive EPA fantasy SOS snapshot through Week {asof}: {out}")


if __name__ == "__main__":
    main()
