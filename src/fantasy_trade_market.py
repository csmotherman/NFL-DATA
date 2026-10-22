#!/usr/bin/env python3
"""Current-week fantasy PPR regression and remaining-schedule research board.

Uses published NFL weekly box-score opportunities. This is a transparent,
non-calibrated trade *screen*, not a validated player prop or injury model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from fantasy_leaderboard import fit_position_graph, load_current_season

POSITIONS = ("QB", "RB", "WR", "TE")
STAT_COLS = (
    "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_epa", "receiving_air_yards",
)


def div(a: float, b: float) -> float:
    return float(a) / float(b) if b and pd.notna(b) else 0.0


def normal_rates(players: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for pos, x in players.groupby("position"):
        a, c, t = x["attempts"].sum(), x["carries"].sum(), x["targets"].sum()
        result[pos] = {
            "pass_yards": div(x["passing_yards"].sum(), a),
            "pass_td": div(x["passing_tds"].sum(), a),
            "pass_int": div(x["passing_interceptions"].sum(), a),
            "rush_yards": div(x["rushing_yards"].sum(), c),
            "rush_td": div(x["rushing_tds"].sum(), c),
            "catch": div(x["receptions"].sum(), t),
            "rec_yards": div(x["receiving_yards"].sum(), t),
            "rec_td": div(x["receiving_tds"].sum(), t),
        }
    return result


def expected_points(rows: pd.DataFrame, rates: dict) -> pd.Series:
    # Does not include rare two-point conversions, return TDs, lost fumbles.
    x = pd.Series(0.0, index=rows.index)
    for pos, r in rates.items():
        mask = rows["position"].eq(pos)
        z = rows.loc[mask]
        x.loc[mask] = (
            z["attempts"] * (r["pass_yards"] / 25 + 4 * r["pass_td"] - 2 * r["pass_int"])
            + z["carries"] * (r["rush_yards"] / 10 + 6 * r["rush_td"])
            + z["targets"] * (r["catch"] + r["rec_yards"] / 10 + 6 * r["rec_td"])
        )
    return x.clip(lower=0)



def player_skill_rates(current: pd.DataFrame, history: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Empirical-Bayes per-opportunity skill with prior seasons and positional priors.

    Historical observations are strictly from before the projection season.
    Yards/catches use a 20-opportunity positional prior; volatile TD outcomes
    use an 80-opportunity prior. Returning veterans retain their demonstrated
    individual skill rather than being regressed to the average RB/WR/TE.
    """
    pos_rates = normal_rates(current)
    hist = history.copy()
    for col in STAT_COLS:
        if col not in hist.columns:
            hist[col] = 0.
        hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0.)
    stats = {}
    for (pid, pos), g in current.groupby(["player_id", "position"]):
        old = hist[(hist["player_id"].eq(pid)) & (hist["position"].eq(pos))]
        rate = pos_rates[pos]
        result = {}
        for prefix, denominator, fields in [
            ("rush", "carries", [("rush_yards", "rushing_yards", 20.), ("rush_td", "rushing_tds", 80.)]),
            ("rec", "targets", [
                ("catch", "receptions", 20.),
                ("rec_yards", "receiving_yards", 20.),
                ("rec_td", "receiving_tds", 80.),
            ]),
        ]:
            for output, numerator, smoothing in fields:
                cur_n = float(g[numerator].sum())
                cur_d = float(g[denominator].sum())
                old_n = float((old[numerator] * old["history_weight"]).sum()) if not old.empty else 0.
                old_d = float((old[denominator] * old["history_weight"]).sum()) if not old.empty else 0.
                # Cap the amount of historical data so distant seasons cannot
                # overwhelm meaningful changes to 2026 efficiency.
                cap = 200. if smoothing == 20. else 250.
                scale = min(1., cap / old_d) if old_d else 1.
                numerator_shrunk = cur_n + old_n * scale + rate[output] * smoothing
                denominator_shrunk = cur_d + old_d * scale + smoothing
                result[output] = numerator_shrunk / denominator_shrunk
        stats[(str(pid), str(pos))] = result
    return stats


def load_skill_history(season: int) -> pd.DataFrame:
    seasons = list(range(max(2018, season - 3), season))
    if not seasons:
        return pd.DataFrame(columns=["player_id", "position", "history_weight"])
    try:
        hist = nfl.load_player_stats(seasons=seasons, summary_level="week").to_pandas()
    except Exception as exc:
        raise RuntimeError("Historical player stats are required for talent priors") from exc
    if "season_type" in hist.columns:
        hist = hist[hist["season_type"].eq("REG")].copy()
    hist["position"] = hist["position"].fillna("").astype(str).str.upper().replace({"FB": "RB"})
    hist = hist[hist["position"].isin(POSITIONS)].copy()
    hist["history_weight"] = .65 ** (season - pd.to_numeric(hist["season"], errors="coerce"))
    return hist


def team_pace_context(players: pd.DataFrame) -> pd.DataFrame:
    """Same-game totals, eliminating artificial workload spikes from low-play games."""
    totals = players.groupby(["game_id", "team"], as_index=False).agg(
        team_carries=("carries", "sum"),
        team_targets=("targets", "sum"),
    )
    return players.merge(totals, on=["game_id", "team"], how="left", validate="many_to_one")


def projected_role(players: pd.DataFrame, player_rows: pd.DataFrame, team: str) -> tuple[float, float]:
    """Project carries and targets from stable team pace and player share.

    A single pass-heavy, trailing game does not unilaterally redefine the
    player's upcoming carry share. Team/play volume remains dynamic.
    """
    tm = players[players["team"].eq(team)].drop_duplicates("game_id").sort_values("week")
    q = player_rows[player_rows["team"].eq(team)].sort_values("week")
    if q.empty or tm.empty:
        return float(player_rows["carries"].mean()), float(player_rows["targets"].mean())
    tm_recent = tm.tail(3)
    proj_tm_carries = .80 * tm["team_carries"].mean() + .20 * tm_recent["team_carries"].median()
    proj_tm_targets = .80 * tm["team_targets"].mean() + .20 * tm_recent["team_targets"].median()
    carry_sh = (q["carries"] / q["team_carries"].replace(0, np.nan)).clip(0, 1)
    target_sh = (q["targets"] / q["team_targets"].replace(0, np.nan)).clip(0, 1)
    carry_sh = .80 * carry_sh.mean() + .20 * carry_sh.tail(3).median()
    target_sh = .80 * target_sh.mean() + .20 * target_sh.tail(3).median()
    return (
        float(max(0, proj_tm_carries * carry_sh)) if pd.notna(carry_sh) else float(q["carries"].mean()),
        float(max(0, proj_tm_targets * target_sh)) if pd.notna(target_sh) else float(q["targets"].mean()),
    )

def defense_offsets(players: pd.DataFrame) -> dict[str, dict[str, float]]:
    team_pos_game = players.groupby(
        ["game_id", "team", "opponent_team", "position"], as_index=False
    ).agg(position_points=("fantasy_points_ppr", "sum"))
    return {
        pos: fit_position_graph(team_pos_game[team_pos_game["position"].eq(pos)])[1]
        for pos in POSITIONS
    }


def future_schedule(season: int, through_week: int, end_week: int) -> dict[str, list[tuple[int, str]]]:
    s = nfl.load_schedules([season]).to_pandas()
    if "game_type" in s:
        s = s[s["game_type"].eq("REG")]
    s = s[
        pd.to_numeric(s["week"], errors="coerce").between(through_week + 1, end_week)
    ]
    teams: dict[str, list[tuple[int, str]]] = {}
    for r in s.itertuples(index=False):
        teams.setdefault(r.home_team, []).append((int(r.week), r.away_team))
        teams.setdefault(r.away_team, []).append((int(r.week), r.home_team))
    return {team: sorted(games) for team, games in teams.items()}


def build(players: pd.DataFrame, schedule: dict, through_week: int, history: pd.DataFrame) -> pd.DataFrame:
    x = players.copy()
    for col in STAT_COLS:
        if col not in x:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)
    x["fantasy_points_ppr"] = pd.to_numeric(x["fantasy_points_ppr"], errors="coerce").fillna(0)
    x["player_name"] = x["player_name"].fillna(x["player_id"])
    x = x.sort_values(["player_id", "week", "game_id"])
    rates = normal_rates(x)
    x["expected_ppr"] = expected_points(x, rates)
    x["opportunities"] = x["attempts"] + x["carries"] + x["targets"]
    target_by_game = x.groupby(["game_id", "team"])["targets"].transform("sum")
    carries_by_game = x.groupby(["game_id", "team"])["carries"].transform("sum")
    x["target_share"] = x["targets"] / target_by_game.replace(0, np.nan)
    x["carry_share"] = x["carries"] / carries_by_game.replace(0, np.nan)
    x = team_pace_context(x)
    skills = player_skill_rates(x, history)
    defense = defense_offsets(x)
    records = []
    for (pid, pos), p in x.groupby(["player_id", "position"], sort=False):
        games = p["game_id"].nunique()
        if games < 3:
            continue
        latest = p.iloc[-1]
        team = str(latest["team"])
        opps = schedule.get(team, [])
        if not opps:
            continue
        recent = p.tail(3)
        actual_pg = p["fantasy_points_ppr"].mean()
        xp_pg = p["expected_ppr"].mean()
        opp_pg = p["opportunities"].mean()
        # Project shares of stable team offensive opportunities, not the
        # most recent game's raw carries. Player skill is learned with
        # historical seasons and appropriate positional sample-size priors.
        pred_carries, pred_targets = projected_role(x, p, team)
        skill = skills.get((str(pid), str(pos)))
        if skill and pos in ("RB", "WR", "TE"):
            baseline = (
                pred_carries * (skill["rush_yards"] / 10. + 6 * skill["rush_td"])
                + pred_targets * (skill["catch"] + skill["rec_yards"] / 10. + 6 * skill["rec_td"])
            )
        else:
            # QB model remains explicitly a SCREEN only, not a reliable
            # fantasy passing projection (no passing talent priors).
            baseline = .65 * xp_pg + .35 * recent["expected_ppr"].mean()
        offsets = defense[pos]
        past = float(p["opponent_team"].astype(str).map(offsets).fillna(0).mean())
        future = float(np.mean([offsets.get(d, 0) for _, d in opps]))
        group_pos = x[(x["team"].eq(team)) & (x["position"].eq(pos))]
        player_total = p.loc[p["team"].eq(team), "fantasy_points_ppr"].sum()
        team_pos_total = group_pos["fantasy_points_ppr"].sum()
        player_share = min(1.0, max(0., div(player_total, team_pos_total)))
        # Already partly reflected in observed production; do not double-count
        # or permit a noisy 6-game defensive estimate to swamp talent/volume.
        matchup_shift = np.clip((future - past) * player_share * .25, -1.5, 1.5)
        ros_pg = max(0., baseline + matchup_shift)
        gap = actual_pg - ros_pg
        td = p["passing_tds"].sum() + p["rushing_tds"].sum() + p["receiving_tds"].sum()
        opportunities = p["opportunities"].sum()
        carries = p["carries"].sum()
        targets = p["targets"].sum()
        min_role = 12 if pos == "QB" else 6
        reliable = games >= 5 and opp_pg >= min_role
        td_rate = div(td, opportunities)
        # A scoring spike is not a sell signal when elite volume supports it.
        # These are conservative candidate flags, not predicted trade returns.
        touchdown_dependent = (
            pos == "WR" and td_rate >= .075
            and p["targets"].mean() < 9 and p["target_share"].mean() < .27
        ) or (
            pos == "RB" and td_rate >= .055
            and opp_pg < 20 and p["targets"].mean() < 5
        ) or (
            pos == "TE" and td_rate >= .08
            and p["targets"].mean() < 8
        )
        # QB passing/rushing skill needs historical player priors; suppress
        # simplistic same-season league-average buy/sell calls for QBs.
        signal = "WATCH"
        if reliable and pos != "QB":
            if gap >= 2.5 and touchdown_dependent:
                signal = "SELL_HIGH"
            elif gap <= -2.5:
                signal = "BUY_LOW"
        records.append({
            "player_id": pid, "player_name": latest["player_name"], "position": pos,
            "team": team, "games": games, "through_week": through_week,
            "ppr_pg": round(actual_pg, 3), "opportunity_expected_ppr_pg": round(xp_pg, 3),
            "opportunities_pg": round(opp_pg, 3), "recent3_opportunities_pg": round(recent["opportunities"].mean(), 3),
            "projected_carries_pg": round(pred_carries, 3), "projected_targets_pg": round(pred_targets, 3),
            "skill_baseline_ppr_pg": round(float(baseline), 3), "sos_adjustment_ppr_pg": round(float(matchup_shift), 3),
            "targets_pg": round(p["targets"].mean(), 3), "target_share": round(p["target_share"].mean(skipna=True), 3),
            "carries_pg": round(p["carries"].mean(), 3), "carry_share": round(p["carry_share"].mean(skipna=True), 3),
            "yards_per_carry": round(div(p["rushing_yards"].sum(), carries), 3),
            "yards_per_target": round(div(p["receiving_yards"].sum(), targets), 3),
            "catch_rate": round(div(p["receptions"].sum(), targets), 3),
            "rush_epa_per_carry": round(div(p["rushing_epa"].sum(), carries), 3),
            "rec_epa_per_target": round(div(p["receiving_epa"].sum(), targets), 3),
            "touchdowns_per_100_opportunities": round(100 * div(td, opportunities), 3),
            "past_opponent_adjusted_defense_effect": round(past, 3),
            "remaining_opponent_adjusted_defense_effect": round(future, 3),
            "remaining_games": len(opps),
            "past_opponents": " | ".join(f"W{int(wk)} {opp}" for wk, opp in zip(p["week"], p["opponent_team"])),
            "ros_opponents": " | ".join(f"W{wk} {opp}" for wk, opp in opps),
            "projected_ros_ppr_pg": round(ros_pg, 3),
            "projected_ros_ppr_total": round(ros_pg * len(opps), 3),
            "ppr_above_projection_pg": round(gap, 3), "trade_screen": signal,
            "confidence_note": "research only; no injury or trade-market valuation",
        })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(["position", "ppr_above_projection_pg"], ascending=[True, False])
    df["ros_position_rank"] = df.groupby("position")["projected_ros_ppr_pg"].rank(ascending=False, method="min").astype(int)
    team_sos = df[["team", "position", "remaining_opponent_adjusted_defense_effect"]].drop_duplicates(["team", "position"]).copy()
    team_sos["ros_sos_rank_easiest"] = team_sos.groupby("position")["remaining_opponent_adjusted_defense_effect"].rank(ascending=False, method="min").astype(int)
    df = df.merge(team_sos[["team", "position", "ros_sos_rank_easiest"]], on=["team", "position"], how="left")
    return df


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--season", type=int, default=2026)
    a.add_argument("--end-week", type=int, default=17)
    a.add_argument("--output-dir", default="outputs")
    args = a.parse_args()
    players = load_current_season(args.season)
    if players.empty:
        raise RuntimeError("No current-season weekly PPR stats available")
    through_week = int(players["week"].max())
    if through_week >= args.end_week:
        raise RuntimeError(f"No ROS games after Week {through_week}")
    schedules = future_schedule(args.season, through_week, args.end_week)
    history = load_skill_history(args.season)
    result = build(players, schedules, through_week, history)
    if result.empty:
        raise RuntimeError("No players with at least three games and a ROS schedule")
    # Exception feed records verified material injuries and role uncertainty.
    # Do not auto-remove old reports: stale health evidence must be rechecked.
    health_file = Path("inputs/fantasy_status_overrides.csv")
    result["health_status"] = ""
    result["health_asof"] = ""
    result["health_context"] = ""
    if health_file.exists():
        health = pd.read_csv(health_file).fillna("")
        for row in health.itertuples(index=False):
            match = result["player_name"].eq(str(row.player_name))
            if not match.any():
                continue
            result.loc[match, "health_status"] = str(row.status)
            result.loc[match, "health_asof"] = str(row.asof)
            result.loc[match, "health_context"] = str(row.context)
            result.loc[match, "trade_screen"] = "INJURY_REVIEW"
            result.loc[match, "confidence_note"] = (
                "Availability or role materially uncertain. Do not trade solely "
                "on this healthy-usage projection; verify current news."
            )
            # Missing weeks are not priced by the unconditional PPR projection.
            result.loc[match, "projected_ros_ppr_total"] = np.nan
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "fantasy_trade_market.csv", index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": args.season, "through_week": through_week,
        "remaining_weeks": [through_week + 1, args.end_week],
        "rows": len(result), "signals": result["trade_screen"].value_counts().to_dict(),
        "model_type": "player-specific historical empirical-Bayes skill + stable team-share volume + shrunk opponent adjustment",
        "limitations": [
            "Not an empirically validated rest-of-season player forecast or market-price model",
            "No snap share, routes run, injury, depth-chart, or quarterback-change adjustment",
            "Recent usage is the last three games played, not necessarily last three team games",
            "Player efficiency uses up to three prior seasons and sample-size positional priors; does not incorporate tracking data",
            "QB trade labels suppressed until passing/rushing talent priors validated",
            "Sell-high labels require a touchdown-dependent role without elite volume",
            "Manual injury overrides are sticky and must be reverified; unadjusted healthy PPR estimates are not valid availability forecasts",
            "No playoff schedule after Week 17, no Week 18",
            "Touchdown regression uses past carry/target rates; goal-line and red-zone participation not explicitly available",
            "Remaining opponent scores shrink team-position output for opponent strength, not individual coverage",
        ],
    }
    (output / "fantasy_trade_market_report.json").write_text(json.dumps(metadata, indent=2))
    print(f"Fantasy trade research: {len(result)} players, Weeks 1-{through_week}; ", metadata["signals"])


if __name__ == "__main__":
    main()
