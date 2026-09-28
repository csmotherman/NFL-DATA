#!/usr/bin/env python3
"""Build an auditable forward NFL betting ledger for the website.

The weekly game model can be rerun at any time, so historical "bets" must not be
reconstructed after results are known. This script only locks a week's board
when none of that week's games has a final score. Once locked, that snapshot is
never overwritten and is graded against future results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl


ODDS_COLUMNS = [
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "over_odds",
    "under_odds",
]

HISTORY_COLUMNS = [
    "season",
    "week",
    "gameday",
    "game_id",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "final_score",
    "locked_at_utc",
    "spread_pick",
    "spread_pick_line",
    "spread_edge",
    "spread_status",
    "spread_result",
    "spread_odds",
    "spread_profit_units",
    "total_pick",
    "total_line",
    "total_edge",
    "total_status",
    "total_result",
    "total_odds",
    "total_profit_units",
    "moneyline_pick",
    "moneyline_price",
    "moneyline_result",
    "moneyline_profit_units",
    "model_home_margin",
    "model_total",
    "model_away_score",
    "model_home_score",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    schedules = nfl.load_schedules(sorted(set(seasons))).to_pandas()
    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"].eq("REG")].copy()

    schedules["gameday"] = pd.to_datetime(
        schedules.get("gameday"), errors="coerce"
    )

    for column in [
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        *ODDS_COLUMNS,
    ]:
        if column not in schedules.columns:
            schedules[column] = np.nan
        schedules[column] = numeric(schedules[column])

    return schedules


def format_team_spread(row: pd.Series, team: str) -> float:
    line = row.get("spread_line", np.nan)
    if pd.isna(line):
        return np.nan
    return -float(line) if team == row.get("home_team") else float(line)


def status_for_edge(
    edge: float,
    watch_threshold: float | None,
    validated_threshold: float | None,
) -> str:
    if pd.isna(edge):
        return "NO LINE"
    magnitude = abs(float(edge))
    if validated_threshold is not None and magnitude >= validated_threshold:
        return "VALIDATED"
    if watch_threshold is not None and magnitude >= watch_threshold:
        return "WATCH"
    return "LEAN"


def american_profit(odds) -> float:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    if value <= -100:
        return 100.0 / abs(value)
    if value >= 100:
        return value / 100.0
    if value > 1:
        return value - 1.0
    return np.nan


def profit_for_result(result: str, odds) -> float:
    if result == "W":
        return american_profit(odds)
    if result == "L":
        return -1.0
    if result == "P":
        return 0.0
    return np.nan


def result_from_delta(delta: float, pick_positive: bool) -> str:
    if pd.isna(delta):
        return ""
    if abs(float(delta)) < 1e-12:
        return "P"
    winner_positive = float(delta) > 0
    return "W" if winner_positive == pick_positive else "L"


def enrich_board(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    report: dict,
) -> pd.DataFrame:
    odds_cols = ["game_id", "home_score", "away_score", *ODDS_COLUMNS]
    available = [column for column in odds_cols if column in schedules.columns]
    board = predictions.merge(
        schedules[available].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    watch_spread = report.get("spread_watch_threshold", 3.0)
    watch_total = report.get("total_watch_threshold", 3.0)
    valid_spread = report.get("validated_spread_threshold")
    valid_total = report.get("validated_total_threshold")

    board["spread_pick"] = np.where(
        numeric(board["spread_edge"]) >= 0,
        board["home_team"],
        board["away_team"],
    )
    board["spread_pick_line"] = board.apply(
        lambda row: format_team_spread(row, row["spread_pick"]),
        axis=1,
    )
    board["spread_status"] = board["spread_edge"].apply(
        lambda edge: status_for_edge(edge, watch_spread, valid_spread)
    )

    board["total_pick"] = np.where(
        numeric(board["total_edge"]) >= 0,
        "OVER",
        "UNDER",
    )
    board["total_status"] = board["total_edge"].apply(
        lambda edge: status_for_edge(edge, watch_total, valid_total)
    )

    board["moneyline_pick"] = board["model_favorite"]
    board["moneyline_price"] = np.where(
        board["moneyline_pick"].eq(board["home_team"]),
        board.get("home_moneyline", np.nan),
        board.get("away_moneyline", np.nan),
    )
    board["spread_odds"] = np.where(
        board["spread_pick"].eq(board["home_team"]),
        board.get("home_spread_odds", np.nan),
        board.get("away_spread_odds", np.nan),
    )
    board["total_odds"] = np.where(
        board["total_pick"].eq("OVER"),
        board.get("over_odds", np.nan),
        board.get("under_odds", np.nan),
    )

    board["model_home_score"] = (
        numeric(board["model_total"]) + numeric(board["model_home_margin"])
    ) / 2.0
    board["model_away_score"] = (
        numeric(board["model_total"]) - numeric(board["model_home_margin"])
    ) / 2.0

    board["max_edge"] = pd.concat(
        [
            numeric(board["spread_edge"]).abs(),
            numeric(board["total_edge"]).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return board


def can_lock_week(board: pd.DataFrame) -> bool:
    if board.empty:
        return False
    completed = (
        board.get("home_score", pd.Series(index=board.index, dtype=float)).notna()
        | board.get("away_score", pd.Series(index=board.index, dtype=float)).notna()
    )
    return not bool(completed.any())


def archive_board(board: pd.DataFrame, archive_dir: Path) -> Path | None:
    if board.empty:
        return None

    season = int(board.iloc[0]["season"])
    week = int(board.iloc[0]["week"])
    path = archive_dir / f"{season}_week_{week:02d}_game_bets.csv"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Game ledger already locked: {path}")
        return path

    if not can_lock_week(board):
        print(
            "Not locking current game board because at least one target-week "
            "game already has a final score. This prevents hindsight leakage."
        )
        return None

    locked = board.copy()
    locked["locked_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    locked.to_csv(path, index=False)
    print(f"Locked forward game ledger: {path}")
    return path


def seasons_from_archives(archive_dir: Path) -> set[int]:
    seasons: set[int] = set()
    if not archive_dir.exists():
        return seasons
    for path in archive_dir.glob("*_week_*_game_bets.csv"):
        try:
            seasons.add(int(path.name.split("_", 1)[0]))
        except (TypeError, ValueError):
            continue
    return seasons


def grade_archives(
    archive_dir: Path,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    if not archive_dir.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    finals = schedules[
        schedules["home_score"].notna() & schedules["away_score"].notna()
    ].copy()
    final_map = finals.set_index("game_id", drop=False)

    for path in sorted(archive_dir.glob("*_week_*_game_bets.csv")):
        locked = pd.read_csv(path)
        for _, source in locked.iterrows():
            game_id = source.get("game_id")
            if game_id not in final_map.index:
                continue

            final = final_map.loc[game_id]
            if isinstance(final, pd.DataFrame):
                final = final.iloc[0]

            home_score = float(final["home_score"])
            away_score = float(final["away_score"])
            actual_margin = home_score - away_score
            actual_total = home_score + away_score

            spread_line = pd.to_numeric(
                pd.Series([source.get("spread_line")]), errors="coerce"
            ).iloc[0]
            total_line = pd.to_numeric(
                pd.Series([source.get("total_line")]), errors="coerce"
            ).iloc[0]

            spread_pick = str(source.get("spread_pick", ""))
            if pd.notna(spread_line) and spread_pick:
                spread_delta = actual_margin - float(spread_line)
                spread_result = result_from_delta(
                    spread_delta,
                    spread_pick == source.get("home_team"),
                )
            else:
                spread_result = ""

            total_pick = str(source.get("total_pick", ""))
            if pd.notna(total_line) and total_pick:
                total_delta = actual_total - float(total_line)
                total_result = result_from_delta(
                    total_delta,
                    total_pick == "OVER",
                )
            else:
                total_result = ""

            if home_score == away_score:
                winner = "TIE"
            elif home_score > away_score:
                winner = str(source.get("home_team"))
            else:
                winner = str(source.get("away_team"))

            ml_pick = str(source.get("moneyline_pick", ""))
            if winner == "TIE":
                ml_result = "P"
            elif not ml_pick:
                ml_result = ""
            else:
                ml_result = "W" if ml_pick == winner else "L"

            row = source.to_dict()
            row.update(
                {
                    "away_score": away_score,
                    "home_score": home_score,
                    "final_score": (
                        f"{source.get('away_team')} {int(away_score)} - "
                        f"{source.get('home_team')} {int(home_score)}"
                    ),
                    "spread_result": spread_result,
                    "spread_profit_units": profit_for_result(
                        spread_result, source.get("spread_odds")
                    ),
                    "total_result": total_result,
                    "total_profit_units": profit_for_result(
                        total_result, source.get("total_odds")
                    ),
                    "moneyline_result": ml_result,
                    "moneyline_profit_units": profit_for_result(
                        ml_result, source.get("moneyline_price")
                    ),
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.DataFrame(rows)
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan

    return history[HISTORY_COLUMNS].sort_values(
        ["season", "week", "gameday", "game_id"],
        ascending=[False, False, True, True],
    )


def summarize_market(
    history: pd.DataFrame,
    result_col: str,
    profit_col: str,
    mask: pd.Series | None = None,
) -> dict:
    if history.empty:
        return {"decisions": 0, "wins": 0, "losses": 0, "pushes": 0}

    rows = history if mask is None else history[mask]
    rows = rows[rows[result_col].isin(["W", "L", "P"])].copy()

    wins = int(rows[result_col].eq("W").sum())
    losses = int(rows[result_col].eq("L").sum())
    pushes = int(rows[result_col].eq("P").sum())
    decisions = wins + losses
    hit_rate = wins / decisions if decisions else None

    profits = numeric(rows[profit_col]).dropna()
    roi = float(profits.sum() / len(profits)) if len(profits) else None

    return {
        "decisions": decisions,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": None if hit_rate is None else round(hit_rate, 4),
        "profit_units": round(float(profits.sum()), 3) if len(profits) else None,
        "roi_per_bet": None if roi is None else round(roi, 4),
    }


def write_summary(history: pd.DataFrame, output_dir: Path) -> None:
    if history.empty:
        spread_mask = pd.Series(dtype=bool)
        total_mask = pd.Series(dtype=bool)
    else:
        spread_mask = history["spread_status"].isin(["WATCH", "VALIDATED"])
        total_mask = history["total_status"].isin(["WATCH", "VALIDATED"])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "integrity_note": (
            "Forward records only. Weeks are locked before any target-week "
            "game has a final score; retroactive reruns are never graded."
        ),
        "spread_qualified": summarize_market(
            history,
            "spread_result",
            "spread_profit_units",
            spread_mask if not history.empty else None,
        ),
        "total_qualified": summarize_market(
            history,
            "total_result",
            "total_profit_units",
            total_mask if not history.empty else None,
        ),
        "moneyline_all_model_picks": summarize_market(
            history,
            "moneyline_result",
            "moneyline_profit_units",
        ),
    }

    with (output_dir / "bet_history_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="outputs/latest_predictions.csv",
    )
    parser.add_argument(
        "--model-report",
        default="outputs/model_report.json",
    )
    parser.add_argument(
        "--archive-dir",
        default="outputs/game_predictions",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    report_path = Path(args.model_report)
    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise RuntimeError(f"Missing predictions file: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    report = load_report(report_path)

    seasons = seasons_from_archives(archive_dir)
    if not predictions.empty:
        seasons.update(
            int(value)
            for value in predictions["season"].dropna().unique()
        )

    schedules = load_schedules(sorted(seasons))
    board = enrich_board(predictions, schedules, report)
    board.to_csv(output_dir / "latest_betting_board.csv", index=False)

    archive_board(board, archive_dir)

    # Reload schedules if archives include additional seasons.
    all_seasons = seasons_from_archives(archive_dir) | seasons
    schedules = load_schedules(sorted(all_seasons))
    history = grade_archives(archive_dir, schedules)
    history.to_csv(output_dir / "game_bet_history.csv", index=False)
    write_summary(history, output_dir)

    print("\nGAME BETTING LEDGER COMPLETE")
    print("=" * 72)
    print(f"Current board rows: {len(board):,}")
    print(f"Settled forward game rows: {len(history):,}")


if __name__ == "__main__":
    main()
