"""Regression tests for forward-locked game betting history grading."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from grade_game_history import grade_archives, write_summary


class GameHistoryGradingTests(unittest.TestCase):
    def test_missing_picks_never_count_as_wagers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "locked"
            archive.mkdir()
            games = [
                {
                    "season": 2026, "week": 8, "gameday": "2026-11-01",
                    "game_id": "game_without_picks", "home_team": "DET", "away_team": "MIN",
                    "spread_line": 3.5, "total_line": 45.5,
                    "spread_pick": "", "total_pick": "", "moneyline_pick": "",
                    "spread_status": "NO BET", "total_status": "NO BET",
                },
                {
                    "season": 2026, "week": 8, "gameday": "2026-11-01",
                    "game_id": "game_with_picks", "home_team": "PIT", "away_team": "CLE",
                    "spread_line": 3.5, "total_line": 45.5,
                    "spread_pick": "PIT", "total_pick": "UNDER", "moneyline_pick": "PIT",
                    "spread_status": "VALIDATED", "total_status": "VALIDATED",
                    "spread_odds": -110, "total_odds": -110, "moneyline_price": -120,
                },
                {
                    "season": 2026, "week": 8, "gameday": "2026-11-01",
                    "game_id": "tied_without_pick", "home_team": "CHI", "away_team": "SEA",
                    "spread_line": 0.0, "total_line": 40.0,
                    "spread_pick": "", "total_pick": "", "moneyline_pick": "",
                    "spread_status": "NO BET", "total_status": "NO BET",
                },
            ]
            pd.DataFrame(games).to_csv(archive / "2026_week_08_game_bets.csv", index=False)
            schedule = pd.DataFrame([
                {"game_id": "game_without_picks", "home_score": 24, "away_score": 17},
                {"game_id": "game_with_picks", "home_score": 24, "away_score": 17},
                {"game_id": "tied_without_pick", "home_score": 20, "away_score": 20},
            ])
            graded = grade_archives(archive, schedule).set_index("game_id")
            for game_id in ["game_without_picks", "tied_without_pick"]:
                row = graded.loc[game_id]
                for market in ["spread", "total", "moneyline"]:
                    self.assertEqual(row[f"{market}_result"], "")
                    self.assertTrue(pd.isna(row[f"{market}_profit_units"]))

            placed = graded.loc["game_with_picks"]
            for market in ["spread", "total", "moneyline"]:
                self.assertEqual(placed[f"{market}_result"], "W")
                self.assertGreater(placed[f"{market}_profit_units"], 0)

            write_summary(graded.reset_index(), root)
            report = json.loads((root / "bet_history_summary.json").read_text())
            for market in ["spread_qualified", "total_qualified", "moneyline_all_model_picks"]:
                self.assertEqual(report[market]["decisions"], 1)
                self.assertEqual(report[market]["wins"], 1)
                self.assertEqual(report[market]["losses"], 0)
                self.assertEqual(report[market]["pushes"], 0)


if __name__ == "__main__":
    unittest.main()
