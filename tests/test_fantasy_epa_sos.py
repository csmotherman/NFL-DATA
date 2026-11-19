"""Regression tests for the fantasy opponent-adjusted EPA schedule."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fantasy_epa_sos import calculate_defenses, build_schedules, last_fully_completed_week


class FantasyEpaSosTests(unittest.TestCase):
    def setUp(self):
        self.teams = ["A", "B", "C", "D"]
        offenses = {"A": .10, "B": -.03, "C": .02, "D": -.05}
        pass_allowed = {"A": -.15, "B": .22, "C": -.05, "D": .12}
        rush_allowed = {"A": -.10, "B": -.02, "C": .15, "D": .04}
        rows = []
        for week in range(1, 11):
            for ix, team in enumerate(self.teams):
                opponent = self.teams[(ix + week % 3 + 1) % 4]
                if opponent == team:
                    opponent = self.teams[(ix + 1) % 4]
                rows.append({
                    "team": team, "opponent_team": opponent, "week": week,
                    "off_pass_epa_db": offenses[team] + pass_allowed[opponent],
                    "off_rush_epa_att": offenses[team] + rush_allowed[opponent],
                    "off_plays": 60., "off_pass_rate": .60,
                    "points_for": 21, "points_against": 20,
                })
        self.games = pd.DataFrame(rows)

    def test_high_defensive_epa_allowed_is_easier(self):
        d = calculate_defenses(self.games, self.teams, 10).set_index("team")
        self.assertGreater(
            d.loc["B", "blended_def_pass_epa_allowed"],
            d.loc["A", "blended_def_pass_epa_allowed"]
        )
        self.assertGreater(
            d.loc["C", "blended_def_rush_epa_allowed"],
            d.loc["A", "blended_def_rush_epa_allowed"]
        )

    def test_future_games_cannot_contaminate_adjusted_epa(self):
        baseline = calculate_defenses(self.games, self.teams, 10)
        future = self.games.assign(week=11, off_pass_epa_db=400., off_rush_epa_att=400.)
        actual = calculate_defenses(pd.concat([self.games, future]), self.teams, 10)
        pd.testing.assert_frame_equal(baseline, actual)

    def test_separate_pass_rush_and_week14_bye(self):
        schedule = pd.DataFrame([
            {"week": 11, "home_team": "A", "away_team": "B"},
            {"week": 11, "home_team": "C", "away_team": "D"},
            {"week": 12, "home_team": "A", "away_team": "C"},
            {"week": 12, "home_team": "B", "away_team": "D"},
            {"week": 13, "home_team": "A", "away_team": "D"},
            {"week": 13, "home_team": "B", "away_team": "C"},
            {"week": 14, "home_team": "B", "away_team": "C"},
        ])
        defense = calculate_defenses(self.games, self.teams, 10)
        weekly, rankings = build_schedules(schedule, defense, 10, 11, 14, 14)
        self.assertFalse(((weekly["team"] == "A") & (weekly["week"] == 14)).any())
        b = rankings.query("team == 'B' and position == 'QB' and window == 'playoffs'")
        self.assertEqual(int(b.iloc[0]["games"]), 1)
        one = weekly.query("team == 'A' and week == 11")
        self.assertEqual(
            int(one[one["position"].eq("QB")].iloc[0]["opponent_defense_ease_rank"]),
            int(defense.set_index("team").loc["B", "pass_defense_ease_rank"])
        )
        self.assertEqual(
            int(one[one["position"].eq("RB")].iloc[0]["opponent_defense_ease_rank"]),
            int(defense.set_index("team").loc["B", "rush_defense_ease_rank"])
        )

    def test_midweek_not_treated_as_complete(self):
        schedule = pd.DataFrame([
            {"week": 1, "home_score": 20., "away_score": 7.},
            {"week": 2, "home_score": 20., "away_score": 7.},
            {"week": 3, "home_score": 20., "away_score": 7.},
            {"week": 3, "home_score": None, "away_score": None},
            {"week": 4, "home_score": None, "away_score": None},
        ])
        self.assertEqual(last_fully_completed_week(schedule), 2)

    def test_duplicate_schedule_rows_fail(self):
        defense = calculate_defenses(self.games, self.teams, 10)
        schedule = pd.DataFrame([
            {"week": 11, "home_team": "A", "away_team": "B"},
            {"week": 11, "home_team": "A", "away_team": "C"},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_schedules(schedule, defense, 10, 11, 11, 11)


if __name__ == "__main__":
    unittest.main()
