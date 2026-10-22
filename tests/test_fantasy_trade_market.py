"""Regression tests: retain high-skill veterans and resist a one-game carry dip."""
import unittest
import pandas as pd
import numpy as np
from src.fantasy_trade_market import player_skill_rates, projected_role, team_pace_context, STAT_COLS


class FantasyModelContract(unittest.TestCase):
    def test_known_elite_vs_league_smoothing(self):
        cur = pd.DataFrame([
            dict(player_id="elite", position="RB", carries=20., rushing_yards=120.,
                 rushing_tds=1., targets=5., receptions=4., receiving_yards=40., receiving_tds=0.),
            dict(player_id="avg", position="RB", carries=20., rushing_yards=60.,
                 rushing_tds=0., targets=5., receptions=3., receiving_yards=15., receiving_tds=0.),
        ])
        past = pd.DataFrame([
            dict(player_id="elite", position="RB", carries=250., rushing_yards=1500.,
                 rushing_tds=12., targets=70., receptions=55., receiving_yards=630.,
                 receiving_tds=4., history_weight=.65),
            dict(player_id="avg", position="RB", carries=250., rushing_yards=800.,
                 rushing_tds=3., targets=70., receptions=30., receiving_yards=230.,
                 receiving_tds=1., history_weight=.65),
        ])
        for c in STAT_COLS:
            if c not in cur: cur[c]=0.
        skill = player_skill_rates(cur, past)
        self.assertGreater(skill[("elite", "RB")]["rush_yards"], skill[("avg", "RB")]["rush_yards"])
        self.assertGreater(skill[("elite", "RB")]["rec_yards"], skill[("avg", "RB")]["rec_yards"])
        self.assertGreater(skill[("elite", "RB")]["rush_td"], skill[("avg", "RB")]["rush_td"])

    def test_game_script_does_not_override_yearlong_role(self):
        rows=[]
        for week, carries in enumerate([20,22,21,20,22,9],start=1):
            rows.extend([
                dict(game_id=f"g{week}", week=week, team="ATL", player_id="bijan",
                     carries=float(carries), targets=5.,position="RB"),
                dict(game_id=f"g{week}", week=week, team="ATL", player_id="backup",
                     carries=float(30-carries), targets=25.,position="RB"),
            ])
        all_players=team_pace_context(pd.DataFrame(rows))
        p=all_players[all_players["player_id"].eq("bijan")]
        proj_c,proj_t=projected_role(all_players,p,"ATL")
        self.assertGreater(proj_c, 15., "One bad carry week should not define role")
        self.assertLess(proj_c, 21.)
        self.assertAlmostEqual(proj_t, 5., places=2)

    def test_no_historical_rows_still_produces_finite_rates(self):
        cur = pd.DataFrame([dict(player_id="rookie", position="WR", carries=0,
              rushing_yards=0,rushing_tds=0,targets=6,receptions=3,
              receiving_yards=55,receiving_tds=0)])
        history = pd.DataFrame(columns=list(cur.columns) + ["history_weight"])
        for c in STAT_COLS:
            if c not in cur: cur[c]=0.
        rate = player_skill_rates(cur, history)[("rookie","WR")]
        self.assertTrue(all(np.isfinite(x) for x in rate.values()))
        self.assertGreater(rate["rec_yards"], 0)


if __name__ == "__main__":
    unittest.main()
