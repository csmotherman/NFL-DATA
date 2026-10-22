# 2026 Week 6 fantasy PPR trade audit (October 22, 2026)

## Data integrity
The repository's last completed model outputs were published October 15 and its opponent-adjusted fantasy leaderboard is through Week 5. Its older `fantasy_ros.py` is an intentionally Week-3-frozen model of Weeks 4-16, **not** a current Week 7-17 forecast.

The new `src/fantasy_trade_market.py` screens QB/RB/WR/TE through the newest weekly stat data available at runtime: actual PPR vs opportunity-normalized PPR; per-play efficiency, target/rush shares, defensive strength faced and future opponent schedule through Week 17. Output: `outputs/fantasy_trade_market.csv` and `outputs/fantasy_trade_market_report.json`. Those files require a successful Actions job to populate. This new model is a heuristic research screen, *not* validated against historical seasons or calibrated to injury/market values.

## Immediate targets — current external Weeks 1-6 data; Week-5 repo adjustment explicitly labeled

### Buy: Chase Brown (CIN RB)
- Weeks 1-6: 14.2 carries and 5.6 targets per game; 70.2% snap share; 0.41 PPR points per opportunity, 20th percentile in comparison feed. 13.0 PPR/game actual vs 15.9 independent ROS projection.
- Through Week 5, repo opponent-adjustment boosted PPR/game 13.00 to 13.82 (+6.3%), supporting tougher-than-average early opposing defenses.
- A stable ~20-opportunity role with underproduced touchdowns is more investable than hoping for random efficiency. Verify continued role and Bengals health before an offer.
- Sources: https://statchasers.com/nfl/players/chase-brown/stats/ ; https://propsims.com/articles/fantasy/2026-week-7-trade-targets-buy-low-sell-high/

### Buy: Parker Washington (JAX WR)
- Through Week 6: approximately 26.3% targets per route run (charted); a recent NBC analysis ranked target share 15th and targets per route run 18th, with the 7th-easiest remaining WR schedule and 6th-easiest fantasy playoff schedule.
- A route/target-driven buy, not a blind low-PPR buy. Verify recent teammate returns before paying a premium.
- Sources: https://www.thefantasyfootballers.com/analysis/targets-per-route-run-report-week-6-fantasy-football-4/ ; https://www.nbcsports.com/fantasy/football/news/moves-to-make-before-week-7-fantasy-football-buy-parker-washington-sell-bryce-young

### Buy: Drake London (ATL WR)
- 550 receiving yards, zero TD through six games, and 10 targets in Week 6 per contemporaneous reporting. Strong opportunity with fewer TDs than market might expect, but no TD is guaranteed.
- Source: https://www.profootballnetwork.com/fantasy-football/buy-low-sell-high-trade-targets-week-7-2026/

### Sell high at WR1 pricing: Christian Watson (GB WR)
- Six receiving touchdowns on 48 targets through six games: 12.5% TD/target. 8 targets/game, 11.2 yards/target, 2.61 yards/route and +24.4 PPR points over one vendor's expected-points estimate.
- PPR 19.8/game vs an independent ROS of 12.2/game. Through Week 5 the repo adjusted him *up* +5.6% for opponents, so this is a TD-efficiency regression case, **not** an easy-defenses case. Strong usage argues against selling for WR3 value.
- Sources: https://corefootballlabs.com/players/c61ba03f-ee3d-45f1-b6ae-a6bb896bbd1b ; https://statchasers.com/nfl/players/christian-watson/stats/ ; https://propsims.com/articles/fantasy/2026-week-7-trade-targets-buy-low-sell-high/

### Conditional sell at RB1 pricing: Chuba Hubbard (CAR RB)
- 19.6 PPR/game, 6 total touchdowns in 5 games; 37% of fantasy production from touchdowns. But 94% snap share and 30 opportunities in Week 6 are sustainable-usage positives.
- Through Week 5 opponent adjustment *raised* his 19.75 PPR/game to 21.30 (+7.8%). Do not call easy defenses his cause for selling.
- An independent ROS model projects 12.2 PPR/game. The high recent role makes that projection contestable. Only trade if a counterpart prices him like an upper RB1.
- Sources: https://statchasers.com/nfl/players/chuba-hubbard/stats/ ; https://fantasytradetarget.com/fantasy-football-usage/week-6/running-backs ; https://propsims.com/articles/fantasy/2026-week-7-trade-targets-buy-low-sell-high/

### Hold / disagree with knee-jerk sell: Nico Collins (HOU WR)
- Through Week 5 repo adjustment reduced his raw 20.87 to 18.50 PPR/game (-11.3%) for easy defenses. But current route efficiency is ~2.67 YPRR and ~23-26% TPRR, with ~23-26% team target share depending on charting feed.
- One ROS model rates him only WR29; a contemporaneous CBS PPR trade chart values him substantially higher. The role/efficiency evidence argues against a fire sale.
- Sources: https://www.therundown.gg/research/nfl/player/nico-collins?gt=REG&season=2026 ; https://www.cbssports.com/fantasy/football/news/dave-richards-2026-week-7-trade-chart-and-rest-of-season-rankings/ ; https://propsims.com/articles/fantasy/2026-week-7-trade-targets-buy-low-sell-high/

### Conditional sell: Ashton Jeanty (LV RB)
- Week 6: 10 carries for 22 yards, 4 catches for 36 yards and a rushing TD; Mike Washington had 9 carries, narrowing the workload.
- Through Week 5 repo opponent adjustment reduced 16.92 to 15.63 PPR/game (-7.6%). Contradictory external assessments: one evaluator says sell; another projects Jeanty RB5 ROS. Treat as a **contested** case, sell only for RB1 offers.
- Source: https://www.profootballnetwork.com/fantasy-football/buy-low-sell-high-trade-targets-week-7-2026/

## Quality controls and limits
1. Signal strength demands sustained opportunity and a meaningful value gap, not simply last week's box score.
2. Future opponents must be strength-adjusted by the offensive position groups the defenses faced, rather than raw PPR allowed.
3. Touchdowns, routes run, injury likelihood, roster changes, target earning and QB changes affect real value; the code's new screen does not model all of them.
4. Real sell-high/buy-low is **price conditional**. Verify who your trading partner will actually give up.
5. Avoid using the stale Week-3-frozen ROS CSV as current trade rankings, even after the Thursday workflow refresh.
