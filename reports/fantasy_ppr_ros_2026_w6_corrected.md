# Fantasy FLEX rest of season — Full PPR (Week 6 snapshot)

Generated October 22, 2026. Based on Weeks 1-6 with per-game projections for scheduled Weeks 7-17. **These are uncalibrated conditional-on-playing forecasts**, not trade-market rankings. Current Week 7 injury reports and in-season role changes may materially change them.

Model: personal efficiency rates (current season + up to three prior years with sample-size priors); team pace and carry/target shares (80% season / 20% recent median); opponent effect capped at 1.5 PPR points/game. Past performance does not guarantee future performance.

## Top 25 by projected PPR per game

| # | Player | Pos | Team | Current PPR/G | Corrected ROS PPR/G |
|--:|---|:--:|:--:|--:|--:|
| 1 | Jahmyr Gibbs | RB | DET | 30.7 | **29.6** |
| 2 | Puka Nacua | WR | LA | 22.9 | **24.5** |
| 3 | Jaxon Smith-Njigba | WR | SEA | 27.5 | **21.9** |
| 4 | CeeDee Lamb | WR | DAL | 21.9 | **21.8** |
| 5 | Jonathan Taylor | RB | IND | 19.8 | **21.2** |
| 6 | Kenneth Walker III | RB | KC | 27.6 | **20.6** |
| 7 | Amon-Ra St. Brown | WR | DET | 20.8 | **20.6** |
| 8 | Bijan Robinson | RB | ATL | 21.4 | **20.4** |
| 9 | Kyren Williams | RB | LA | 19.8 | **20.1** |
| 10 | Isaiah Likely | TE | NYG | 18.2 | **19.5** |
| 11 | DeVonta Smith | WR | PHI | 17.3 | **19.5** |
| 12 | Derrick Henry | RB | BAL | 19.5 | **19.2** |
| 13 | Trey McBride | TE | ARI | 17.2 | **19.2** |
| 14 | James Cook | RB | BUF | 18.9 | **18.8** |
| 15 | Brock Bowers | TE | LV | 20.6 | **18.7** |
| 16 | Christian McCaffrey | RB | SF | 20.3 | **18.5** |
| 17 | Jalen Coker | WR | CAR | 17.0 | **18.2** |
| 18 | Malik Nabers | WR | NYG | 17.1 | **17.6** |
| 19 | Jaylen Warren | RB | PIT | 17.1 | **17.6** |
| 20 | Zay Flowers | WR | BAL | 18.9 | **17.6** |
| 21 | Ja'Marr Chase | WR | CIN | 16.0 | **17.1** |
| 22 | Tetairoa McMillan | WR | CAR | 17.8 | **17.1** |
| 23 | Chris Olave | WR | NO | 18.8 | **16.7** |
| 24 | Aaron Jones | RB | MIN | 14.0 | **16.6** |
| 25 | Christian Watson | WR | GB | 19.8 | **16.6** |

## Historical holdout check

- 2024 and 2025 frozen at Week 6; true Weeks 7-17 used only for evaluating active-game scoring.
- 609 player-season samples across RB, WR, TE; corrected model MAE **2.5597 PPR/G** versus season-to-date PPG baseline **2.635** (improvement 2.9%).
- Important: 2024 WR holdout worsened (model MAE 3.0156 versus 2.8296 baseline), which demonstrates this is not equally robust for every position or season.
- Missing-games risk is NOT modeled; backtest requires 3 or more subsequent active games and therefore has survival bias.
- More validation years, goal-line opportunities, routes run, QB changes, team injuries and contextual coaching adjustments still need work.

## Trade candidates (conditional on achievable prices)

| Side | Player | Current PPR/G | ROS PPR/G | Mechanism | Qualification |
|---|---|---:|---:|---|---|
| BUY | Tucker Kraft | 9.4 | 13.2 | Receiving opportunity + expected scoring rebound | Check any TE depth-chart changes |
| BUY | DK Metcalf | 11.5 | 14.6 | Stable target volume and expected efficiency rebound | Buy at WR3 price, not established WR1 price |
| BUY | Chase Brown | 13.0 | 16.1 | ~5.6 targets and ~20 opportunities per game | Verify Bengals backfield remains stable |
| BUY | George Pickens | 13.6 | 16.2 | ~9 target opportunities and per-target upside | Competes with CeeDee Lamb; WR2-3 pricing only |
| BUY | Aaron Jones | 14.0 | 16.6 | Strong rushing workload despite modest point total | Investigate week-to-week committee and health |
| SELL | Christian Watson | 19.8 | 16.6 | TD pace has outrun longer-term rates | Only at high WR2 / WR1 value |
| SELL | Javonte Williams | 18.2 | 14.6 | TD dependence with ~18 opportunities a game | Only if returning solid RB1 price |

## No automatic trade signal

- Bijan Robinson: corrected from 16.2 to 20.4 projected PPR/G after preserving his historical talent and normalizing role/game script. Do not sell Bijan for a generic FLEX8 price.
- Nico Collins: 15.7 modeled PPR/G, but only four observed games; competing Houston targets and health demand manual review.
- Emeka Egbuka: 12.6 modeled PPR/G, but ongoing concussion/foot issues; no unconditional buy signal.
- Brock Bowers, Gibbs, JSN: elite target/usage profiles; do not sell based solely on scoring regression.
- Davante Adams and Quinshon Judkins: explicit injury/role override rows, exclude from unconditional offers.

## Files

- [Full player-level data](../outputs/fantasy_trade_market.csv)
- [Validation](../outputs/fantasy_trade_validation.json)
- [Implementation](../src/fantasy_trade_market.py)
- [Tests](../tests/test_fantasy_trade_market.py)
- [Successful refresh run](https://github.com/csmotherman/NFL-DATA/actions/runs/39801598948)
