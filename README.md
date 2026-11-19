# NFL-DATA

Automated NFL betting-research pipeline built on free nflverse data.

## What it does

Every Thursday at **12:00 PM America/Detroit**, GitHub Actions:

1. refreshes nflverse team and schedule data;
2. rebuilds rolling team offense/defense profiles with no future-game leakage;
3. trains historical models for **home margin** and **game total**;
4. evaluates the model on the most recent completed holdout season;
5. produces the upcoming week's game projections;
6. compares independent model projections with available market spread/total lines;
7. creates pass/run matchup scores and rough volume projections;
8. writes fresh CSV/JSON outputs to `outputs/`;
9. commits those outputs back to the repository.

The goal is **research and price discovery**, not to force a bet on every game.

## Main outputs

- `outputs/latest_predictions.csv` — upcoming games, model margin/total, market lines and edges.
- `outputs/latest_matchups.csv` — pass/run matchup and projected-volume board.
- `outputs/latest_candidates.csv` — only games passing configurable edge filters.
- `outputs/model_report.json` — holdout MAE, directional accuracy and threshold backtests.
- `outputs/team_state.csv` — current rolling team profiles.\n- `outputs/team_stats.csv` — current-season raw and opponent-adjusted offense/defense EPA per play, pass and rush, with ranks.\n- `outputs/latest_player_props.csv` — upcoming QB/RB/WR/TE projections.\n- `outputs/latest_player_prop_edges.csv` — projection-vs-line discrepancies when prop lines are supplied.\n- `outputs/player_prop_model_report.json` — walk-forward MAE, baseline comparison, blend weights and uncertainty by prop market.\n- `outputs/player_prop_walkforward_oof.csv` — out-of-fold historical player prop predictions.

## Model design

The model uses rolling **pregame-only** team information. Historical features are shifted before each game, so a game's result is never used to predict itself.

The site also exposes a separate **Team Stats** view for current-season EPA. It fits a play-weighted, regularized two-way model to each team-game observation so offensive EPA is adjusted for the defenses faced and defensive EPA allowed is adjusted for the offenses faced. Pass EPA is per dropback (attempts + sacks), rush EPA is per carry, and defensive EPA is displayed as EPA allowed, where lower is better. Raw EPA is retained beside the adjusted values in `outputs/team_stats.csv` for auditing.

Core inputs include:

- pass EPA/dropback
- rush EPA/carry
- yards/dropback
- yards/carry
- CPOE
- sack rate
- first-down rates
- pass/rush volume
- turnover rate
- opponent versions of those metrics (defensive allowance)
- recent scoring margin
- plays/game
- rest differential
- week
- neutral-site indicator

The pipeline has two separate modeling layers:

1. **Independent football model** — predicts home margin and game total without sportsbook lines.
2. **Market-residual model** — uses historical football features plus the posted spread/total to learn where outcomes have systematically differed from market prices.

This separation matters. The independent model is a football sanity check; the residual model is the betting-research layer. The report always compares both against the market on held-out data instead of assuming model disagreement is profitable.

## Run locally

```bash
python -m pip install -r requirements.txt
python src/nfl_edge.py --auto
```

Force a specific season/week:

```bash
python src/nfl_edge.py --season 2026 --week 4
```

Run with a different candidate threshold:

```bash
python src/nfl_edge.py --auto --spread-edge 3.0 --total-edge 3.0
```

## GitHub Actions

The scheduled workflow is in:

```
.github/workflows/thursday-nfl-edge.yml
```

It also supports **workflow_dispatch**, so you can run it manually from the Actions tab at any time.

## Important interpretation

A model edge is not automatically a profitable bet. The useful question is whether the model's disagreement with the market has held up **out of sample**.

The report therefore includes threshold backtests (for example 1.5, 2.5, 3.5 and 4.5 points) on the holdout season. Treat small samples skeptically.

If the market beats the model out of sample, that is a failed hypothesis—not a signal to bet harder. The point of this repository is to find repeatable edges and reject the ones that do not survive validation.

Free nflverse data is the foundation of this project. nflreadpy is the Python loader.


## Player prop models

The prop engine currently projects:

- pass attempts
- completions
- passing yards
- rush attempts
- rushing yards
- receptions
- receiving yards

Player models use recent 3-game and 6-game form, season-to-date form, usage shares, efficiency, team pace and pass rate, opponent defensive efficiency, rest, spread, total and implied team points.

Each prop market is validated with walk-forward seasons rather than a random train/test split. The pipeline compares ML against a simple recent-player baseline and automatically chooses the blend weight that produced the lowest out-of-sample MAE.

To compare against sportsbook lines, populate:

```
inputs/player_prop_lines.csv
```

with rows like:

```csv
player_id,player_name,market,line,book,price_over,price_under
00-0034857,Josh Allen,passing_yards,267.5,example,-110,-110
```

Supported market names are `pass_attempts`, `completions`, `passing_yards`, `rush_attempts`, `rushing_yards`, `receptions`, and `receiving_yards`.

The resulting edge probability is currently a normal-error approximation based on historical residual dispersion. It is deliberately labeled as approximate until historical sportsbook prop-line archives are added and calibrated directly.


## Automatic live player prop lines

The Thursday workflow can pull live NFL player prop markets automatically from PropLine.

PropLine currently offers a free API key with 1,000 requests/day. Add the key to the GitHub repository as an Actions secret named:

```
PROPLINE_API_KEY
```

When that secret exists, the workflow automatically refreshes `inputs/player_prop_lines.csv` before running the player models. If the secret is absent, the pipeline falls back to whatever manual CSV is already present.

The live feed is used only as the market comparison layer. It does not replace nflverse as the football/statistical source.

## Matchup Breakdown (descriptive, not a betting model)

Use the **Matchup** tab to select any two teams. Both directions are shown: Team A offense vs Team B defense, and vice versa. Season-long opponent-adjusted EPA/play, per dropback and per carry come from `outputs/team_stats.csv`; recent unadjusted yards, first-down, completion, CPOE, sack and turnover rates come from rolling `outputs/team_state.csv`.

For each split, compute a sample standard deviation and empirical midrank percentile across the available NFL teams, separately for offensive production and defensive production allowed. Reverse the sign for offensive sack/turnover rates and defensive sacks/takeaways, where lower opponent pressure is favorable to an offense. The matchup index = (offense league z-score + defense *susceptibility* league z-score) / 2. Side-to-side index gaps describe which offense has the relatively easier efficiency environment. Indices are **not** outcome probabilities, predicted yardage, projected final scores, or betting edges. Percentiles describe 32 teams, not probabilistic odds, and the opponent-adjusted EPA splits are regularized early in the season. A strength-vs-strength matchup can have a neutral composite index. If data are missing or have zero variance, the interface suppresses the edge instead of inventing one.

The Matchup tab shares the weekly pipeline data refresh and is deployed by Vercel via `matchup.js`. `NFL Matchup Site Checks` runs independent JavaScript syntax and calculation tests on frontend changes.


## Fantasy EPA strength of schedule (NFL 2026, fantasy playoffs Weeks 14-17)

Run the NFL API-backed schedule screen after a fully completed week:

```bash
python src/fantasy_epa_sos.py --season 2026 --start-week 11 --playoff-start 14 --end-week 17 --output-dir outputs
```

This imports current-season completed team-game stats and the scheduled regular-season opponents using `nflreadpy.load_team_stats` and `nflreadpy.load_schedules`. It fits the same play-weighted **opponent-adjusted offense/defense two-way ridge** used by Team Stats separately for pass EPA per dropback and rush EPA per carry. The *recent* estimate reruns the entire opponent-adjusted model with exponential time-decay (**four-week half-life**), retaining schedule connectivity instead of assuming last-three-game *raw* defensive EPA is opponent adjusted.

The SOS score blends **65% full-season adjusted defense EPA allowed** with **35% recent adjusted defense EPA allowed**. Higher is an easier opponent matchup. QB, WR and TE use the opposing defense's passing EPA **allowed**; RB uses rushing EPA **allowed**. These are team-level **schedule rankings**, not a player scoring forecast; targets, usage, injuries, pace, scoring settings, and TD opportunities are separate.

Outputs (refreshed by the existing Thursday pipeline):

- `outputs/fantasy_epa_defense.csv`: 32 defensive teams' overall, recent, blended adjusted pass/rush EPA allowed and susceptibility ranks.
- `outputs/fantasy_epa_sos_weekly.csv`: per-team, per-position, per-week matchup rankings with the three EPA values.
- `outputs/fantasy_epa_sos_team.csv`: pre-playoff, rest-of-season and **Weeks 14–17 fantasy-playoff** SOS rankings and actual opponents; `ease_rank_matchups_only` ranks matchup difficulty, while `rank_including_byes` puts teams with more playable games first.
- `outputs/fantasy_epa_sos_report.json`: data freshness, methodology and model limitations.

**Week-14 byes matter:** do not treat a favorable three-game playoff average as better than a four-game schedule. The script never invents an opponent for a bye, rejects future or unfinished game results, and advances the starting week as the season progresses. The old `fantasy_ros.py` Week-3-frozen Week-4-to-16 player model is a *separate historical backtest*; do not use its old SOS output for current Week-11 trade decisions.
