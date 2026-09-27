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
- `outputs/team_state.csv` — current rolling team profiles.\n- `outputs/latest_player_props.csv` — upcoming QB/RB/WR/TE projections.\n- `outputs/latest_player_prop_edges.csv` — projection-vs-line discrepancies when prop lines are supplied.\n- `outputs/player_prop_model_report.json` — walk-forward MAE, baseline comparison, blend weights and uncertainty by prop market.\n- `outputs/player_prop_walkforward_oof.csv` — out-of-fold historical player prop predictions.

## Model design

The model uses rolling **pregame-only** team information. Historical features are shifted before each game, so a game's result is never used to predict itself.

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
