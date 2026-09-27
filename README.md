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
- `outputs/team_state.csv` — current rolling team profiles.

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

Two independent regressors predict:

- home-team scoring margin
- game total

The model does **not** use the sportsbook spread or total as a training feature. Market numbers are kept separate so the output can measure model-vs-market disagreement honestly.

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

Free nflverse data is the foundation of this project. nflreadpy is the Python loader.
