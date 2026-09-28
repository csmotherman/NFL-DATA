from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PRICE = -110.0
OUTER_SEASONS = list(range(2018, 2026))
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
SIDE_MODES = ["both", "positive", "negative"]
# Reverse orientation remains useful as a diagnostic, but allowing production
# to flip the meaning of the classifier is too easy to data-mine. A bet must
# agree with the trained probability direction.
ORIENTATIONS = ["normal"]


def numeric(values):
    return pd.to_numeric(values, errors="coerce")


def american_profit(odds: float) -> float:
    if odds is None or not np.isfinite(odds):
        odds = DEFAULT_PRICE
    odds = float(odds)
    if odds <= -100:
        return 100.0 / abs(odds)
    if odds >= 100:
        return odds / 100.0
    return 100.0 / 110.0


def break_even_probability(odds: float) -> float:
    if odds is None or not np.isfinite(odds):
        odds = DEFAULT_PRICE
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return 110.0 / 210.0


class MarketClassifier:
    """Low-variance probability model for ATS cover / over outcomes."""

    def __init__(self):
        self.linear = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.12,
                max_iter=3000,
                solver="lbfgs",
            )),
        ])
        self.tree = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=220,
                max_leaf_nodes=7,
                min_samples_leaf=45,
                l2_regularization=12.0,
                random_state=42,
            )),
        ])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ):
        if sample_weight is None:
            self.linear.fit(X, y)
            self.tree.fit(X, y)
        else:
            self.linear.fit(
                X, y, model__sample_weight=sample_weight
            )
            self.tree.fit(
                X, y, model__sample_weight=sample_weight
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_linear = self.linear.predict_proba(X)[:, 1]
        p_tree = self.tree.predict_proba(X)[:, 1]
        return np.clip(0.75 * p_linear + 0.25 * p_tree, 0.01, 0.99)


def implied_probability(values: pd.Series) -> pd.Series:
    odds = numeric(values)
    return pd.Series(
        np.where(
            odds < 0,
            (-odds) / ((-odds) + 100.0),
            np.where(
                odds > 0,
                100.0 / (odds + 100.0),
                np.nan,
            ),
        ),
        index=values.index,
        dtype=float,
    )


def no_vig_share(
    positive_odds: pd.Series,
    negative_odds: pd.Series,
) -> pd.Series:
    positive = implied_probability(positive_odds)
    negative = implied_probability(negative_odds)
    denom = positive + negative
    return pd.Series(
        np.where(denom > 0, positive / denom, np.nan),
        index=positive_odds.index,
        dtype=float,
    )


def engineered_features(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    extra_columns = [
        "spread_line", "total_line",
        "home_spread_odds", "away_spread_odds",
        "over_odds", "under_odds",
        "home_moneyline", "away_moneyline",
        "div_game", "weekday", "gametime", "roof",
    ]
    x = df.reindex(columns=base_features + extra_columns).copy()

    spread = numeric(x["spread_line"])
    total = numeric(x["total_line"])
    x["abs_spread_line"] = spread.abs()
    x["spread_line_sq"] = spread.pow(2)
    x["total_line_sq"] = total.pow(2)
    x["spread_total_interaction"] = spread * total
    x["home_favorite"] = (spread > 0).astype(float)

    # NFL spreads cluster around key football margins. Distance to those
    # numbers is more useful than asking a tree to rediscover the geometry.
    x["spread_dist_3"] = (spread.abs() - 3.0).abs()
    x["spread_dist_6"] = (spread.abs() - 6.0).abs()
    x["spread_dist_7"] = (spread.abs() - 7.0).abs()
    x["spread_dist_10"] = (spread.abs() - 10.0).abs()

    # The market price contains information beyond the headline line. Convert
    # both sides to no-vig shares so juice imbalance is comparable over time.
    x["spread_home_novig"] = no_vig_share(
        x["home_spread_odds"],
        x["away_spread_odds"],
    )
    x["total_over_novig"] = no_vig_share(
        x["over_odds"],
        x["under_odds"],
    )
    x["moneyline_home_novig"] = no_vig_share(
        x["home_moneyline"],
        x["away_moneyline"],
    )
    x["spread_juice_signal"] = x["spread_home_novig"] - 0.5
    x["total_juice_signal"] = x["total_over_novig"] - 0.5

    # Pregame-known context only. Avoid historical game-time weather because
    # the exact observed weather is unavailable at Thursday inference time.
    x["divisional_game"] = numeric(x["div_game"])
    roof = x["roof"].astype(str).str.lower()
    x["indoors"] = roof.isin(["dome", "closed"]).astype(float)
    weekday = x["weekday"].astype(str).str.lower()
    gametime_hour = pd.to_numeric(
        x["gametime"].astype(str).str.slice(0, 2),
        errors="coerce",
    )
    x["primetime"] = (
        weekday.str.startswith("thurs")
        | weekday.str.startswith("mon")
        | gametime_hour.ge(19)
    ).astype(float)

    # Drop raw strings after deriving stable numeric indicators.
    x = x.drop(
        columns=["weekday", "gametime", "roof"],
        errors="ignore",
    )

    for feature in base_features:
        if not feature.startswith("home_pre_"):
            continue
        suffix = feature[len("home_"):]
        away_feature = "away_" + suffix
        if away_feature in df.columns:
            x["diff_" + suffix] = numeric(df[feature]) - numeric(df[away_feature])

    return x


def training_weights(rows: pd.DataFrame) -> np.ndarray:
    """
    Markets and scoring environments adapt quickly. Down-weight stale seasons
    so a pattern from four or five years ago cannot dominate the current fit.
    """
    seasons = numeric(rows["season"]).to_numpy(dtype=float)
    latest = np.nanmax(seasons)
    gaps = np.maximum(latest - seasons, 0.0)
    return np.maximum(np.power(0.70, gaps), 0.10)


def market_rows(dataset: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "spread":
        line = numeric(dataset["spread_line"])
        delta = numeric(dataset["target_margin"]) - line
    else:
        line = numeric(dataset["total_line"])
        delta = numeric(dataset["target_total"]) - line

    rows = dataset[line.notna() & delta.notna() & delta.ne(0)].copy()
    if kind == "spread":
        rows["market_target"] = (
            numeric(rows["target_margin"]) - numeric(rows["spread_line"]) > 0
        ).astype(int)
    else:
        rows["market_target"] = (
            numeric(rows["target_total"]) - numeric(rows["total_line"]) > 0
        ).astype(int)
    return rows


def odds_columns(kind: str) -> tuple[str, str]:
    if kind == "spread":
        return "home_spread_odds", "away_spread_odds"
    return "over_odds", "under_odds"


def price_array(rows: pd.DataFrame, column: str) -> np.ndarray:
    if column not in rows.columns:
        return np.full(len(rows), DEFAULT_PRICE, dtype=float)
    values = numeric(rows[column]).to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = DEFAULT_PRICE
    return values


def annotate_probabilities(
    rows: pd.DataFrame,
    probability_positive: np.ndarray,
    kind: str,
    orientation: str = "normal",
) -> pd.DataFrame:
    out = rows.copy()
    p = np.asarray(probability_positive, dtype=float)
    if orientation == "reverse":
        p = 1.0 - p

    positive_odds_col, negative_odds_col = odds_columns(kind)
    positive_odds = price_array(out, positive_odds_col)
    negative_odds = price_array(out, negative_odds_col)

    positive_break_even = np.array([
        break_even_probability(v) for v in positive_odds
    ])
    negative_break_even = np.array([
        break_even_probability(v) for v in negative_odds
    ])

    positive_edge = p - positive_break_even
    negative_edge = (1.0 - p) - negative_break_even

    choose_positive = positive_edge >= negative_edge
    out["market_probability_positive"] = p
    out["market_side"] = np.where(choose_positive, 1, -1)
    out["market_side_probability"] = np.where(
        choose_positive, p, 1.0 - p
    )
    out["market_probability_edge"] = np.where(
        choose_positive, positive_edge, negative_edge
    )
    out["market_price"] = np.where(
        choose_positive, positive_odds, negative_odds
    )
    return out


def strategy_segments(kind: str) -> list[str]:
    if kind == "spread":
        return [
            "all",
            "favorite",
            "underdog",
            "short_line",
            "medium_line",
            "large_line",
            "division",
        ]
    return [
        "all",
        "low_total",
        "mid_total",
        "high_total",
        "outdoors",
        "indoors",
        "primetime",
    ]


def segment_mask(rows: pd.DataFrame, kind: str, segment: str) -> pd.Series:
    mask = pd.Series(True, index=rows.index, dtype=bool)
    if segment == "all":
        return mask

    if kind == "spread":
        spread = numeric(rows["spread_line"])
        side = numeric(rows["market_side"])
        favorite_side = np.sign(spread).replace(0, np.nan)

        if segment == "favorite":
            return side.eq(favorite_side)
        if segment == "underdog":
            return side.eq(-favorite_side)
        if segment == "short_line":
            return spread.abs().le(3.5)
        if segment == "medium_line":
            return spread.abs().gt(3.5) & spread.abs().le(7.5)
        if segment == "large_line":
            return spread.abs().gt(7.5)
        if segment == "division":
            return numeric(rows.get("div_game", np.nan)).eq(1)

    total = numeric(rows["total_line"])
    if segment == "low_total":
        return total.le(42.5)
    if segment == "mid_total":
        return total.gt(42.5) & total.lt(48.0)
    if segment == "high_total":
        return total.ge(48.0)

    roof = rows.get(
        "roof", pd.Series("", index=rows.index)
    ).astype(str).str.lower()
    if segment == "indoors":
        return roof.isin(["dome", "closed"])
    if segment == "outdoors":
        return ~roof.isin(["dome", "closed"])

    if segment == "primetime":
        weekday = rows.get(
            "weekday", pd.Series("", index=rows.index)
        ).astype(str).str.lower()
        gametime = rows.get(
            "gametime", pd.Series("", index=rows.index)
        ).astype(str)
        hour = pd.to_numeric(
            gametime.str.slice(0, 2), errors="coerce"
        )
        return (
            weekday.str.startswith("thurs")
            | weekday.str.startswith("mon")
            | hour.ge(19)
        )

    return mask


def evaluate_strategy(
    rows: pd.DataFrame,
    kind: str,
    strategy: dict,
) -> dict:
    if rows.empty or not strategy or strategy.get("side_mode") == "none":
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "roi": None,
            "profit_units": 0.0,
        }

    threshold = float(strategy["threshold"])
    side_mode = strategy["side_mode"]
    segment = strategy.get("segment", "all")

    mask = numeric(rows["market_probability_edge"]).ge(threshold)
    mask &= segment_mask(rows, kind, segment)
    if side_mode == "positive":
        mask &= numeric(rows["market_side"]).eq(1)
    elif side_mode == "negative":
        mask &= numeric(rows["market_side"]).eq(-1)

    bets = rows[mask].copy()
    if bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "roi": None,
            "profit_units": 0.0,
        }

    actual_side = np.where(
        numeric(bets["market_target"]).eq(1), 1, -1
    )
    picked_side = numeric(bets["market_side"]).to_numpy(dtype=int)
    wins_mask = picked_side == actual_side

    wins = int(np.sum(wins_mask))
    losses = int(len(bets) - wins)
    payouts = np.array([
        american_profit(v) for v in numeric(bets["market_price"])
    ])
    profits = np.where(wins_mask, payouts, -1.0)
    total_profit = float(np.sum(profits))
    roi = total_profit / len(bets)

    return {
        "bets": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(bets), 4),
        "roi": round(roi, 4),
        "profit_units": round(total_profit, 3),
    }


def season_breakdown(
    rows: pd.DataFrame,
    kind: str,
    strategy: dict,
) -> list[dict]:
    output = []
    seasons = sorted(
        int(value) for value in rows["season"].dropna().unique()
    )
    for season in seasons:
        stats = evaluate_strategy(
            rows[rows["season"].eq(season)],
            kind,
            strategy,
        )
        if stats["bets"]:
            output.append({"season": season, **stats})
    return output


def select_strategy(rows: pd.DataFrame, kind: str) -> dict:
    """
    Search direction, market regime and price-adjusted probability threshold
    using only prior out-of-sample rows. If nothing is repeatable, return
    NO BET.
    """
    if rows.empty:
        return {
            "side_mode": "none",
            "orientation": "normal",
            "segment": "all",
            "threshold": 1.0,
            "stable": False,
            "reason": "No prior out-of-sample rows.",
        }

    seasons = sorted(
        int(value) for value in rows["season"].dropna().unique()
    )
    min_bets = max(50, 18 * max(len(seasons), 1))
    candidates = []

    for orientation in ORIENTATIONS:
        oriented = annotate_probabilities(
            rows,
            rows["raw_probability_positive"].to_numpy(dtype=float),
            kind,
            orientation=orientation,
        )

        for segment in strategy_segments(kind):
            for side_mode in SIDE_MODES:
                for threshold in EDGE_THRESHOLDS:
                    strategy = {
                        "orientation": orientation,
                        "side_mode": side_mode,
                        "segment": segment,
                        "threshold": float(threshold),
                    }
                    stats = evaluate_strategy(
                        oriented, kind, strategy
                    )
                    if (
                        stats["bets"] < min_bets
                        or stats["roi"] is None
                    ):
                        continue

                    by_season = season_breakdown(
                        oriented, kind, strategy
                    )
                    profitable = sum(
                        1 for row in by_season
                        if row["roi"] is not None
                        and row["roi"] > 0
                    )
                    season_count = len(by_season)
                    season_rois = [
                        row["roi"] for row in by_season
                        if row["roi"] is not None
                    ]
                    roi_sd = (
                        float(np.std(season_rois))
                        if season_rois else 1.0
                    )
                    profitable_share = (
                        profitable / season_count
                        if season_count else 0.0
                    )

                    recent_rows = by_season[-2:]
                    recent_bets = sum(
                        row["bets"] for row in recent_rows
                    )
                    recent_profit = sum(
                        row["profit_units"] for row in recent_rows
                    )
                    recent_roi = (
                        recent_profit / recent_bets
                        if recent_bets else -1.0
                    )
                    latest_roi = (
                        by_season[-1]["roi"]
                        if by_season else -1.0
                    )

                    robust_score = (
                        0.55 * float(stats["roi"])
                        + 0.45 * recent_roi
                        - 0.25 * roi_sd
                        + 0.010 * math.log1p(stats["bets"])
                        + 0.02 * profitable_share
                    )
                    stable = (
                        season_count >= min(2, len(seasons))
                        and stats["roi"] > 0
                        and recent_roi > 0
                        and latest_roi is not None
                        and latest_roi > 0
                        and profitable_share >= 0.60
                    )

                    candidates.append({
                        **strategy,
                        **stats,
                        "seasons_with_bets": season_count,
                        "profitable_seasons": profitable,
                        "profitable_season_share": round(
                            profitable_share, 4
                        ),
                        "roi_sd": round(roi_sd, 4),
                        "recent_roi": round(
                            float(recent_roi), 4
                        ),
                        "latest_season_roi": round(
                            float(latest_roi), 4
                        ),
                        "robust_score": round(
                            robust_score, 6
                        ),
                        "stable": bool(stable),
                        "by_season": by_season,
                    })

    stable_candidates = [
        candidate for candidate in candidates
        if candidate["stable"]
    ]
    if not stable_candidates:
        return {
            "orientation": "normal",
            "side_mode": "none",
            "segment": "all",
            "threshold": 1.0,
            "stable": False,
            "reason": (
                "No prior out-of-sample strategy cleared "
                "volume and stability gates."
            ),
            "candidates_tested": len(candidates),
        }

    best = max(
        stable_candidates,
        key=lambda candidate: (
            candidate["robust_score"],
            candidate["roi"],
            candidate["bets"],
        ),
    )
    best["candidates_tested"] = len(candidates)
    return best


def generate_oof_predictions(
    dataset: pd.DataFrame,
    base_features: list[str],
    kind: str,
    max_season_exclusive: int,
) -> pd.DataFrame:
    rows = market_rows(dataset, kind)
    seasons = sorted(
        int(season)
        for season in rows["season"].dropna().unique()
        if int(season) < max_season_exclusive
    )

    outputs = []
    for validation_season in seasons:
        train = rows[rows["season"] < validation_season].copy()
        valid = rows[rows["season"] == validation_season].copy()

        if len(train) < 400 or len(valid) < 80:
            continue
        if train["market_target"].nunique() < 2:
            continue

        model = MarketClassifier().fit(
            engineered_features(train, base_features),
            train["market_target"],
            sample_weight=training_weights(train),
        )
        probability = model.predict_proba(
            engineered_features(valid, base_features)
        )

        fold = valid.copy()
        fold["raw_probability_positive"] = probability
        outputs.append(fold)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def fixed_threshold_diagnostics(
    rows: pd.DataFrame,
    kind: str,
) -> dict:
    if rows.empty:
        return {}

    oriented = annotate_probabilities(
        rows,
        rows["raw_probability_positive"].to_numpy(dtype=float),
        kind,
        orientation="normal",
    )
    output = {}

    for threshold in EDGE_THRESHOLDS:
        strategy = {
            "orientation": "normal",
            "side_mode": "both",
            "threshold": threshold,
        }
        stats = evaluate_strategy(oriented, kind, strategy)
        by_season = season_breakdown(
            oriented, kind, strategy
        )
        profitable = sum(
            1 for row in by_season
            if row.get("roi") is not None and row["roi"] > 0
        )
        season_count = len(by_season)
        stable = (
            stats["bets"] >= 100
            and stats["roi"] is not None
            and stats["roi"] > 0
            and season_count >= 3
            and profitable >= math.ceil(season_count * 0.60)
        )

        output[f"{threshold:.2f}"] = {
            **stats,
            "roi_at_minus_110": stats["roi"],
            "seasons_with_bets": season_count,
            "profitable_seasons": profitable,
            "stable": bool(stable),
            "by_season": by_season,
        }

    return output


def frozen_confirmation_test(
    dataset: pd.DataFrame,
    base_features: list[str],
    kind: str,
    discovery_end_season: int = 2021,
    confirmation_start_season: int = 2022,
    current_season: int = 2026,
) -> dict:
    """
    Select exactly one policy on early OOF seasons, freeze it, then grade that
    unchanged policy on later untouched seasons. This is intentionally harder
    than retuning a threshold every season.
    """
    full_oof = generate_oof_predictions(
        dataset,
        base_features,
        kind,
        max_season_exclusive=current_season,
    )
    discovery = full_oof[
        full_oof["season"] <= discovery_end_season
    ].copy()
    confirmation = full_oof[
        (full_oof["season"] >= confirmation_start_season)
        & (full_oof["season"] < current_season)
    ].copy()

    strategy = select_strategy(discovery, kind)
    if confirmation.empty:
        return {
            "discovery_through": int(discovery_end_season),
            "confirmation_from": int(confirmation_start_season),
            "frozen_strategy": strategy,
            "passed": False,
            "reason": "No confirmation rows available.",
        }

    confirmation = annotate_probabilities(
        confirmation,
        confirmation["raw_probability_positive"].to_numpy(dtype=float),
        kind,
        orientation=strategy.get("orientation", "normal"),
    )
    stats = evaluate_strategy(
        confirmation,
        kind,
        strategy,
    )
    by_season = season_breakdown(
        confirmation,
        kind,
        strategy,
    )
    profitable_seasons = sum(
        1 for row in by_season
        if row["roi"] is not None and row["roi"] > 0
    )
    seasons_with_bets = len(by_season)

    passed = (
        stats["bets"] >= 75
        and stats["roi"] is not None
        and stats["roi"] > 0
        and stats["win_rate"] is not None
        and stats["win_rate"] > break_even_probability(DEFAULT_PRICE)
        and seasons_with_bets >= 3
        and profitable_seasons
        >= math.ceil(seasons_with_bets * 0.60)
    )

    return {
        "discovery_through": int(discovery_end_season),
        "confirmation_from": int(confirmation_start_season),
        "frozen_strategy": strategy,
        **stats,
        "seasons_with_bets": int(seasons_with_bets),
        "profitable_seasons": int(profitable_seasons),
        "by_season": by_season,
        "passed": bool(passed),
    }


def walk_forward_strategy_validation(
    dataset: pd.DataFrame,
    base_features: list[str],
    current_season: int,
) -> dict:
    """
    Nested rolling-origin validation using one frozen OOF ledger per market.

    Every validation-season probability is produced by a model trained only on
    earlier seasons. Strategy selection for an outer season then sees only OOF
    rows from seasons before that outer season. This is equivalent to the
    previous nested process but avoids repeatedly fitting the same folds.
    """
    output = {}
    available = {
        int(season)
        for season in dataset["season"].dropna().unique()
        if int(season) < current_season
    }

    for kind in ["spread", "total"]:
        full_oof = generate_oof_predictions(
            dataset,
            base_features,
            kind,
            max_season_exclusive=current_season,
        )
        outer = []

        for test_season in OUTER_SEASONS:
            if test_season not in available:
                continue

            prior_oof = full_oof[
                full_oof["season"] < test_season
            ].copy()
            test_eval = full_oof[
                full_oof["season"] == test_season
            ].copy()

            if len(test_eval) < 80:
                continue

            strategy = select_strategy(
                prior_oof, kind
            )
            test_eval = annotate_probabilities(
                test_eval,
                test_eval[
                    "raw_probability_positive"
                ].to_numpy(dtype=float),
                kind,
                orientation=strategy.get(
                    "orientation", "normal"
                ),
            )
            stats = evaluate_strategy(
                test_eval, kind, strategy
            )

            outer.append({
                "season": int(test_season),
                "strategy": strategy,
                **stats,
            })

        total_bets = sum(row["bets"] for row in outer)
        total_wins = sum(row["wins"] for row in outer)
        total_losses = sum(row["losses"] for row in outer)
        total_profit = sum(
            row["profit_units"] for row in outer
        )

        roi = (
            total_profit / total_bets
            if total_bets else None
        )
        win_rate = (
            total_wins / total_bets
            if total_bets else None
        )
        profitable_seasons = sum(
            1 for row in outer
            if row["bets"]
            and row["roi"] is not None
            and row["roi"] > 0
        )
        seasons_with_bets = sum(
            1 for row in outer if row["bets"]
        )

        final_strategy = select_strategy(
            full_oof, kind
        )

        # Current-season shadow holdout: strategy is frozen using only prior
        # seasons, then tested on completed games from the current season.
        current_rows = market_rows(dataset, kind)
        current_rows = current_rows[
            current_rows["season"].eq(current_season)
        ].copy()
        current_shadow = {
            "season": int(current_season),
            "games": int(len(current_rows)),
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "roi": None,
            "profit_units": 0.0,
        }
        if len(current_rows) and final_strategy.get("side_mode") != "none":
            historical_rows = market_rows(dataset, kind)
            historical_rows = historical_rows[
                historical_rows["season"] < current_season
            ].copy()
            shadow_model = MarketClassifier().fit(
                engineered_features(historical_rows, base_features),
                historical_rows["market_target"],
                sample_weight=training_weights(historical_rows),
            )
            shadow_probability = shadow_model.predict_proba(
                engineered_features(current_rows, base_features)
            )
            shadow_eval = current_rows.copy()
            shadow_eval["raw_probability_positive"] = shadow_probability
            shadow_eval = annotate_probabilities(
                shadow_eval,
                shadow_probability,
                kind,
                orientation=final_strategy.get("orientation", "normal"),
            )
            current_shadow.update(
                evaluate_strategy(
                    shadow_eval, kind, final_strategy
                )
            )

        strategy_validated = (
            total_bets >= 100
            and roi is not None
            and roi > 0
            and seasons_with_bets >= 3
            and profitable_seasons
            >= math.ceil(seasons_with_bets * 0.60)
        )
        frozen_confirmation = frozen_confirmation_test(
            dataset,
            base_features,
            kind,
            discovery_end_season=2021,
            confirmation_start_season=2022,
            current_season=current_season,
        )

        output[kind] = {
            "status": "ok",
            "method": "nested_walk_forward_classifier",
            "oof_games": int(len(full_oof)),
            "outer_seasons": outer,
            "bets": int(total_bets),
            "wins": int(total_wins),
            "losses": int(total_losses),
            "win_rate": (
                None if win_rate is None
                else round(float(win_rate), 4)
            ),
            "roi": (
                None if roi is None
                else round(float(roi), 4)
            ),
            "profit_units": round(
                float(total_profit), 3
            ),
            "seasons_with_bets": int(
                seasons_with_bets
            ),
            "profitable_seasons": int(
                profitable_seasons
            ),
            "strategy_validated": bool(
                strategy_validated
                and frozen_confirmation.get("passed", False)
            ),
            "nested_strategy_validated": bool(
                strategy_validated
            ),
            "frozen_confirmation": frozen_confirmation,
            "production_strategy": (
                frozen_confirmation.get("frozen_strategy")
                if frozen_confirmation.get("passed", False)
                else {
                    "orientation": "normal",
                    "side_mode": "none",
                    "segment": "all",
                    "threshold": 1.0,
                    "reason": "Frozen confirmation did not pass.",
                }
            ),
            "current_season_shadow": current_shadow,
            "final_strategy": final_strategy,
            "thresholds": fixed_threshold_diagnostics(
                full_oof, kind
            ),
            "validated_threshold": (
                frozen_confirmation.get("frozen_strategy", {}).get("threshold")
                if strategy_validated
                and frozen_confirmation.get("passed", False)
                and frozen_confirmation.get("frozen_strategy", {}).get("side_mode")
                != "none"
                else None
            ),
        }

    return output


def fit_final_market_model(
    dataset: pd.DataFrame,
    base_features: list[str],
    kind: str,
) -> MarketClassifier:
    rows = market_rows(dataset, kind)
    return MarketClassifier().fit(
        engineered_features(rows, base_features),
        rows["market_target"],
        sample_weight=training_weights(rows),
    )


def predict_current_market(
    model: MarketClassifier,
    games: pd.DataFrame,
    base_features: list[str],
    kind: str,
    validation: dict,
) -> pd.DataFrame:
    out = games.copy()
    raw_probability = model.predict_proba(
        engineered_features(out, base_features)
    )

    strategy = (
        validation.get("production_strategy")
        or validation.get("final_strategy")
        or {
            "orientation": "normal",
            "side_mode": "none",
            "segment": "all",
            "threshold": 1.0,
        }
    )

    annotated = annotate_probabilities(
        out,
        raw_probability,
        kind,
        orientation=strategy.get(
            "orientation", "normal"
        ),
    )

    side_mode = strategy.get(
        "side_mode", "none"
    )
    threshold = float(
        strategy.get("threshold", 1.0)
    )

    qualified = numeric(
        annotated["market_probability_edge"]
    ).ge(threshold)

    if side_mode == "positive":
        qualified &= numeric(
            annotated["market_side"]
        ).eq(1)
    elif side_mode == "negative":
        qualified &= numeric(
            annotated["market_side"]
        ).eq(-1)
    elif side_mode == "none":
        qualified &= False

    line_column = (
        "spread_line" if kind == "spread"
        else "total_line"
    )
    has_line = numeric(
        annotated[line_column]
    ).notna()
    qualified &= has_line

    validated = bool(
        validation.get("strategy_validated")
    )
    status = np.where(
        ~has_line,
        "NO LINE",
        np.where(
            qualified & validated,
            "VALIDATED",
            np.where(
                qualified, "WATCH", "NO BET"
            ),
        ),
    )

    side = numeric(
        annotated["market_side"]
    ).fillna(0).astype(int)

    if kind == "spread":
        pick = np.where(
            side.eq(1),
            annotated["home_team"],
            annotated["away_team"],
        )
        prefix = "spread"
    else:
        pick = np.where(
            side.eq(1), "OVER", "UNDER"
        )
        prefix = "total"

    annotated[f"{prefix}_pick"] = pick
    annotated[f"{prefix}_probability"] = numeric(
        annotated["market_side_probability"]
    )
    annotated[f"{prefix}_probability_edge"] = numeric(
        annotated["market_probability_edge"]
    )
    annotated[f"{prefix}_market_price"] = numeric(
        annotated["market_price"]
    )
    annotated[f"{prefix}_status"] = status

    payouts = np.array([
        american_profit(value)
        for value in numeric(
            annotated["market_price"]
        )
    ])
    side_probability = numeric(
        annotated["market_side_probability"]
    ).to_numpy(dtype=float)
    annotated[f"{prefix}_expected_value"] = (
        side_probability * payouts
        - (1.0 - side_probability)
    )

    return annotated
