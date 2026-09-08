"""Immutable prediction ledger and evaluation persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import PREDICTION_DB_PATH
from src.evaluation.metrics import aggregate_metrics, evaluate_prediction


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    timestamp: str
    expires_at: int
    model_name: str
    model_version: str
    symbol: str
    timeframe: str
    prediction_horizon: int
    anchor_time: int
    anchor_price: float
    predicted_price: float
    confidence: float
    bullish_probability: float
    bearish_probability: float
    risk_level: str
    expected_volatility: float
    support_levels: list[float]
    resistance_levels: list[float]
    reasoning_summary: str
    historical_candles: list[dict[str, Any]]
    technical_indicators: dict[str, Any]
    prediction_values: dict[str, Any]
    feature_importance: dict[str, float]
    market_regime: str


class PredictionStore:
    """SQLite-backed append-only store for forecasts and evaluations."""

    def __init__(self, db_path: str | Path = PREDICTION_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_prediction(self, **kwargs: Any) -> PredictionRecord:
        prediction_id = kwargs.get("prediction_id") or str(uuid.uuid4())
        timestamp = kwargs.get("timestamp") or datetime.now(timezone.utc).isoformat()
        record = PredictionRecord(
            prediction_id=prediction_id,
            timestamp=timestamp,
            expires_at=int(kwargs["expires_at"]),
            model_name=str(kwargs["model_name"]),
            model_version=str(kwargs.get("model_version", "1.0.0")),
            symbol=str(kwargs["symbol"]).upper(),
            timeframe=str(kwargs["timeframe"]),
            prediction_horizon=int(kwargs["prediction_horizon"]),
            anchor_time=int(kwargs["anchor_time"]),
            anchor_price=float(kwargs["anchor_price"]),
            predicted_price=float(kwargs["predicted_price"]),
            confidence=float(kwargs.get("confidence", 0.0)),
            bullish_probability=float(kwargs.get("bullish_probability", 0.5)),
            bearish_probability=float(kwargs.get("bearish_probability", 0.5)),
            risk_level=str(kwargs.get("risk_level", "medium")),
            expected_volatility=float(kwargs.get("expected_volatility", 0.0)),
            support_levels=list(kwargs.get("support_levels", [])),
            resistance_levels=list(kwargs.get("resistance_levels", [])),
            reasoning_summary=str(kwargs.get("reasoning_summary", "")),
            historical_candles=list(kwargs.get("historical_candles", [])),
            technical_indicators=dict(kwargs.get("technical_indicators", {})),
            prediction_values=dict(kwargs.get("prediction_values", {})),
            feature_importance=dict(kwargs.get("feature_importance", {})),
            market_regime=str(kwargs.get("market_regime", "sideways")),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    prediction_id, timestamp, expires_at, model_name, model_version,
                    symbol, timeframe, prediction_horizon, anchor_time, anchor_price,
                    predicted_price, confidence, bullish_probability, bearish_probability,
                    risk_level, expected_volatility, support_levels, resistance_levels,
                    reasoning_summary, historical_candles, technical_indicators,
                    prediction_values, feature_importance, market_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.prediction_id,
                    record.timestamp,
                    record.expires_at,
                    record.model_name,
                    record.model_version,
                    record.symbol,
                    record.timeframe,
                    record.prediction_horizon,
                    record.anchor_time,
                    record.anchor_price,
                    record.predicted_price,
                    record.confidence,
                    record.bullish_probability,
                    record.bearish_probability,
                    record.risk_level,
                    record.expected_volatility,
                    _json(record.support_levels),
                    _json(record.resistance_levels),
                    record.reasoning_summary,
                    _json(record.historical_candles),
                    _json(record.technical_indicators),
                    _json(record.prediction_values),
                    _json(record.feature_importance),
                    record.market_regime,
                ),
            )
        return record

    def evaluate_prediction(
        self,
        prediction_id: str,
        *,
        actual_time: int,
        actual_price: float,
        actual_candles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prediction = self.get_prediction(prediction_id)
        if prediction is None:
            raise KeyError(f"Unknown prediction_id: {prediction_id}")

        metrics = evaluate_prediction(
            anchor_price=float(prediction["anchor_price"]),
            predicted_price=float(prediction["predicted_price"]),
            actual_price=float(actual_price),
            confidence=float(prediction["confidence"]),
            bullish_probability=float(prediction["bullish_probability"]),
        )
        failure_pattern = classify_failure_pattern(prediction, actual_price)
        recommendation = recommendation_for_failure(failure_pattern)
        evaluated_at = datetime.now(timezone.utc).isoformat()
        payload = metrics.to_dict()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_evaluations (
                    prediction_id, evaluated_at, actual_time, actual_price,
                    actual_candles, metrics, direction_correct, price_error,
                    absolute_error, percentage_error, rmse, mae, mape, r2,
                    directional_accuracy, hit_rate, precision, recall, f1_score,
                    maximum_drawdown, profit_factor, sharpe_ratio, sortino_ratio,
                    calmar_ratio, confidence_calibration, brier_score,
                    failure_pattern, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    evaluated_at=excluded.evaluated_at,
                    actual_time=excluded.actual_time,
                    actual_price=excluded.actual_price,
                    actual_candles=excluded.actual_candles,
                    metrics=excluded.metrics,
                    direction_correct=excluded.direction_correct,
                    price_error=excluded.price_error,
                    absolute_error=excluded.absolute_error,
                    percentage_error=excluded.percentage_error,
                    rmse=excluded.rmse,
                    mae=excluded.mae,
                    mape=excluded.mape,
                    r2=excluded.r2,
                    directional_accuracy=excluded.directional_accuracy,
                    hit_rate=excluded.hit_rate,
                    precision=excluded.precision,
                    recall=excluded.recall,
                    f1_score=excluded.f1_score,
                    maximum_drawdown=excluded.maximum_drawdown,
                    profit_factor=excluded.profit_factor,
                    sharpe_ratio=excluded.sharpe_ratio,
                    sortino_ratio=excluded.sortino_ratio,
                    calmar_ratio=excluded.calmar_ratio,
                    confidence_calibration=excluded.confidence_calibration,
                    brier_score=excluded.brier_score,
                    failure_pattern=excluded.failure_pattern,
                    recommendation=excluded.recommendation
                """,
                (
                    prediction_id,
                    evaluated_at,
                    int(actual_time),
                    float(actual_price),
                    _json(actual_candles or []),
                    _json(payload),
                    int(metrics.direction_correct),
                    metrics.price_error,
                    metrics.absolute_error,
                    metrics.percentage_error,
                    metrics.rmse,
                    metrics.mae,
                    metrics.mape,
                    metrics.r2,
                    metrics.directional_accuracy,
                    metrics.hit_rate,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score,
                    metrics.maximum_drawdown,
                    metrics.profit_factor,
                    metrics.sharpe_ratio,
                    metrics.sortino_ratio,
                    metrics.calmar_ratio,
                    metrics.confidence_calibration,
                    metrics.brier_score,
                    failure_pattern,
                    recommendation,
                ),
            )
        result = dict(prediction)
        result.update(payload)
        result.update(
            {
                "actual_time": int(actual_time),
                "actual_price": float(actual_price),
                "failure_pattern": failure_pattern,
                "recommendation": recommendation,
            }
        )
        return result

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return _decode_row(row) if row else None

    def due_predictions(self, now_ts: int | None = None) -> list[dict[str, Any]]:
        now_ts = now_ts or int(datetime.now(timezone.utc).timestamp())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM predictions p
                LEFT JOIN prediction_evaluations e USING (prediction_id)
                WHERE e.prediction_id IS NULL AND p.expires_at <= ?
                ORDER BY p.expires_at ASC
                """,
                (now_ts,),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def evaluated_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT p.*, e.actual_time, e.actual_price, e.direction_correct,
                   e.price_error, e.absolute_error, e.percentage_error,
                   e.rmse, e.mae, e.mape, e.r2, e.directional_accuracy,
                   e.hit_rate, e.precision, e.recall, e.f1_score,
                   e.maximum_drawdown, e.profit_factor, e.sharpe_ratio,
                   e.sortino_ratio, e.calmar_ratio, e.confidence_calibration,
                   e.brier_score, e.failure_pattern, e.recommendation,
                   e.evaluated_at
            FROM prediction_evaluations e
            JOIN predictions p USING (prediction_id)
            ORDER BY p.timestamp DESC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_decode_row(row) for row in rows]

    def leaderboard(self) -> list[dict[str, Any]]:
        rows = self.evaluated_rows()
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((row["model_name"], row["model_version"]), []).append(row)

        board = []
        for (model_name, version), sample in grouped.items():
            metrics = aggregate_metrics(sample)
            last_updated = max(str(row["evaluated_at"]) for row in sample)
            board.append(
                {
                    "model": model_name,
                    "version": version,
                    "predictions": int(metrics["predictions"]),
                    "directional_accuracy": metrics["directional_accuracy"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "sharpe": metrics["sharpe_ratio"],
                    "confidence_calibration": metrics["confidence_calibration"],
                    "last_updated": last_updated,
                }
            )
        return sorted(
            board,
            key=lambda item: (
                item["directional_accuracy"],
                -item["mae"],
                item["sharpe"],
                -item["confidence_calibration"],
            ),
            reverse=True,
        )

    def dashboard(self) -> dict[str, Any]:
        rows = self.evaluated_rows()
        return {
            "overall": aggregate_metrics(rows),
            "last_100": aggregate_metrics(rows[:100]),
            "last_500": aggregate_metrics(rows[:500]),
            "last_1000": aggregate_metrics(rows[:1000]),
            "by_symbol": _group_metrics(rows, "symbol"),
            "by_timeframe": _group_metrics(rows, "timeframe"),
            "by_model": _group_metrics(rows, "model_name"),
            "by_market_regime": _group_metrics(rows, "market_regime"),
            "bull_market": aggregate_metrics(
                [row for row in rows if row["market_regime"] == "bull"]
            ),
            "bear_market": aggregate_metrics(
                [row for row in rows if row["market_regime"] == "bear"]
            ),
            "failure_patterns": recurring_failure_patterns(rows),
            "recommendations": continuous_learning_recommendations(rows),
            "leaderboard": self.leaderboard(),
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    prediction_horizon INTEGER NOT NULL,
                    anchor_time INTEGER NOT NULL,
                    anchor_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    bullish_probability REAL NOT NULL,
                    bearish_probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    expected_volatility REAL NOT NULL,
                    support_levels TEXT NOT NULL,
                    resistance_levels TEXT NOT NULL,
                    reasoning_summary TEXT NOT NULL,
                    historical_candles TEXT NOT NULL,
                    technical_indicators TEXT NOT NULL,
                    prediction_values TEXT NOT NULL,
                    feature_importance TEXT NOT NULL,
                    market_regime TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_evaluations (
                    prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
                    evaluated_at TEXT NOT NULL,
                    actual_time INTEGER NOT NULL,
                    actual_price REAL NOT NULL,
                    actual_candles TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    direction_correct INTEGER NOT NULL,
                    price_error REAL NOT NULL,
                    absolute_error REAL NOT NULL,
                    percentage_error REAL NOT NULL,
                    rmse REAL NOT NULL,
                    mae REAL NOT NULL,
                    mape REAL NOT NULL,
                    r2 REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    hit_rate REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    maximum_drawdown REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    sortino_ratio REAL NOT NULL,
                    calmar_ratio REAL NOT NULL,
                    confidence_calibration REAL NOT NULL,
                    brier_score REAL NOT NULL,
                    failure_pattern TEXT NOT NULL,
                    recommendation TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_due
                ON predictions(expires_at);

                CREATE INDEX IF NOT EXISTS idx_predictions_scoreboard
                ON predictions(model_name, model_version, symbol, timeframe);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def classify_failure_pattern(prediction: dict[str, Any], actual_price: float) -> str:
    anchor = float(prediction["anchor_price"])
    predicted = float(prediction["predicted_price"])
    confidence = float(prediction["confidence"])
    volatility = float(prediction["expected_volatility"])
    predicted_return = (predicted - anchor) / anchor
    actual_return = (actual_price - anchor) / anchor
    if (predicted_return >= 0) != (actual_return >= 0):
        if confidence >= 0.75:
            return "model_overconfidence"
        if abs(actual_return) >= max(0.08, volatility * 3):
            return "volatility_spike"
        return "wrong_trend"
    if abs(actual_return) >= max(0.08, volatility * 3):
        return "late_reversal"
    if abs(predicted_return) > abs(actual_return) * 2 and confidence >= 0.65:
        return "false_breakout"
    if volatility >= 0.08:
        return "liquidity_sweep"
    return "indicator_conflict"


def recommendation_for_failure(pattern: str) -> str:
    return {
        "wrong_trend": "Switch to a different model or ensemble multiple models.",
        "late_reversal": "Decrease prediction horizon and retrain with more recent data.",
        "volatility_spike": "Increase lookback window and add volatility regime features.",
        "false_breakout": "Reduce MACD influence and increase RSI weight.",
        "liquidity_sweep": "Increase liquidity filters and reduce confidence during thin markets.",
        "model_overconfidence": "Recalibrate probabilistic outputs and lower position sizing.",
        "indicator_conflict": "Analyze feature importance and rebalance conflicting indicators.",
    }.get(pattern, "Review failed predictions before retraining.")


def recurring_failure_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        pattern = str(row.get("failure_pattern") or "unknown")
        if not bool(row.get("direction_correct")):
            counts[pattern] = counts.get(pattern, 0) + 1
    return [
        {"pattern": pattern, "count": count, "recommendation": recommendation_for_failure(pattern)}
        for pattern, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def continuous_learning_recommendations(rows: list[dict[str, Any]]) -> list[str]:
    patterns = recurring_failure_patterns(rows)
    recommendations = []
    for item in patterns[:5]:
        rec = item["recommendation"]
        if rec not in recommendations:
            recommendations.append(rec)
    if rows and aggregate_metrics(rows)["confidence_calibration"] > 0.4:
        recommendations.append("Recalibrate confidence scores before retraining.")
    return recommendations


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {group: aggregate_metrics(sample) for group, sample in grouped.items()}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in (
        "support_levels",
        "resistance_levels",
        "historical_candles",
        "technical_indicators",
        "prediction_values",
        "feature_importance",
        "actual_candles",
        "metrics",
    ):
        if key in data and isinstance(data[key], str):
            data[key] = json.loads(data[key])
    if "direction_correct" in data:
        data["direction_correct"] = bool(data["direction_correct"])
    return data
