"""API integration tests using a real ASGI TestClient.

Before this file existed, ``tests/test_api.py`` was 0 bytes — there was
no test anywhere that actually sent an HTTP request through a live route
handler. This is precisely the kind of gap that let a real bug ship
undetected: ``routes_forecast.py::predict()`` referenced
``request.app.state.http_session`` without ``request: Request`` in its
own function signature, so every call to ``GET /api/predict/{symbol}``
raised ``NameError: name 'request' is not defined`` — caught only by
actually starting the server and hitting the endpoint, not by any of the
unit tests exercising ``ModelManager``/``PredictionStore`` directly.
These tests exercise the routes themselves, through the same ASGI
lifespan (startup/shutdown) FastAPI uses in production, so a missing
route parameter, a broken dependency wire-up, or a lifespan/state bug
shows up here instead of only in production.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.evaluation.prediction_store import PredictionStore
from src.models.model_manager import ForecastMetadata, ForecastObject


def _fake_forecast() -> ForecastObject:
    metadata = ForecastMetadata(
        direction="bullish",
        confidence=0.62,
        expected_price=51000.0,
        expected_change_pct=2.0,
        volatility_est=0.02,
        model_name="xgboost",
        generated_at="2024-01-01T00:00:00+00:00",
    )
    return ForecastObject(
        symbol="BTCUSDT",
        timeframe="1H",
        n_candles=20,
        future_prices=[50100.0] * 19 + [51000.0],
        upper_band=[51500.0] * 20,
        lower_band=[49500.0] * 20,
        timestamps=[1_700_000_000 + i * 3600 for i in range(1, 21)],
        metadata=metadata,
        anchor_time=1_700_000_000,
        anchor_price=50000.0,
        target_timestamp=1_700_000_000 + 20 * 3600,
        horizon_seconds=20 * 3600,
        model_version="abc123def456",
        feature_version="feat123",
    )


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_models_endpoint(client: TestClient) -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert "xgboost" in resp.json()["models"]


def test_predict_endpoint_reaches_model_manager_without_a_request_object_bug(
    client: TestClient, tmp_path
) -> None:
    """Regression test: predict() must actually receive the ASGI `request`
    object so `request.app.state.http_session` doesn't raise NameError.
    Mocks ModelManager.predict itself (no real network needed) and asserts
    it was called with a real http_session, then that the route returns a
    well-formed 200 rather than a 500 from a broken route signature.

    Also swaps in a throwaway PredictionStore (tmp_path-backed) for the
    duration of the call -- otherwise a successful prediction here would
    write a real row into the repo's shared data/prediction_evaluation.sqlite3.
    """
    isolated_store = PredictionStore(tmp_path / "test_predictions.sqlite3")
    with (
        patch("src.api.routes_forecast._model_manager.predict", new_callable=AsyncMock) as mock_predict,
        patch("src.api.routes_forecast._prediction_store", isolated_store),
    ):
        mock_predict.return_value = _fake_forecast()
        resp = client.get("/api/predict/BTCUSDT?tf=1H&n=20")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["direction"] == "bullish"
    assert body["prediction_id"] is not None  # PredictionStore.create_prediction ran

    # The bug this test exists to catch: predict() must be called with a
    # real (non-None) http_session pulled from request.app.state, not blow
    # up before ever reaching ModelManager.
    _, kwargs = mock_predict.call_args
    assert kwargs["http_session"] is not None


def test_predict_endpoint_rejects_unsupported_symbol(client: TestClient) -> None:
    resp = client.get("/api/predict/NOTACOIN")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_predict_endpoint_fails_safely_without_network(client: TestClient) -> None:
    """With no mock and no real network access, the underlying Binance
    fetch fails -- the route must return a clean JSON error, never a raw
    traceback or an unhandled exception."""
    resp = client.get("/api/predict/BTCUSDT")
    assert resp.status_code in (400, 500)
    assert "error" in resp.json()


def test_performance_dashboard_on_empty_store(client: TestClient) -> None:
    resp = client.get("/api/performance/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"]["predictions"] == 0
    assert body["calibration"] is None
    assert body["drift"] is None


def test_calibration_endpoint_reports_no_data_honestly(client: TestClient) -> None:
    resp = client.get("/api/performance/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert body["calibration"] is None
    assert "message" in body


def test_drift_endpoint_reports_no_data_honestly(client: TestClient) -> None:
    resp = client.get("/api/performance/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["drift"] is None


def test_rankings_endpoint_returns_no_opportunity_without_network(client: TestClient) -> None:
    """No mock: with no real exchange access, every candidate fetch fails,
    so the scanner must return its honest "no opportunity" message with
    per-symbol fetch errors, not a 500."""
    resp = client.get("/api/rankings/opportunities?symbols=BTCUSDT")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert body["message"]
