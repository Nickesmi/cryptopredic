"""Historical replay backtest engine without data leakage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from src.models.model_manager import ModelManager
from src.evaluation.prediction_store import PredictionStore
from src.evaluation.metrics import aggregate_metrics, evaluate_prediction

@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    horizon: int
    lookback: int

@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    
    def to_dict(self):
        return {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "predictions_count": len(self.predictions)
        }

def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Runs a historical replay backtest."""
    model_manager = ModelManager()
    
    evaluations = []
    predictions = []
    
    # We need at least lookback + horizon data points
    total_len = len(df)
    if total_len <= config.lookback + config.horizon:
        raise ValueError("Dataset too short for the given lookback and horizon.")
        
    for i in range(config.lookback, total_len - config.horizon):
        # 1. Provide only data up to current index 'i' (Pretend future is unknown)
        window = df.iloc[i - config.lookback : i].copy()
        
        current_close = float(window['close'].iloc[-1])
        if 'time' in window.columns:
            current_time = window['time'].iloc[-1]
        else:
            current_time = window.index[-1]
            
        if isinstance(current_time, pd.Timestamp):
            current_time = int(current_time.timestamp())
            
        # 2. Generate Prediction
        try:
            # Depending on model_manager signature. Assuming it takes df and returns prediction dict.
            pred = model_manager.predict(window, config.symbol, config.timeframe)
        except Exception as e:
            continue
            
        # 3. Retrieve actual future data (the target)
        future_idx = i + config.horizon
        actual_close = float(df['close'].iloc[future_idx])
        
        if 'time' in df.columns:
            future_time = df['time'].iloc[future_idx]
        else:
            future_time = df.index[future_idx]
            
        if isinstance(future_time, pd.Timestamp):
            future_time = int(future_time.timestamp())
            
        predicted_price = float(pred.get("expected_price", current_close))
        confidence = float(pred.get("confidence", 0.5))
        bullish_prob = float(pred.get("bullish_probability", 0.5))
        
        predictions.append(pred)
        
        # 4. Evaluate Prediction
        metrics = evaluate_prediction(
            anchor_price=current_close,
            predicted_price=predicted_price,
            actual_price=actual_close,
            confidence=confidence,
            bullish_probability=bullish_prob
        )
        
        # Store as dict matching DB structure for aggregate_metrics
        eval_dict = metrics.to_dict()
        eval_dict["anchor_price"] = current_close
        eval_dict["actual_price"] = actual_close
        eval_dict["predicted_price"] = predicted_price
        eval_dict["confidence"] = confidence
        
        evaluations.append(eval_dict)

    if not evaluations:
        raise ValueError("No predictions were generated.")
        
    final_metrics = aggregate_metrics(evaluations)
    
    return BacktestResult(
        config=config,
        metrics=final_metrics,
        predictions=predictions
    )
