from dataclasses import dataclass, asdict
from typing import Any, List, Dict
import math

@dataclass
class EvaluationMetrics:
    direction_correct: bool
    price_error: float
    absolute_error: float
    percentage_error: float
    rmse: float
    mae: float
    mape: float
    r2: float
    directional_accuracy: float
    hit_rate: float
    precision: float
    recall: float
    f1_score: float
    maximum_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    confidence_calibration: float
    brier_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def evaluate_prediction(
    anchor_price: float,
    predicted_price: float,
    actual_price: float,
    confidence: float,
    bullish_probability: float
) -> EvaluationMetrics:
    """Evaluates a single prediction."""
    predicted_return = (predicted_price - anchor_price) / anchor_price if anchor_price else 0.0
    actual_return = (actual_price - anchor_price) / anchor_price if anchor_price else 0.0
    
    direction_correct = (predicted_return > 0 and actual_return > 0) or (predicted_return < 0 and actual_return < 0)
    
    price_error = predicted_price - actual_price
    absolute_error = abs(price_error)
    percentage_error = (absolute_error / actual_price * 100) if actual_price else 0.0
    
    rmse = absolute_error  # For a single prediction, RMSE is just absolute error
    mae = absolute_error
    mape = percentage_error
    r2 = 0.0 # R2 undefined for single prediction
    
    acc = 1.0 if direction_correct else 0.0
    
    # Brier score
    outcome = 1.0 if actual_return > 0 else 0.0
    brier_score = (bullish_probability - outcome) ** 2
    
    # Simple calibration for a single prediction
    confidence_calibration = 1.0 - abs(confidence - acc)
    
    return EvaluationMetrics(
        direction_correct=direction_correct,
        price_error=price_error,
        absolute_error=absolute_error,
        percentage_error=percentage_error,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        directional_accuracy=acc,
        hit_rate=acc,
        precision=acc,
        recall=acc,
        f1_score=acc,
        maximum_drawdown=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        confidence_calibration=confidence_calibration,
        brier_score=brier_score
    )

def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates metrics over a list of evaluated prediction rows."""
    if not rows:
        return {
            "predictions": 0,
            "directional_accuracy": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "sharpe_ratio": 0.0,
            "confidence_calibration": 0.0
        }
        
    correct = sum(1 for r in rows if r.get("direction_correct"))
    mae_sum = sum(float(r.get("absolute_error", 0)) for r in rows)
    mse_sum = sum(float(r.get("price_error", 0))**2 for r in rows)
    mape_sum = sum(float(r.get("percentage_error", 0)) for r in rows)
    
    predictions = len(rows)
    acc = correct / predictions
    mae = mae_sum / predictions
    rmse = math.sqrt(mse_sum / predictions)
    mape = mape_sum / predictions
    
    # Approximation of confidence calibration (mean confidence vs accuracy)
    mean_conf = sum(float(r.get("confidence", 0.5)) for r in rows) / predictions
    conf_cal = 1.0 - abs(mean_conf - acc)
    
    # Simulated sharpe
    returns = []
    for r in rows:
        anchor = float(r.get("anchor_price", 1.0))
        actual = float(r.get("actual_price", 1.0))
        direction = 1 if float(r.get("predicted_price", 1.0)) > anchor else -1
        trade_ret = direction * (actual - anchor) / anchor
        returns.append(trade_ret)
        
    if len(returns) > 1:
        mean_ret = sum(returns) / len(returns)
        var_ret = sum((rt - mean_ret)**2 for rt in returns) / (len(returns) - 1)
        std_ret = math.sqrt(var_ret)
        sharpe = (mean_ret / std_ret * math.sqrt(365)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "predictions": predictions,
        "directional_accuracy": acc,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "sharpe_ratio": sharpe,
        "confidence_calibration": conf_cal
    }
