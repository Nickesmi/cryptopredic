import logging
from app.core.celery_app import celery_app
from app.services.crypto_api import crypto_fetcher
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)

@celery_app.task(name="app.worker.background_evaluate_gems")
def background_evaluate_gems():
    """
    Periodic task triggered by Celery Beat to fetch new trending coins
    and re-run them through our ML anomaly detector. The results
    would typically be pushed to the DB or Redis Cache here.
    """
    logger.info("Starting background evaluation of high potential coins...")
    try:
        trending_raw = crypto_fetcher.get_coingecko_trending()
        if not isinstance(trending_raw, list) or len(trending_raw) == 0:
            logger.warning("No data retrieved from CoinGecko in background task.")
            return {"status": "skipped", "reason": "no data"}
            
        scored_coins = sentiment_service.calculate_spike_probability(trending_raw)
        
        # Here we would save scored_coins to our TimescaleDB or PostgreSQL
        # ... e.g., db.session.add(models.PredictionRecord(...))
        
        logger.info(f"Successfully evaluated and scored {len(scored_coins)} coins in the background.")
        
        # Return only the top symbol as a log/result
        top_pick = scored_coins[0] if scored_coins else None
        return {"status": "success", "top_pick_detected": top_pick.get("symbol") if top_pick else None}
        
    except Exception as e:
        logger.error(f"Error in background background_evaluate_gems task: {e}")
        return {"status": "error", "error": str(e)}
