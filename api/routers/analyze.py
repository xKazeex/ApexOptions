"""API router for ticker analysis — orchestrates the full engine pipeline."""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import json
import logging

from engine import run_analysis
from api.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ApexStrategyReport,
    ReportMetadata,
    ExecutionParameters,
    RiskManagement,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticker(request: AnalysisRequest):
    """Run the full Apex analytics pipeline on a ticker and return a structured report."""
    ticker = request.ticker.upper()
    account_size = request.account_size

    try:
        # Run the complete engine pipeline
        result_json = run_analysis(
            ticker=ticker,
            account_size=account_size,
            output_format="json",
        )
        result = json.loads(result_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Map engine output to API response schema
    vol = result.get("volatility_summary", {})
    strikes = result.get("strikes", {})
    risk = result.get("risk_parameters", {})

    # Determine strikes (handle iron condor with 4 strikes vs spread with 2)
    sell_strikes = strikes.get("short_strikes") or ([strikes.get("short_strike", 0)] if strikes.get("short_strike") else [])
    buy_strikes = strikes.get("long_strikes") or ([strikes.get("long_strike", 0)] if strikes.get("long_strike") else [])

    report = ApexStrategyReport(
        market_thesis=result.get("market_thesis", ""),
        recommended_strategy=result.get("strategy_name", ""),
        target_asset=result.get("ticker", ticker),
        execution_parameters=ExecutionParameters(
            strategy=result.get("strategy_name", ""),
            sell_strikes=[float(s) for s in sell_strikes],
            buy_strikes=[float(s) for s in buy_strikes],
            expiration_dte=strikes.get("dte", 45),
            target_premium=f"${strikes.get('premium_collected', 0):.2f}",
        ),
        risk_management=RiskManagement(
            max_risk=f"${risk.get('max_risk_dollars', 0):.2f}",
            profit_target=f"${risk.get('profit_target_dollars', 0):.2f}",
            stop_loss_trigger=f"${risk.get('stop_loss_dollars', 0):.2f}",
        ),
        caveats=result.get("trade_rationale", ""),
    )

    metadata = ReportMetadata(
        timestamp=datetime.now(),
        ticker=ticker,
        iv_rank=vol.get("iv_rank", 0),
        current_price=result.get("underlying_price", 0),
    )

    return AnalysisResponse(report=report, metadata=metadata)


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """Stub for async status polling."""
    return {"request_id": request_id, "status": "completed"}