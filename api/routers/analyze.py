"""API router for ticker analysis — orchestrates the full engine pipeline."""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
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

    # Map engine output to API response
    vol = result.get("volatility_summary", {})
    strikes = result.get("strikes", {})
    risk = result.get("risk_parameters", {})
    sizing = result.get("position_sizing", {})

    # Build strike lists from engine's singular keys
    sell_strikes = []
    if strikes.get("short_strike"):
        sell_strikes.append(round(strikes["short_strike"], 2))
    if strikes.get("short_strike_2") and float(strikes["short_strike_2"]) > 0:
        ss2 = float(strikes["short_strike_2"])
        if ss2 not in sell_strikes:
            sell_strikes.append(round(ss2, 2))

    buy_strikes = []
    if strikes.get("long_strike"):
        buy_strikes.append(round(strikes["long_strike"], 2))
    if strikes.get("long_strike_2") and float(strikes["long_strike_2"]) > 0:
        ls2 = float(strikes["long_strike_2"])
        if ls2 not in buy_strikes:
            buy_strikes.append(round(ls2, 2))

    # Compute actual expiration date from DTE
    dte = int(strikes.get("dte", 45))
    expiration_date = (datetime.now() + timedelta(days=dte)).strftime("%Y-%m-%d")

    # Use real premium from engine, with sensible fallback
    premium_collected = float(strikes.get("premium_collected", 0) or 0)
    spread_width = float(strikes.get("spread_width", 0) or 0)
    underlying_price = float(result.get("underlying_price", 0) or 0)

    if premium_collected > 0:
        limit_price = round(premium_collected, 2)
        bid_ask_range = f"${limit_price * 0.85:.2f} - ${limit_price * 1.15:.2f}"
    elif spread_width > 0:
        # Estimate premium as ~20% of spread width when missing
        estimated = round(spread_width * 0.20, 2)
        limit_price = estimated
        bid_ask_range = f"${estimated * 0.85:.2f} - ${estimated * 1.15:.2f}"
    else:
        # Last resort: estimate 0.5% of underlying
        estimated = round(underlying_price * 0.005, 2)
        limit_price = max(estimated, 0.25)
        bid_ask_range = f"${limit_price * 0.85:.2f} - ${limit_price * 1.15:.2f}"

    # Format caveats from trade rationale
    caveats = result.get("trade_rationale", "")

    report = ApexStrategyReport(
        market_thesis=result.get("market_thesis", "No thesis generated."),
        recommended_strategy=result.get("strategy_name", "N/A"),
        target_asset=result.get("ticker", ticker),
        execution_parameters=ExecutionParameters(
            strategy=result.get("strategy_name", "N/A"),
            sell_strikes=sell_strikes,
            buy_strikes=buy_strikes,
            expiration_dte=dte,
            target_premium=f"${premium_collected:.2f}" if premium_collected > 0 else f"${limit_price:.2f}",
            limit_price=f"${limit_price:.2f}",
            bid_ask_range=bid_ask_range,
        ),
        risk_management=RiskManagement(
            max_risk=f"${risk.get('max_risk_dollars', 0):.2f}",
            profit_target=f"${risk.get('profit_target_dollars', 0):.2f}",
            stop_loss_trigger=f"${risk.get('stop_loss_dollars', 0):.2f}",
            contracts=int(sizing.get("contracts", 1)),
            risk_pct=f"{risk.get('max_risk_pct', 0) * 100:.1f}%",
        ),
        caveats=caveats,
    )

    metadata = ReportMetadata(
        timestamp=datetime.now(),
        ticker=ticker,
        iv_rank=round(float(vol.get("iv_rank", 0) or 0), 1),
        current_price=round(underlying_price, 2),
    )

    return AnalysisResponse(report=report, metadata=metadata)


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """Stub for async status polling."""
    return {"request_id": request_id, "status": "completed"}