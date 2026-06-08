"""Pydantic schemas for the Apex Options Analytics API."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ExecutionParameters(BaseModel):
    strategy: str
    sell_strikes: List[float]
    buy_strikes: List[float]
    expiration_dte: int
    target_premium: str
    limit_price: Optional[str] = None
    bid_ask_range: Optional[str] = None


class RiskManagement(BaseModel):
    max_risk: str
    profit_target: str
    stop_loss_trigger: str
    contracts: Optional[int] = None
    risk_pct: Optional[str] = None


class ApexStrategyReport(BaseModel):
    market_thesis: str
    recommended_strategy: str
    target_asset: str
    execution_parameters: ExecutionParameters
    risk_management: RiskManagement
    caveats: str


class ReportMetadata(BaseModel):
    timestamp: datetime
    ticker: str
    iv_rank: float
    current_price: float


class AnalysisResponse(BaseModel):
    report: ApexStrategyReport
    metadata: ReportMetadata


class AnalysisRequest(BaseModel):
    ticker: str
    account_size: float = Field(default=50000.0, description="Account size in dollars")