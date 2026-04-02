from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# ── Position schemas ──────────────────────────────────────────────────────────

class PositionCreate(BaseModel):
    asset: str
    ticker: str
    direction: str  # long | short
    entry_price: float
    entry_date: str
    quantity: float
    cost_basis_usd: float
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None
    trailing_stop: bool = False
    source_signal: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    notes: Optional[str] = None


class PositionUpdate(BaseModel):
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None
    trailing_stop: Optional[bool] = None
    notes: Optional[str] = None
    quantity: Optional[float] = None


class PositionClose(BaseModel):
    exit_price: float
    exit_date: str
    exit_reason: str = "manual"
    notes: Optional[str] = None


class PositionResponse(BaseModel):
    id: int
    asset: str
    ticker: str
    direction: str
    entry_price: float
    entry_date: str
    quantity: float
    cost_basis_usd: float
    stop_loss: Optional[float]
    profit_target: Optional[float]
    trailing_stop: bool
    source_signal: Optional[str]
    exchange: Optional[str]
    symbol: Optional[str]
    notes: Optional[str]
    status: str
    created_at: str
    updated_at: str
    # computed fields (not in DB)
    current_price: Optional[float] = None
    unrealized_pnl_usd: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    distance_to_stop_pct: Optional[float] = None
    distance_to_target_pct: Optional[float] = None
    position_status: Optional[str] = None  # HOLD/ADD/REDUCE/EXIT/STOP_TRIGGERED/TARGET_REACHED/SIGNAL_REVERSED
    latest_signal_action: Optional[str] = None
    latest_signal_bias: Optional[float] = None

    class Config:
        from_attributes = True


# ── Trade schemas ─────────────────────────────────────────────────────────────

class TradeCreate(BaseModel):
    asset: str
    ticker: str
    direction: str
    entry_price: float
    entry_date: str
    exit_price: float
    exit_date: str
    quantity: float
    cost_basis_usd: float
    exit_reason: Optional[str] = "manual"
    notes: Optional[str] = None


class TradeResponse(BaseModel):
    id: int
    position_id: Optional[int]
    asset: str
    ticker: str
    direction: str
    entry_price: float
    entry_date: str
    exit_price: float
    exit_date: str
    quantity: float
    cost_basis_usd: float
    realized_pnl_usd: Optional[float]
    realized_pnl_pct: Optional[float]
    exit_reason: Optional[str]
    holding_days: Optional[int]
    notes: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class TradeStats(BaseModel):
    total: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    total_realized_pnl: float


# ── Order schemas ─────────────────────────────────────────────────────────────

class OrderResponse(BaseModel):
    id: int
    position_id: Optional[int]
    asset: str
    action: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float]
    status: str
    note: Optional[str]
    generated_at: str
    executed_at: Optional[str]

    class Config:
        from_attributes = True


# ── Signal schemas ────────────────────────────────────────────────────────────

class SignalSummary(BaseModel):
    asset: str
    action: str
    bias_score: Optional[float]
    regime: Optional[str]
    entry_zone: Optional[str]
    stop_loss: Optional[float]
    profit_target: Optional[float]
    risk_reward_ratio: Optional[float]
    market_sentiment: Optional[str]
    analysis_date: Optional[str]


# ── Dashboard schemas ─────────────────────────────────────────────────────────

class Alert(BaseModel):
    asset: str
    status: str
    message: str
    severity: str  # error | warning | success


class DashboardSummary(BaseModel):
    portfolio_value: float
    total_unrealized_pnl_usd: float
    total_unrealized_pnl_pct: float
    active_signals: int
    total_assets: int = 12
    needs_action: int
    open_positions: int
    alerts: List[Alert]


class MacroData(BaseModel):
    sentiment: str
    vix: Optional[float] = None
    dxy: Optional[float] = None
    ten_year: Optional[float] = None
    btc_fear_greed: Optional[int] = None
    btc_fear_greed_label: Optional[str] = None
    last_scan_date: Optional[str] = None
    sector_ranking: list = []
    top_opportunities: list = []
