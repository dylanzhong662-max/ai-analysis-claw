from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

def now_str():
    return datetime.now().isoformat()

class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_date = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    cost_basis_usd = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    profit_target = Column(Float, nullable=True)
    trailing_stop = Column(Boolean, default=False)
    source_signal = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    symbol = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    status = Column(String, default="open")
    created_at = Column(String, default=now_str)
    updated_at = Column(String, default=now_str, onupdate=now_str)
    trades = relationship("Trade", back_populates="position")
    orders = relationship("Order", back_populates="position")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)
    asset = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_date = Column(String, nullable=False)
    exit_price = Column(Float, nullable=False)
    exit_date = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    cost_basis_usd = Column(Float, nullable=False)
    realized_pnl_usd = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)
    holding_days = Column(Integer, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(String, default=now_str)
    position = relationship("Position", back_populates="trades")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)
    asset = Column(String, nullable=False)
    action = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    order_type = Column(String, nullable=False)
    price = Column(Float, nullable=True)
    status = Column(String, default="pending")
    note = Column(String, nullable=True)
    generated_at = Column(String, default=now_str)
    executed_at = Column(String, nullable=True)
    position = relationship("Position", back_populates="orders")

class SignalCache(Base):
    __tablename__ = "signals_cache"
    __table_args__ = (UniqueConstraint("asset", "analysis_date", name="uq_asset_date"),)
    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String, nullable=False)
    analysis_date = Column(String, nullable=False)
    action = Column(String, nullable=True)
    bias_score = Column(Float, nullable=True)
    regime = Column(String, nullable=True)
    entry_zone = Column(String, nullable=True)
    stop_loss = Column(Float, nullable=True)
    profit_target = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    estimated_holding_weeks = Column(Integer, nullable=True)
    market_sentiment = Column(String, nullable=True)
    raw_json = Column(String, nullable=True)
    source_file = Column(String, nullable=True)
    created_at = Column(String, default=now_str)
