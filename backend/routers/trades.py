import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Optional

from database import get_db
from models import Trade
from schemas import TradeCreate, TradeResponse, TradeStats

router = APIRouter()


@router.get("/stats", response_model=TradeStats)
def get_stats(db: Session = Depends(get_db)):
    trades = db.query(Trade).all()
    if not trades:
        return TradeStats(total=0, wins=0, losses=0, win_rate=0.0,
                          profit_factor=0.0, avg_win_pct=0.0,
                          avg_loss_pct=0.0, total_realized_pnl=0.0)
    wins = [t for t in trades if (t.realized_pnl_pct or 0) > 0]
    losses = [t for t in trades if (t.realized_pnl_pct or 0) <= 0]
    gross_profit = sum(t.realized_pnl_usd for t in wins if t.realized_pnl_usd) or 0
    gross_loss = abs(sum(t.realized_pnl_usd for t in losses if t.realized_pnl_usd) or 0)
    return TradeStats(
        total=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        avg_win_pct=round(sum(t.realized_pnl_pct for t in wins if t.realized_pnl_pct) / len(wins), 2) if wins else 0,
        avg_loss_pct=round(sum(t.realized_pnl_pct for t in losses if t.realized_pnl_pct) / len(losses), 2) if losses else 0,
        total_realized_pnl=round(sum(t.realized_pnl_usd for t in trades if t.realized_pnl_usd), 2),
    )


@router.get("", response_model=dict)
def list_trades(
    db: Session = Depends(get_db),
    asset: Optional[str] = None,
    direction: Optional[str] = None,
    exit_reason: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    q = db.query(Trade)
    if asset:
        q = q.filter(Trade.asset == asset)
    if direction:
        q = q.filter(Trade.direction == direction)
    if exit_reason:
        q = q.filter(Trade.exit_reason == exit_reason)
    if start_date:
        q = q.filter(Trade.exit_date >= start_date)
    if end_date:
        q = q.filter(Trade.exit_date <= end_date)
    total = q.count()
    trades = q.order_by(Trade.exit_date.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "trades": [TradeResponse.model_validate(t) for t in trades],
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
    }


@router.post("", response_model=TradeResponse)
def create_trade(body: TradeCreate, db: Session = Depends(get_db)):
    if body.direction == "long":
        pnl_usd = (body.exit_price - body.entry_price) * body.quantity
        pnl_pct = (body.exit_price - body.entry_price) / body.entry_price * 100
    else:
        pnl_usd = (body.entry_price - body.exit_price) * body.quantity
        pnl_pct = (body.entry_price - body.exit_price) / body.entry_price * 100

    try:
        d1 = date.fromisoformat(body.entry_date[:10])
        d2 = date.fromisoformat(body.exit_date[:10])
        holding_days = (d2 - d1).days
    except Exception:
        holding_days = None

    trade = Trade(
        asset=body.asset, ticker=body.ticker, direction=body.direction,
        entry_price=body.entry_price, entry_date=body.entry_date,
        exit_price=body.exit_price, exit_date=body.exit_date,
        quantity=body.quantity, cost_basis_usd=body.cost_basis_usd,
        realized_pnl_usd=round(pnl_usd, 2),
        realized_pnl_pct=round(pnl_pct, 2),
        exit_reason=body.exit_reason,
        holding_days=holding_days,
        notes=body.notes,
        created_at=datetime.now().isoformat(),
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return TradeResponse.model_validate(trade)


@router.delete("/{trade_id}")
def delete_trade(trade_id: int, db: Session = Depends(get_db)):
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    db.delete(trade)
    db.commit()
    return {"success": True}
