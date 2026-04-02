import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import Position
from schemas import DashboardSummary, MacroData, Alert
import signal_reader
import price_fetcher

router = APIRouter()


@router.get("/summary", response_model=DashboardSummary)
def get_summary(db: Session = Depends(get_db)):
    positions = db.query(Position).filter(Position.status == "open").all()
    alerts = []
    total_pnl = 0.0
    total_cost = 0.0
    needs_action = 0

    for pos in positions:
        current_price = price_fetcher.get_current_price(pos.asset)
        if not current_price:
            continue

        if pos.direction == "long":
            pnl = (current_price - pos.entry_price) * pos.quantity
            at_stop = bool(pos.stop_loss and current_price <= pos.stop_loss)
            at_target = bool(pos.profit_target and current_price >= pos.profit_target)
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity
            at_stop = bool(pos.stop_loss and current_price >= pos.stop_loss)
            at_target = bool(pos.profit_target and current_price <= pos.profit_target)

        total_pnl += pnl
        total_cost += pos.cost_basis_usd

        if at_stop:
            needs_action += 1
            alerts.append(Alert(
                asset=pos.asset, status="STOP_TRIGGERED",
                message=f"{pos.asset} 已触及止损位 {pos.stop_loss}",
                severity="error",
            ))
        elif at_target:
            needs_action += 1
            alerts.append(Alert(
                asset=pos.asset, status="TARGET_REACHED",
                message=f"{pos.asset} 已达到目标价 {pos.profit_target}",
                severity="success",
            ))
        else:
            sig = signal_reader.extract_signal_summary(pos.asset)
            if sig:
                sig_action = sig.get("action")
                if (pos.direction == "long" and sig_action == "short") or \
                   (pos.direction == "short" and sig_action == "long"):
                    needs_action += 1
                    alerts.append(Alert(
                        asset=pos.asset, status="SIGNAL_REVERSED",
                        message=f"{pos.asset} LLM 信号方向已反转，建议平仓",
                        severity="warning",
                    ))

    all_signals = signal_reader.read_all_signals()
    active_signals = sum(
        1 for s in all_signals.values()
        if s and s.get("action") not in (None, "no_trade")
    )

    portfolio_value = total_cost + total_pnl
    pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    return DashboardSummary(
        portfolio_value=round(portfolio_value, 2),
        total_unrealized_pnl_usd=round(total_pnl, 2),
        total_unrealized_pnl_pct=round(pnl_pct, 2),
        active_signals=active_signals,
        open_positions=len(positions),
        needs_action=needs_action,
        alerts=alerts,
    )


@router.get("/macro", response_model=MacroData)
def get_macro():
    macro = price_fetcher.get_macro_prices()
    scan = signal_reader.read_market_scan() or {}

    sentiment = "Neutral"
    for asset in ["GOLD", "GOOGL", "NVDA"]:
        sig = signal_reader.extract_signal_summary(asset)
        if sig and sig.get("market_sentiment"):
            sentiment = sig["market_sentiment"]
            break

    fear_greed = fear_greed_label = None
    btc_raw = signal_reader.read_signal("BTC")
    if btc_raw:
        summary = btc_raw.get("sentiment_summary", {})
        fear_greed = summary.get("fear_greed_index")
        fear_greed_label = summary.get("fear_greed_classification")

    return MacroData(
        sentiment=sentiment,
        vix=macro.get("VIX"),
        dxy=macro.get("DXY"),
        ten_year=macro.get("TNX"),
        btc_fear_greed=fear_greed,
        btc_fear_greed_label=fear_greed_label,
        last_scan_date=scan.get("scan_date"),
        sector_ranking=scan.get("sector_ranking", []),
        top_opportunities=scan.get("top_opportunities", []),
    )
