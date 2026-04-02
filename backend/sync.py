import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_JSON = os.path.join(PROJECT_ROOT, "portfolio.json")


def sync_to_json(db):
    """Sync open positions from SQLite → portfolio.json for backward compatibility."""
    from models import Position

    positions = db.query(Position).filter(Position.status == "open").all()
    data = {
        "_comment": "Auto-synced from trading.db by FastAPI backend",
        "_last_sync": datetime.now().isoformat(),
        "positions": [
            {
                "asset": p.asset,
                "type": p.direction,
                "entry_price": p.entry_price,
                "entry_date": p.entry_date,
                "quantity": p.quantity,
                "cost_basis_usd": p.cost_basis_usd,
                "stop_loss": p.stop_loss,
                "profit_target": p.profit_target,
                "trailing_stop": p.trailing_stop,
                "source_signal": p.source_signal,
                "exchange": p.exchange,
                "symbol": p.symbol,
                "notes": p.notes,
            }
            for p in positions
        ],
    }
    with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
