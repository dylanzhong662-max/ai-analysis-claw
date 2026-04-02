import json
import os
import re
from datetime import datetime
from typing import Optional, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNAL_FILES = {
    "GOLD": "gold_api_output.txt",
    "SILVER": "slv_api_output.txt",
    "COPPER": "copx_api_output.txt",
    "RARE_EARTH": "remx_api_output.txt",
    "OIL": "uso_api_output.txt",
    "BTC": "btc_api_output.txt",
    "GOOGL": "googl_api_output.txt",
    "MSFT": "msft_api_output.txt",
    "NVDA": "nvda_api_output.txt",
    "AAPL": "aapl_api_output.txt",
    "META": "meta_api_output.txt",
    "AMZN": "amzn_api_output.txt",
}

SCRIPT_MAP = {
    "GOLD":       ("gold_analysis.py", None),
    "BTC":        ("btc_analysis.py", None),
    "SILVER":     ("tech_stock_analysis.py", "SLV"),
    "COPPER":     ("tech_stock_analysis.py", "COPX"),
    "RARE_EARTH": ("tech_stock_analysis.py", "REMX"),
    "OIL":        ("tech_stock_analysis.py", "USO"),
    "GOOGL":      ("tech_stock_analysis.py", "GOOGL"),
    "MSFT":       ("tech_stock_analysis.py", "MSFT"),
    "NVDA":       ("tech_stock_analysis.py", "NVDA"),
    "AAPL":       ("tech_stock_analysis.py", "AAPL"),
    "META":       ("tech_stock_analysis.py", "META"),
    "AMZN":       ("tech_stock_analysis.py", "AMZN"),
}


def parse_json_from_text(text: str) -> Optional[Dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


def get_file_mtime(asset: str) -> Optional[str]:
    filename = SIGNAL_FILES.get(asset, "")
    filepath = os.path.join(PROJECT_ROOT, filename)
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def read_signal(asset: str) -> Optional[Dict]:
    filename = SIGNAL_FILES.get(asset)
    if not filename:
        return None
    filepath = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return parse_json_from_text(f.read())
    except Exception:
        return None


def extract_signal_summary(asset: str) -> Optional[Dict]:
    raw = read_signal(asset)
    if not raw:
        return None
    analyses = raw.get("asset_analysis", [])
    if not analyses:
        return None
    first = analyses[0]
    return {
        "asset": asset,
        "action": first.get("action", "no_trade"),
        "bias_score": first.get("bias_score"),
        "regime": first.get("regime"),
        "entry_zone": first.get("entry_zone"),
        "stop_loss": first.get("stop_loss"),
        "profit_target": first.get("profit_target"),
        "risk_reward_ratio": first.get("risk_reward_ratio"),
        "estimated_holding_weeks": first.get("estimated_holding_weeks"),
        "position_size_pct": first.get("position_size_pct"),
        "invalidation_condition": first.get("invalidation_condition"),
        "justification": first.get("justification"),
        "market_sentiment": raw.get("overall_market_sentiment") or raw.get("macro_environment"),
        "analysis_date": get_file_mtime(asset),
        "raw": raw,
    }


def read_all_signals() -> Dict[str, Optional[Dict]]:
    return {asset: extract_signal_summary(asset) for asset in SIGNAL_FILES}


def read_market_scan() -> Optional[Dict]:
    filepath = os.path.join(PROJECT_ROOT, "market_scan_output.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
