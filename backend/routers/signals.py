import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, BackgroundTasks
import signal_reader

router = APIRouter()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@router.get("")
def get_all_signals():
    signals = signal_reader.read_all_signals()
    result = []
    for asset, sig in signals.items():
        if sig:
            result.append({k: v for k, v in sig.items() if k != "raw"})
        else:
            result.append({
                "asset": asset, "action": None, "bias_score": None,
                "regime": None, "analysis_date": None, "market_sentiment": None,
            })
    return result


@router.get("/{asset}")
def get_signal(asset: str):
    sig = signal_reader.extract_signal_summary(asset.upper())
    if not sig:
        raise HTTPException(status_code=404, detail=f"暂无 {asset} 的分析结果")
    return sig


def _run_script(script: str, ticker: str | None):
    cmd = ["python3", script, "--api"]
    if ticker:
        cmd += ["--ticker", ticker]
    try:
        subprocess.Popen(cmd, cwd=PROJECT_ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Script launch error: {e}")


@router.post("/refresh/{asset}")
def refresh_signal(asset: str, background_tasks: BackgroundTasks):
    asset = asset.upper()
    if asset not in signal_reader.SCRIPT_MAP:
        raise HTTPException(status_code=404, detail=f"未知资产: {asset}")
    script, ticker = signal_reader.SCRIPT_MAP[asset]
    background_tasks.add_task(_run_script, script, ticker)
    return {"status": "started", "message": f"正在分析 {asset}，请 30 秒后刷新"}
