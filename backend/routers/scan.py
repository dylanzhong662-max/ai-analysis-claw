import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, BackgroundTasks
import signal_reader

router = APIRouter()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_scan(group: str):
    try:
        subprocess.Popen(
            ["python3", "market_scan.py", "--group", group, "--api"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Scan error: {e}")


@router.get("/latest")
def get_latest_scan():
    data = signal_reader.read_market_scan()
    if not data:
        return {"message": "暂无扫描结果，请先运行扫描"}
    return data


@router.post("/run")
def run_market_scan(background_tasks: BackgroundTasks, group: str = "quick"):
    background_tasks.add_task(_run_scan, group)
    return {"status": "started", "message": f"正在扫描 [{group}] 分组，完成后刷新查看结果"}
