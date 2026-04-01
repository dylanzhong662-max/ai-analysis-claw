"""
多资产批量回测调度器
- 使用 DeepSeek 官方 API（deepseek-reasoner）
- 自动控制频率（--rate-limit 参数）
- 回测两个周期：2024（行情一般年）和 2025（牛市年）
- 支持断点续跑，各资产结果保存至独立目录

用法：
  python3 run_all_backtests.py

  # 自定义资产和参数
  python3 run_all_backtests.py \\
      --assets GOLD BTC NVDA MSFT GOOGL \\
      --model deepseek-reasoner \\
      --rate-limit 30 \\
      --step 5

  # 只回测某个周期
  python3 run_all_backtests.py --period 2025

  # 只验证数据（不消耗 API）
  python3 run_all_backtests.py --dry-run

  # 续跑（跳过已完成节点）
  python3 run_all_backtests.py --resume
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 回测计划配置
# ─────────────────────────────────────────────

# 两个回测周期
PERIODS = {
    "2024": {"start": "2024-01-01", "end": "2024-12-31", "label": "2024 (行情一般年)"},
    "2025": {"start": "2025-01-01", "end": "2025-12-31", "label": "2025 (去年/牛市年)"},
}

# 资产路由：asset_name -> (script, extra_args)
ASSET_ROUTES = {
    "GOLD":  ("backtest_engine.py",      []),
    "BTC":   ("btc_backtest_engine.py",  []),
    "NVDA":  ("tech_backtest_engine.py", ["--ticker", "NVDA"]),
    "MSFT":  ("tech_backtest_engine.py", ["--ticker", "MSFT"]),
    "GOOGL": ("tech_backtest_engine.py", ["--ticker", "GOOGL"]),
    "AAPL":  ("tech_backtest_engine.py", ["--ticker", "AAPL"]),
    "META":  ("tech_backtest_engine.py", ["--ticker", "META"]),
    "AMZN":  ("tech_backtest_engine.py", ["--ticker", "AMZN"]),
}

DEFAULT_ASSETS = ["GOLD", "BTC", "NVDA", "MSFT", "GOOGL"]


def _get_output_dirs(asset: str) -> list[Path]:
    """返回 asset 的结果目录列表。"""
    if asset == "GOLD":
        return [Path("backtest_results")]
    if asset == "BTC":
        return [Path("btc_backtest_results")]
    return [Path(f"{asset.lower()}_backtest_results")]


def _count_existing(asset: str) -> int:
    """统计已完成的回测节点数。"""
    total = 0
    for d in _get_output_dirs(asset):
        sig_file = d / "signals.csv"
        if sig_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(sig_file)
                total += len(df)
            except Exception:
                pass
    return total


def run_task(script: str, extra_args: list, start: str, end: str,
             model: str, step: int, rate_limit: int,
             dry_run: bool, resume: bool, env: dict) -> int:
    """执行单个回测任务，返回 returncode。"""
    cmd = [
        sys.executable, "-u", script,
        "--start", start,
        "--end", end,
        "--step", str(step),
        "--model", model,
        "--rate-limit", str(rate_limit),
    ] + extra_args

    if dry_run:
        cmd.append("--dry-run")
    if resume:
        cmd.append("--resume")

    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def print_summary(results: list[dict]):
    """打印最终汇总。"""
    print("\n" + "=" * 70)
    print("批量回测完成汇总")
    print("=" * 70)
    print(f"{'资产':<8}  {'周期':<6}  {'状态':<10}  {'已有记录':>8}  {'脚本'}")
    print("-" * 70)
    for r in results:
        status = "✓ 成功" if r["returncode"] == 0 else f"✗ 失败(rc={r['returncode']})"
        existing = _count_existing(r["asset"])
        print(f"  {r['asset']:<6}  {r['period']:<6}  {status:<10}  {existing:>8}  {r['script']}")
    print("=" * 70)

    failed = [r for r in results if r["returncode"] != 0]
    if failed:
        print(f"\n⚠️  {len(failed)} 个任务失败：{[r['asset']+'/'+r['period'] for r in failed]}")
        print("可使用 --resume 重新运行，已完成的节点会自动跳过。")
    else:
        print(f"\n✓ 全部 {len(results)} 个任务完成！")


def estimate_runtime(assets: list, periods: list, step: int, rate_limit: int) -> str:
    """估算运行时间。"""
    trading_days_per_year = 252
    calls_per_run = trading_days_per_year // step
    total_calls = len(assets) * len(periods) * calls_per_run
    total_secs  = total_calls * rate_limit
    hours, rem  = divmod(total_secs, 3600)
    mins        = rem // 60
    return f"~{total_calls} 次 API 调用，预计 {hours}小时{mins}分钟（rate_limit={rate_limit}s/次）"


def main():
    parser = argparse.ArgumentParser(
        description="多资产批量回测调度器（DeepSeek API）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境变量：
  DEEPSEEK_API_KEY  DeepSeek API Key

使用示例：
  # 完整回测（2024+2025，5个资产）
  python3 run_all_backtests.py

  # 只回测 2025 年，NVDA + MSFT
  python3 run_all_backtests.py --assets NVDA MSFT --period 2025

  # 快速验证（不调用 API）
  python3 run_all_backtests.py --dry-run

  # 续跑
  python3 run_all_backtests.py --resume
        """
    )
    parser.add_argument("--assets",     nargs="+", default=DEFAULT_ASSETS,
                        help=f"要回测的资产（默认: {' '.join(DEFAULT_ASSETS)}）\n可选: {' '.join(ASSET_ROUTES.keys())}")
    parser.add_argument("--period",     choices=["2024", "2025", "both"], default="both",
                        help="回测周期（默认 both = 2024+2025）")
    parser.add_argument("--model",      default="deepseek-reasoner",
                        help="模型 ID（默认 deepseek-reasoner）")
    parser.add_argument("--step",       default=5, type=int,
                        help="每隔N个交易日触发一次（默认5，一年约50次API调用）")
    parser.add_argument("--rate-limit", default=30, type=int,
                        help="API 调用间隔秒数（默认30，约2 RPM，安全控频）")
    parser.add_argument("--dry-run",    action="store_true",
                        help="只验证数据和 Prompt，不调用 API")
    parser.add_argument("--resume",     action="store_true",
                        help="跳过已完成节点，断点续跑")
    parser.add_argument("--asset-delay", default=60, type=int,
                        help="不同资产/周期之间的等待秒数（默认60s，让 API 冷却）")
    args = parser.parse_args()

    # 验证资产
    invalid = [a for a in args.assets if a.upper() not in ASSET_ROUTES]
    if invalid:
        print(f"[错误] 不支持的资产: {invalid}")
        print(f"支持的资产: {list(ASSET_ROUTES.keys())}")
        sys.exit(1)
    assets = [a.upper() for a in args.assets]

    # 确定周期
    if args.period == "both":
        periods = ["2024", "2025"]
    else:
        periods = [args.period]

    # API Key
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "sk-9574b3366dfd41178a5493d0f6af33c0")

    # 构建子进程环境变量
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = deepseek_key

    # 打印运行计划
    print("=" * 70)
    print("多资产批量回测调度器")
    print("=" * 70)
    print(f"资产列表:  {assets}")
    print(f"回测周期:  {[PERIODS[p]['label'] for p in periods]}")
    print(f"模型:      {args.model}")
    print(f"API端点:   https://api.deepseek.com/v1")
    print(f"步长:      每 {args.step} 个交易日一次")
    print(f"控频:      {args.rate_limit}s / API 调用")
    print(f"资产延迟:  {args.asset_delay}s")
    print(f"断点续跑:  {args.resume}")
    print(f"预计运行:  {estimate_runtime(assets, periods, args.step, args.rate_limit)}")
    print("=" * 70)
    print()

    if not args.dry_run:
        print("启动倒计时 5 秒（Ctrl+C 取消）...")
        try:
            for i in range(5, 0, -1):
                print(f"  {i}...", end="", flush=True)
                time.sleep(1)
            print(" 开始！\n")
        except KeyboardInterrupt:
            print("\n已取消。")
            sys.exit(0)

    results = []
    total = len(assets) * len(periods)
    task_num = 0

    for period_key in periods:
        period_info = PERIODS[period_key]
        start_date  = period_info["start"]
        end_date    = period_info["end"]

        for asset in assets:
            task_num += 1
            script, extra_args = ASSET_ROUTES[asset]

            print(f"\n{'─'*70}")
            print(f"[{task_num}/{total}] {asset} | {period_info['label']}")
            print(f"{'─'*70}")

            t0 = time.time()
            rc = run_task(
                script=script,
                extra_args=extra_args,
                start=start_date,
                end=end_date,
                model=args.model,
                step=args.step,
                rate_limit=args.rate_limit,
                dry_run=args.dry_run,
                resume=args.resume,
                env=env,
            )
            elapsed = time.time() - t0

            results.append({
                "asset":      asset,
                "period":     period_key,
                "script":     script,
                "returncode": rc,
                "elapsed_min": f"{elapsed/60:.1f}m",
            })

            status = "✓" if rc == 0 else f"✗ (rc={rc})"
            print(f"\n  [{status}] {asset}/{period_key} 完成，耗时 {elapsed/60:.1f} 分钟")

            # 任务间延迟（让 API 冷却）
            if task_num < total and not args.dry_run:
                print(f"  等待 {args.asset_delay}s 再开始下一个资产...", end="", flush=True)
                time.sleep(args.asset_delay)
                print(" 继续")

    print_summary(results)

    # 失败时返回非零退出码
    if any(r["returncode"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
