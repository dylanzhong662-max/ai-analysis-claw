"""
黄金分析脚本
自动获取黄金价格数据、计算技术指标，生成可粘贴到 Claude.ai 的分析提示词

依赖安装：
    pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import numpy as np
import urllib3
from curl_cffi import requests as curl_requests
from datetime import datetime, timedelta

# 禁用 SSL 警告（企业网络/VPN 自签名证书环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────
# 技术指标计算
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, adjust=False).mean()


def fmt_series(series: pd.Series, decimals: int = 2, n: int = 10) -> str:
    """将 Series 最后 n 个值格式化为字符串列表"""
    values = series.dropna().tail(n).round(decimals).tolist()
    return ", ".join(str(v) for v in values)


# ─────────────────────────────────────────────
# 数据获取与指标计算
# ─────────────────────────────────────────────

def _make_session() -> curl_requests.Session:
    """创建禁用 SSL 验证的 curl_cffi session（yfinance 新版要求），用于企业网络/VPN 环境"""
    return curl_requests.Session(impersonate="chrome", verify=False)


def fetch_gold_data():
    ticker = "GC=F"
    print(f"正在获取黄金期货数据 ({ticker})...")

    session = _make_session()

    # 日线：3个月
    daily = yf.download(
        ticker, period="3mo", interval="1d",
        auto_adjust=True, progress=False, session=session
    )
    # 周线：6个月
    weekly = yf.download(
        ticker, period="6mo", interval="1wk",
        auto_adjust=True, progress=False, session=session
    )

    if daily.empty or weekly.empty:
        raise ValueError("数据获取失败，请检查网络或 ticker 是否正确")

    print(f"日线数据：{len(daily)} 条  |  周线数据：{len(weekly)} 条")
    return daily, weekly


def compute_indicators(df: pd.DataFrame):
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()

    indicators = {}
    indicators['ema20']     = calc_ema(close, 20)
    indicators['ema50']     = calc_ema(close, 50)
    indicators['ema200']    = calc_ema(close, 200)
    macd, signal, hist      = calc_macd(close)
    indicators['macd']      = macd
    indicators['macd_sig']  = signal
    indicators['macd_hist'] = hist
    indicators['rsi14']     = calc_rsi(close, 14)
    indicators['rsi7']      = calc_rsi(close, 7)
    indicators['atr14']     = calc_atr(df, 14)
    indicators['atr3']      = calc_atr(df, 3)
    return indicators


# ─────────────────────────────────────────────
# 周涨跌幅计算
# ─────────────────────────────────────────────

def weekly_change(daily: pd.DataFrame):
    close = daily['Close'].squeeze()
    last_close  = close.iloc[-1]
    prev_close  = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]

    # 过去5个交易日（约一周）
    close_5d_ago = close.iloc[-6] if len(close) >= 6 else close.iloc[0]

    day_chg     = (last_close - prev_close) / prev_close * 100
    week_chg    = (last_close - close_5d_ago) / close_5d_ago * 100
    return last_close, day_chg, week_chg, close_5d_ago


# ─────────────────────────────────────────────
# 生成提示词文本
# ─────────────────────────────────────────────

def build_prompt(daily: pd.DataFrame, weekly: pd.DataFrame) -> str:
    d_ind = compute_indicators(daily)
    w_ind = compute_indicators(weekly)

    close_d = daily['Close'].squeeze()
    current_price   = round(float(close_d.iloc[-1]), 2)
    current_ema20   = round(float(d_ind['ema20'].iloc[-1]), 2)
    current_macd    = round(float(d_ind['macd'].iloc[-1]), 2)
    current_rsi7    = round(float(d_ind['rsi7'].iloc[-1]), 2)
    current_rsi14   = round(float(d_ind['rsi14'].iloc[-1]), 2)

    last_price, day_chg, week_chg, price_5d_ago = weekly_change(daily)

    # 日线最新交易日 OHLCV
    last_day = daily.iloc[-1]
    today_open  = round(float(last_day['Open'].squeeze()), 2)
    today_high  = round(float(last_day['High'].squeeze()), 2)
    today_low   = round(float(last_day['Low'].squeeze()), 2)
    today_close = current_price
    today_vol   = int(last_day['Volume'].squeeze())

    # 日线序列（最近15条）
    n = 15
    daily_closes  = fmt_series(close_d, 2, n)
    daily_ema20   = fmt_series(d_ind['ema20'], 2, n)
    daily_macd    = fmt_series(d_ind['macd'], 2, n)
    daily_rsi7    = fmt_series(d_ind['rsi7'], 2, n)
    daily_rsi14   = fmt_series(d_ind['rsi14'], 2, n)

    # 周线序列（最近10条）
    close_w = weekly['Close'].squeeze()
    weekly_closes = fmt_series(close_w, 2, 10)
    weekly_macd   = fmt_series(w_ind['macd'], 2, 10)
    weekly_rsi14  = fmt_series(w_ind['rsi14'], 2, 10)

    # 4h 替代用日线长期指标
    ema20_4h  = round(float(d_ind['ema20'].iloc[-1]), 2)
    ema50_4h  = round(float(d_ind['ema50'].iloc[-1]), 2)
    atr3_4h   = round(float(d_ind['atr3'].iloc[-1]), 2)
    atr14_4h  = round(float(d_ind['atr14'].iloc[-1]), 2)

    vol_current = int(daily['Volume'].squeeze().iloc[-1])
    vol_avg     = int(daily['Volume'].squeeze().tail(20).mean())

    today_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
# 黄金 (XAU/USD) 宏观大宗商品分析请求
**分析日期**: {today_str}
**数据来源**: COMEX 黄金期货 (GC=F)

---

## 价格概要

- **当前价格**: ${current_price}
- **今日 O/H/L/C**: {today_open} / {today_high} / {today_low} / {today_close}
- **今日涨跌幅**: {day_chg:+.2f}%
- **过去5交易日涨跌**: ${price_5d_ago:.2f} → ${last_price:.2f}  ({week_chg:+.2f}%)
- **今日成交量**: {today_vol:,}

---

## 当前技术指标快照

- current_price = {current_price}
- current_ema20 (日线) = {current_ema20}
- current_macd (日线) = {current_macd}
- current_rsi7 (日线) = {current_rsi7}
- current_rsi14 (日线) = {current_rsi14}

---

## 日线序列数据（最近 {n} 个交易日，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

收盘价:   [{daily_closes}]
EMA-20:   [{daily_ema20}]
MACD:     [{daily_macd}]
RSI-7:    [{daily_rsi7}]
RSI-14:   [{daily_rsi14}]

---

## 周线序列数据（最近 10 周，**从旧到新排列**）

⚠️ 最后一个数值 = 最新数据

周收盘价: [{weekly_closes}]
MACD:     [{weekly_macd}]
RSI-14:   [{weekly_rsi14}]

---

## 长期趋势背景（日线替代4H）

20日EMA: {ema20_4h}  vs.  50日EMA: {ema50_4h}
ATR-3:   {atr3_4h}   vs.  ATR-14: {atr14_4h}
今日成交量: {vol_current:,}  vs.  20日均量: {vol_avg:,}

---

## 分析任务

请基于以上数据，按照大宗商品分析框架，完成以下任务：

1. **判断当前市场制度**（Trending / Mean-Reverting / Choppy）
2. **判断整体市场情绪**（Risk-On / Risk-Off / Neutral）
3. **分析 DXY 对黄金的潜在压力方向**
4. **针对黄金给出交易建议**，严格按以下 JSON 格式输出：

```json
{{
  "period": "Daily",
  "overall_market_sentiment": "Risk-On | Risk-Off | Neutral",
  "dxy_assessment": "<DXY 趋势及对黄金的影响>",
  "asset_analysis": [
    {{
      "asset": "GOLD",
      "regime": "Trending | Mean-Reverting | Choppy",
      "action": "long | short | no_trade",
      "bias_score": <0.0 到 1.0>,
      "entry_zone": "<价格区间>",
      "profit_target": <数字 或 null>,
      "stop_loss": <数字 或 null>,
      "risk_reward_ratio": <数字 或 null>,
      "invalidation_condition": "<使该观点失效的具体信号>",
      "macro_catalyst": "<驱动此次行情的宏观逻辑>",
      "technical_setup": "<指标信号综合描述>",
      "justification": "<不超过300字的综合判断>"
    }}
  ]
}}
```

**注意**：
- profit_target 做多时必须高于入场价，做空时必须低于入场价
- risk_reward_ratio 必须 ≥ 2.0，否则改为 no_trade
- 当 action = no_trade 时，profit_target / stop_loss / risk_reward_ratio 填 null
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    daily, weekly = fetch_gold_data()
    prompt = build_prompt(daily, weekly)

    output_path = "gold_prompt_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("\n" + "=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\n已保存到文件: {output_path}")
    print("请将上方内容复制粘贴到 Claude.ai 对话框，即可获得分析结果。")


if __name__ == "__main__":
    main()
