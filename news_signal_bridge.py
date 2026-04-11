"""
新闻信号桥接模块 — news_signal_bridge.py
==========================================
将新闻监控系统（RAG API + n8n SQLite）的结构化信号接入主分析系统。

数据源优先级（按顺序尝试）：
  1. RAG API（http://43.139.5.125:8080）— 语义检索 + LLM 情绪标注，最高质量
  2. SQLite 数据库（n8n 工作流直连）— 旧有结构化信号，本地/ECS 可用
  3. yfinance.news — 降级兜底，仅提供原始标题

环境变量配置：
  RAG_API_URL  — RAG API 地址（默认: http://43.139.5.125:8080）
  RAG_API_KEY  — RAG API Key（默认: 12345678）
  NEWS_DB_PATH — SQLite 数据库路径（可选，优先于内置路径）

用法（在 tech_stock_analysis.py 主流程中替换 fetch_recent_news）：
    from news_signal_bridge import fetch_news_signals, format_news_signals_section

    # 替换原来的:
    # news_items = fetch_recent_news(ticker)
    news_context = fetch_news_signals(ticker, lookback_hours=72)

    # 替换原来的:
    # {_format_news_section(news_items) if news_items else ""}
    {format_news_signals_section(news_context)}
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────
# RAG API 配置
# ──────────────────────────────────────────────────────────────────

RAG_API_URL = os.environ.get("RAG_API_URL", "http://43.139.5.125:8080")
RAG_API_KEY = os.environ.get("RAG_API_KEY", "12345678")


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

_NEWS_DB_CANDIDATES = [
    os.environ.get("NEWS_DB_PATH", ""),                          # 优先：环境变量
    str(Path(__file__).parent / "news_signals.db"),              # 本地同目录
    "/opt/finance-analysis/n8n_data/database.sqlite",            # 阿里云 ECS 路径
]

# 资产关键词映射（与新闻系统 portfolio_config 中的 keywords 对齐）
_ASSET_KEYWORDS = {
    "NVDA":  ["NVDA", "Nvidia", "Jensen", "GPU", "H100", "Blackwell", "CUDA"],
    "MSFT":  ["MSFT", "Microsoft", "Azure", "Copilot", "OpenAI"],
    "GOOGL": ["GOOGL", "Google", "Alphabet", "Gemini", "YouTube", "Cloud"],
    "AAPL":  ["AAPL", "Apple", "iPhone", "Vision Pro", "App Store"],
    "META":  ["META", "Meta", "Facebook", "Instagram", "WhatsApp", "Llama"],
    "AMZN":  ["AMZN", "Amazon", "AWS", "Bedrock", "Prime"],
    "BTC":   ["BTC", "Bitcoin", "crypto", "cryptocurrency", "Coinbase"],
    "GOLD":  ["gold", "XAU", "PAXG", "GLD", "precious metals", "Fed", "FOMC"],
    "SLV":   ["silver", "SLV", "gold-silver ratio", "precious metals"],
    "COPX":  ["copper", "COPX", "Chile", "Peru", "PMI", "mining"],
    "REMX":  ["rare earth", "REMX", "tungsten", "neodymium", "China export"],
    "USO":   ["oil", "crude", "OPEC", "WTI", "Brent", "EIA", "barrel"],
}


# ──────────────────────────────────────────────────────────────────
# RAG API 数据获取
# ──────────────────────────────────────────────────────────────────

def _fetch_rag_news(
    ticker: str,
    hours: int = 72,
    top_k: int = 8,
    min_importance: float = 0.3,
) -> list[dict]:
    """
    调用 RAG API 获取语义检索新闻，返回原始 results 列表。
    失败时返回空列表，不抛异常。

    策略：先用 asset_filter 精确过滤，若无结果则退回纯语义搜索。
    """
    if not RAG_API_URL or not RAG_API_KEY:
        return []

    keywords = _ASSET_KEYWORDS.get(ticker.upper(), [ticker])
    query = f"{ticker} {' '.join(keywords[:3])} stock news analysis"

    try:
        import httpx
        headers = {"X-API-Key": RAG_API_KEY, "Content-Type": "application/json"}
        payload = {
            "query": query,
            "top_k": top_k,
            "asset_filter": ticker,
            "min_importance": min_importance,
            "hours": hours,
        }
        resp = httpx.post(
            f"{RAG_API_URL}/api/v1/rag/search",
            headers=headers,
            json=payload,
            timeout=10.0,
            verify=False,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                print(f"  [RAG] {ticker}: 获取 {len(results)} 条新闻（asset_filter 匹配）")
                return results
            # 无结果时退回纯语义搜索（不带 asset_filter）
            payload_no_filter = {k: v for k, v in payload.items() if k != "asset_filter"}
            resp2 = httpx.post(
                f"{RAG_API_URL}/api/v1/rag/search",
                headers=headers,
                json=payload_no_filter,
                timeout=10.0,
                verify=False,
            )
            if resp2.status_code == 200:
                results2 = resp2.json().get("results", [])
                # Client-side relevance filter: only keep results that mention
                # the target ticker's keywords — prevents unrelated companies
                # (e.g., gold mining stocks) from polluting NVDA/MSFT analysis
                kws = _ASSET_KEYWORDS.get(ticker.upper(), [ticker])
                results2_filtered = [
                    r for r in results2
                    if any(kw.lower() in (r.get("chunk_text") or "").lower() for kw in kws)
                ]
                if results2_filtered:
                    print(f"  [RAG] {ticker}: 语义搜索获取 {len(results2)} 条，相关性过滤后保留 {len(results2_filtered)} 条")
                    return results2_filtered
                else:
                    print(f"  [RAG] {ticker}: 语义搜索结果均与 {ticker} 无关，过滤后为空，跳过注入")
                    return []
        else:
            print(f"  [RAG] {ticker}: API 返回 {resp.status_code}")
    except Exception as e:
        print(f"  [RAG] {ticker}: 请求失败 ({e})")

    return []


# SEC EDGAR 样板文本特征（地址/表头/元数据，无实质内容）
_SEC_BOILERPLATE_SIGNALS = [
    "mailing address", "business address", "street1", "central index key",
    "accession number", "conformed submission type", "form type", "filer id",
    "zip code", "state of incorporation",
]


def _is_sec_boilerplate(chunk_text: str) -> bool:
    """
    Detect SEC EDGAR filing header/address chunks that contain no material content.
    Returns True if the text is boilerplate (address, filing metadata, etc.).
    """
    if not chunk_text or len(chunk_text.strip()) < 20:
        return True
    text_lower = chunk_text.lower()
    hits = sum(1 for pat in _SEC_BOILERPLATE_SIGNALS if pat in text_lower)
    return hits >= 2  # ≥2 boilerplate patterns = almost certainly header, not material event


def _rag_results_to_signals(rag_results: list[dict]) -> list[dict]:
    """
    将 RAG API 返回的 results 转换为内部 trade_signals 格式。

    实际响应结构：
      {
        "chunk_text": "...",
        "similarity_score": 0.85,
        "metadata": {
          "sentiment": "bullish|bearish|neutral",
          "importance_score": 0.72,
          "source_name": "sec_edgar|polygon|alpha_vantage|cnbc|wsj",
          "published_at": "2026-04-08T10:00:00Z",
          "summary_cn": "中文摘要",
          "change_type": "fundamental|macro|sentiment",
          ...
        }
      }
    """
    signals = []
    for r in rag_results:
        meta = r.get("metadata") or {}

        chunk = r.get("chunk_text") or ""
        if _is_sec_boilerplate(chunk):
            continue  # Skip SEC boilerplate (address/header) — no material content

        importance = float(meta.get("importance_score", 0.3))
        sentiment  = meta.get("sentiment", "neutral")

        if importance >= 0.7:
            confidence = "high"
        elif importance >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        source_name = meta.get("source_name", "")
        # 优先使用 metadata 里已标注的 change_type，否则按来源推断
        change_type = meta.get("change_type") or (
            "fundamental" if ("sec_edgar" in source_name or "edgar" in source_name)
            else "macro" if source_name in ("wsj", "cnbc")
            else "sentiment"
        )

        # is_structural only when SEC content has high importance AND is not boilerplate
        is_structural = (
            (("sec_edgar" in source_name) and importance >= 0.6)
            or (importance >= 0.8)
        )

        published = (meta.get("published_at") or "")[:16]
        reason = (r.get("chunk_text") or "")[:200]   # English original for prompt injection

        signals.append({
            "direction": sentiment,       # "bullish" | "bearish" | "neutral"
            "timeframe": meta.get("timeframe", "medium"),
            "confidence": confidence,
            "change_type": change_type,
            "action": "watch",
            "reason": reason,
            "created_at": published,
            "source": source_name,
            "is_structural": is_structural,
            "relevance_score": round(importance * 10),
            "title": (r.get("chunk_text") or "")[:120],
            "importance_score": importance,
        })
    return signals


# ──────────────────────────────────────────────────────────────────
# 数据库连接
# ──────────────────────────────────────────────────────────────────

def _find_news_db() -> Optional[str]:
    """找到可用的新闻系统数据库路径"""
    for path in _NEWS_DB_CANDIDATES:
        if path and Path(path).exists():
            return path
    return None


def _query_news_signals(ticker: str, lookback_hours: int = 72) -> list[dict]:
    """
    从新闻系统 SQLite 查询结构化交易信号。

    trade_signals 表结构（来自新闻监控系统）：
        id, article_id, asset, direction, timeframe, confidence,
        change_type, action, reason, llm_model, notified, created_at
    """
    db_path = _find_news_db()
    if not db_path:
        return []

    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row

        # 查询有向信号（排除 neutral）
        rows = conn.execute("""
            SELECT
                ts.asset,
                ts.direction,
                ts.timeframe,
                ts.confidence,
                ts.change_type,
                ts.action,
                ts.reason,
                ts.created_at,
                a.title,
                a.source,
                a.is_structural,
                a.relevance_score
            FROM trade_signals ts
            LEFT JOIN articles a ON ts.article_id = a.id
            WHERE ts.asset = ?
              AND ts.direction != 'neutral'
              AND ts.created_at >= ?
            ORDER BY ts.created_at DESC
            LIMIT 10
        """, (ticker, cutoff_str)).fetchall()

        conn.close()
        return [dict(row) for row in rows]

    except Exception as e:
        print(f"  [新闻桥接] 数据库查询失败 ({db_path}): {e}")
        return []


def _query_structural_events(ticker: str, lookback_hours: int = 168) -> list[dict]:
    """
    查询近期结构性变化事件（is_structural=1），窗口更长（默认7天）。
    结构性事件比普通信号更持久，影响中长期走势。
    包括：SEC 8-K、FOMC声明、监管政策、重大并购等。
    """
    db_path = _find_news_db()
    if not db_path:
        return []

    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                a.title,
                a.source,
                a.published_at,
                a.relevance_score,
                ts.direction,
                ts.change_type,
                ts.confidence,
                ts.reason
            FROM articles a
            LEFT JOIN trade_signals ts ON ts.article_id = a.id
            WHERE a.is_structural = 1
              AND a.created_at >= ?
              AND (
                  ts.asset = ?
                  OR a.title LIKE '%' || ? || '%'
              )
            ORDER BY a.published_at DESC
            LIMIT 5
        """, (cutoff_str, ticker, ticker)).fetchall()

        conn.close()
        return [dict(row) for row in rows]

    except Exception as e:
        print(f"  [新闻桥接] 结构性事件查询失败: {e}")
        return []


# ──────────────────────────────────────────────────────────────────
# 主接口
# ──────────────────────────────────────────────────────────────────

def fetch_news_signals(
    ticker: str,
    lookback_hours: int = 72,
    fallback_to_yfinance: bool = True,
) -> dict:
    """
    获取新闻信号，返回结构化上下文供 build_prompt_equity() 使用。

    数据源优先级：
      1. RAG API（语义检索 + LLM 情绪标注）
      2. SQLite 数据库（n8n 本地/ECS）
      3. yfinance.news（原始标题降级）

    返回格式：
    {
        "source": "rag_api" | "news_system" | "yfinance" | "none",
        "ticker": str,
        "trade_signals": [...],
        "structural_events": [...],
        "signal_summary": str,
        "bullish_count": int,
        "bearish_count": int,
        "has_structural": bool,
    }
    """
    # ── Step 1: RAG API（最高质量，优先尝试）──
    rag_results = _fetch_rag_news(ticker, hours=lookback_hours)
    if rag_results:
        trade_signals = _rag_results_to_signals(rag_results)
        directional = [s for s in trade_signals if s["direction"] in ("bullish", "bearish")]
        bullish = [s for s in directional if s["direction"] == "bullish"]
        bearish = [s for s in directional if s["direction"] == "bearish"]
        structural_events = [s for s in trade_signals if s.get("is_structural")]
        has_structural = bool(structural_events)

        summary_parts = []
        if bullish:
            summary_parts.append(f"{len(bullish)} bullish")
        if bearish:
            summary_parts.append(f"{len(bearish)} bearish")
        high_conf = [s for s in directional if s["confidence"] == "high"]
        if high_conf:
            summary_parts.append(f"{len(high_conf)} high-confidence")
        if has_structural:
            summary_parts.append("⚠️ structural event detected")
        neutral = [s for s in trade_signals if s["direction"] == "neutral"]
        if neutral:
            summary_parts.append(f"{len(neutral)} neutral background")
        signal_summary = ", ".join(summary_parts) if summary_parts else f"No directional signals in last {lookback_hours}h"

        print(f"  [RAG桥接] {ticker}: {signal_summary}")
        return {
            "source": "rag_api",
            "ticker": ticker,
            "trade_signals": directional,     # 仅保留有方向的信号参与 bias 调整
            "structural_events": structural_events,
            "signal_summary": signal_summary,
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "has_structural": has_structural,
            "_rag_raw": trade_signals,         # 含 neutral，供格式化时展示
        }

    # ── Step 2: SQLite 数据库（本地/ECS n8n 工作流）──
    db_path = _find_news_db()
    if db_path:
        trade_signals = _query_news_signals(ticker, lookback_hours)
        structural_events = _query_structural_events(ticker, lookback_hours * 2)
        bullish = [s for s in trade_signals if s["direction"] == "bullish"]
        bearish = [s for s in trade_signals if s["direction"] == "bearish"]
        has_structural = any(s.get("is_structural") for s in trade_signals) or bool(structural_events)

        if trade_signals:
            high_conf = [s for s in trade_signals if s["confidence"] == "high"]
            summary_parts = []
            if bullish:
                summary_parts.append(f"{len(bullish)} 条看多信号")
            if bearish:
                summary_parts.append(f"{len(bearish)} 条看空信号")
            if high_conf:
                summary_parts.append(f"其中 {len(high_conf)} 条高置信度")
            if has_structural:
                summary_parts.append("⚠️ 含结构性变化事件")
            signal_summary = "，".join(summary_parts)
        else:
            signal_summary = f"近 {lookback_hours}h 无针对 {ticker} 的定向信号"

        print(f"  [SQLite桥接] {ticker}: {signal_summary} (db={Path(db_path).name})")
        return {
            "source": "news_system",
            "ticker": ticker,
            "trade_signals": trade_signals,
            "structural_events": structural_events,
            "signal_summary": signal_summary,
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "has_structural": has_structural,
        }

    # ── Step 3: yfinance 降级 ──
    print(f"  [新闻桥接] RAG API 和数据库均不可用，回退到 yfinance.news")
    if fallback_to_yfinance:
        return _fallback_yfinance(ticker)
    return _empty_context(ticker, "none")


def _fallback_yfinance(ticker: str) -> dict:
    """降级：使用 yfinance.news（原有行为）"""
    try:
        import yfinance as yf
        from curl_cffi import requests as curl_requests
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        session = curl_requests.Session(impersonate="chrome", verify=False, proxy=proxy)
        t = yf.Ticker(ticker, session=session)
        raw = t.news or []
        items = []
        for n in raw[:8]:
            published_ts = n.get("providerPublishTime", 0)
            try:
                published_str = pd.Timestamp(published_ts, unit="s").strftime("%Y-%m-%d %H:%M")
            except Exception:
                published_str = "unknown"
            items.append({
                "direction": None,
                "title": n.get("title", ""),
                "source": n.get("publisher", ""),
                "created_at": published_str,
                "is_structural": False,
                "relevance_score": 1,
                "confidence": None,
                "change_type": "sentiment",
                "reason": n.get("title", ""),
            })
        return {
            "source": "yfinance",
            "ticker": ticker,
            "trade_signals": items,
            "structural_events": [],
            "signal_summary": f"yfinance 降级: {len(items)} 条原始标题",
            "bullish_count": 0,
            "bearish_count": 0,
            "has_structural": False,
        }
    except Exception:
        return _empty_context(ticker, "none")


def _empty_context(ticker: str, source: str) -> dict:
    return {
        "source": source,
        "ticker": ticker,
        "trade_signals": [],
        "structural_events": [],
        "signal_summary": "无新闻数据",
        "bullish_count": 0,
        "bearish_count": 0,
        "has_structural": False,
    }


# ──────────────────────────────────────────────────────────────────
# Prompt 格式化
# ──────────────────────────────────────────────────────────────────

def format_news_signals_section(ctx: dict) -> str:
    """
    Format news signal context as English Markdown for injection into build_prompt_equity().

    Supports three data sources:
    - rag_api    : RAG semantic search (highest quality, with importance scores)
    - news_system: SQLite n8n workflow (structured signals)
    - yfinance   : raw headline fallback
    """
    if not ctx or (not ctx.get("trade_signals") and not ctx.get("structural_events")
                   and not ctx.get("_rag_raw")):
        return ""

    source = ctx.get("source", "unknown")
    source_label = {
        "rag_api":     "RAG API (Semantic Search + LLM Sentiment)",
        "news_system": "SQLite News System",
        "yfinance":    "yfinance (Fallback Mode)",
    }.get(source, source)

    lines = [
        "",
        "---",
        "",
        "## IV. Recent News & Market Intelligence",
        "",
        f"> **Data source**: {source_label} | **Signals**: {ctx.get('signal_summary', '')}",
        "",
    ]

    # ── High-impact structural events (highest priority) ──
    structural = ctx.get("structural_events", [])
    if structural:
        lines += [
            "### High-Impact Structural Events (Last 7 Days)",
            "",
            "| Date | Source | Direction | Type | Key Insight |",
            "|------|--------|-----------|------|-------------|",
        ]
        for ev in structural:
            direction_str = {
                "bullish": "Bullish ↑", "bearish": "Bearish ↓"
            }.get(ev.get("direction", ""), "Neutral")
            change_type = ev.get("change_type") or "—"
            src = (ev.get("source") or "").replace("sec_edgar", "SEC 8-K")
            published = (ev.get("published_at") or ev.get("created_at") or "")[:10]
            reason = (ev.get("reason") or ev.get("title") or "")[:100]
            lines.append(f"| {published} | {src} | {direction_str} | {change_type} | {reason} |")
        lines.append("")
        lines += [
            "> **Structural event adjustment rules**:",
            "> - Structural bullish (SEC 8-K / FOMC / regulatory tailwind) → bias_score **+0.05 to +0.10**",
            "> - Structural bearish (export controls / regulatory crackdown / guidance cut) → bias_score **-0.05 to -0.15**, cap position at 0.3",
            "",
        ]

    # ── RAG API format (with importance_score) ──
    if source == "rag_api":
        all_signals = ctx.get("_rag_raw") or ctx.get("trade_signals", [])
        directional = [s for s in all_signals if s.get("direction") in ("bullish", "bearish")]
        neutral_bg  = [s for s in all_signals if s.get("direction") == "neutral"]

        if directional:
            lines += [
                "### Directional Signals (RAG Semantic Search, Last 72h)",
                "",
                "| Date | Direction | Importance | Source | News Excerpt |",
                "|------|-----------|------------|--------|--------------|",
            ]
            for sig in directional[:6]:
                direction_str = {"bullish": "Bullish ↑", "bearish": "Bearish ↓"}.get(sig.get("direction", ""), "—")
                importance = sig.get("importance_score", 0)
                imp_str = f"{importance:.2f} ({'High' if importance >= 0.7 else ('Med' if importance >= 0.4 else 'Low')})"
                src = (sig.get("source") or "").replace("sec_edgar", "SEC 8-K")
                created = (sig.get("created_at") or "")[:10]
                reason = (sig.get("reason") or "")[:100]
                lines.append(f"| {created} | {direction_str} | {imp_str} | {src} | {reason} |")
            lines.append("")

        if neutral_bg:
            lines += [
                "### Background Context (Neutral)",
                "",
            ]
            for i, sig in enumerate(neutral_bg[:3], 1):
                created = (sig.get("created_at") or "")[:10]
                reason = (sig.get("reason") or "")[:120]
                src = sig.get("source") or ""
                lines.append(f"{i}. **[{created}]** {reason}  _({src})_")
            lines.append("")

        lines += [
            "### News Signal Bias Adjustment Rules (RAG Mode)",
            "",
            "| Condition | bias_score Adjustment |",
            "|-----------|----------------------|",
            "| High-confidence bullish (importance≥0.7, fundamental/macro type, ≥1 item) | **+0.05** (apply once) |",
            "| High-confidence bearish (importance≥0.7, ≥1 item) | **-0.08** (overrides bullish) |",
            "| Medium-confidence (importance 0.4–0.7) aligned with technical direction | **+0.03** |",
            "| Pure sentiment signals | **±0.02** (lowest weight) |",
            "| SEC 8-K material structural event (see above) | See structural rules |",
            "| Conflicting bullish and bearish signals | **0** (no adjustment) |",
        ]

    # ── SQLite news system format ──
    elif source == "news_system":
        trade_signals = [s for s in ctx.get("trade_signals", []) if s.get("direction")]
        if trade_signals:
            lines += [
                "### Directional Signals (Last 72h)",
                "",
                "| Date | Direction | Confidence | Type | Key Insight |",
                "|------|-----------|------------|------|-------------|",
            ]
            for sig in trade_signals[:6]:
                direction_str = {"bullish": "Bullish ↑", "bearish": "Bearish ↓"}.get(sig.get("direction", ""), "—")
                conf_str = {"high": "High", "medium": "Medium", "low": "Low"}.get(sig.get("confidence", ""), "—")
                change_type = sig.get("change_type") or "—"
                reason = (sig.get("reason") or "")[:80]
                created = (sig.get("created_at") or "")[:16]
                lines.append(f"| {created} | {direction_str} | {conf_str} | {change_type} | {reason} |")
            lines.append("")

        lines += [
            "### News Signal Bias Adjustment Rules",
            "",
            "| Condition | bias_score Adjustment |",
            "|-----------|----------------------|",
            "| High-confidence bullish (≥1 item, fundamental/macro type) | **+0.05** (apply once) |",
            "| High-confidence bearish (≥1 item) | **-0.08** (overrides bullish) |",
            "| Medium-confidence aligned with technical direction | **+0.03** |",
            "| Pure sentiment signals | **±0.02** (lowest weight) |",
            "| Conflicting bullish and bearish signals | **0** (no adjustment) |",
        ]

    # ── yfinance fallback mode ──
    else:
        raw_items = ctx.get("trade_signals", [])
        if raw_items:
            lines += [
                "### Recent News Headlines (yfinance Fallback)",
                "",
                "> ⚠️ RAG API unavailable. Using raw headlines — lower signal quality, reduce bias adjustment by 50%.",
                "",
            ]
            for i, n in enumerate(raw_items[:8], 1):
                lines.append(f"{i}. **[{n.get('created_at', '')}]** {n.get('reason', '')}  _({n.get('source', '')})_")
            lines.append("")

        lines += [
            "### News Signal Bias Adjustment Rules (Fallback Mode)",
            "",
            "| Condition | bias_score Adjustment |",
            "|-----------|----------------------|",
            "| Headlines aligned with technical signal direction | **+0.02** |",
            "| Major negative event (regulatory / litigation / guidance cut) | **-0.05** |",
        ]

    lines += [
        "",
        "> **Note**: News signals are supplementary only. When news conflicts with the technical signal, the technical signal takes precedence.",
        "",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# 便捷函数：bias_score 预调整（供 Python 层使用，不依赖 LLM）
# ──────────────────────────────────────────────────────────────────

def compute_news_bias_adjustment(ctx: dict, base_direction: str = "long") -> float:
    """
    根据新闻信号计算 bias_score 的预调整量（仅作参考，最终由 LLM 决定）。

    参数
    ----
    ctx            : fetch_news_signals() 返回的上下文
    base_direction : 技术分析的主方向（"long" 或 "short"）

    返回
    ----
    float: 建议的 bias_score 调整量（-0.15 到 +0.10）
    """
    if not ctx or ctx.get("source") == "none":
        return 0.0

    trade_signals = ctx.get("trade_signals", [])
    if not trade_signals:
        return 0.0

    adjustment = 0.0
    tech_direction_map = {"long": "bullish", "short": "bearish"}
    aligned_direction = tech_direction_map.get(base_direction, "bullish")

    high_conf_aligned    = 0
    high_conf_opposite   = 0
    has_structural_positive = False
    has_structural_negative = False

    for sig in trade_signals:
        direction = sig.get("direction", "")
        confidence = sig.get("confidence", "low")
        change_type = sig.get("change_type", "sentiment")
        is_structural = sig.get("is_structural", False)

        # 结构性事件权重最高
        if is_structural or change_type in ("fundamental", "macro", "regulatory"):
            weight = 1.5
        else:
            weight = 1.0

        if direction == aligned_direction:
            if confidence == "high":
                high_conf_aligned += 1
                adjustment += 0.05 * weight
            elif confidence == "medium":
                adjustment += 0.03 * weight
            else:
                adjustment += 0.01

            if is_structural:
                has_structural_positive = True

        elif direction != aligned_direction and direction in ("bullish", "bearish"):
            if confidence == "high":
                high_conf_opposite += 1
                adjustment -= 0.08 * weight
            elif confidence == "medium":
                adjustment -= 0.04 * weight

            if is_structural:
                has_structural_negative = True

    # 矛盾信号相互抵消
    if high_conf_aligned > 0 and high_conf_opposite > 0:
        adjustment = 0.0

    # 结构性负面信号强制拉低
    if has_structural_negative and not has_structural_positive:
        adjustment = min(adjustment, -0.10)

    # 硬上下限
    adjustment = max(-0.15, min(0.10, adjustment))

    return round(adjustment, 3)


# ──────────────────────────────────────────────────────────────────
# CLI 调试入口
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="新闻信号桥接模块 — 调试工具")
    parser.add_argument("--ticker", default="NVDA", help="资产代码")
    parser.add_argument("--hours", type=int, default=72, help="回看时间窗口（小时）")
    parser.add_argument("--show-prompt", action="store_true", help="显示 prompt 区块")
    args = parser.parse_args()

    db = _find_news_db()
    print(f"数据库路径: {db or '未找到（将使用 yfinance 降级）'}")
    print()

    ctx = fetch_news_signals(args.ticker, lookback_hours=args.hours)

    print(f"\n=== {args.ticker} 新闻信号 ===")
    print(f"来源: {ctx['source']}")
    print(f"摘要: {ctx['signal_summary']}")
    print(f"看多: {ctx['bullish_count']} | 看空: {ctx['bearish_count']} | 含结构性: {ctx['has_structural']}")

    adj = compute_news_bias_adjustment(ctx, base_direction="long")
    print(f"\nbias_score 预调整（long方向）: {adj:+.3f}")

    if args.show_prompt:
        print("\n=== Prompt 区块预览 ===")
        print(format_news_signals_section(ctx))
