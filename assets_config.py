"""
资产配置注册表
所有分析脚本和持仓跟踪器共用此注册表。

asset_key     : 内部唯一标识（大写）
ticker        : yfinance 行情 ticker
type          : equity / commodity / crypto / etf
script        : 调用哪个分析脚本
script_args   : 额外传给脚本的参数列表
output_file   : API 分析结果输出文件
prompt_file   : 提示词输出文件
backtest_dir  : 回测结果目录（None = 尚未实现）
sector        : 所属板块（用于横向对比）
ccy           : 计价货币（USD / USDT）
description   : 简短描述
"""

ASSET_UNIVERSE: dict[str, dict] = {

    # ─────────────── 贵金属 / 大宗商品 ───────────────
    "GOLD": {
        "ticker":       "GC=F",
        "type":         "commodity",
        "script":       "gold_analysis.py",
        "script_args":  [],
        "output_file":  "gold_api_output.txt",
        "prompt_file":  "gold_prompt_output.txt",
        "backtest_dir": "backtest_results",
        "sector":       "Commodities/PreciousMetals",
        "ccy":          "USD",
        "description":  "COMEX 黄金期货",
    },
    "SILVER": {
        "ticker":       "SLV",          # iShares Silver Trust ETF（有成交量，yfinance 友好）
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "SLV"],
        "output_file":  "slv_api_output.txt",
        "prompt_file":  "slv_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/PreciousMetals",
        "ccy":          "USD",
        "description":  "白银 ETF（SLV）",
    },
    "COPPER": {
        "ticker":       "COPX",         # Global X Copper Miners ETF
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "COPX"],
        "output_file":  "copx_api_output.txt",
        "prompt_file":  "copx_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/IndustrialMetals",
        "ccy":          "USD",
        "description":  "铜矿 ETF（COPX）",
    },
    "RARE_EARTH": {
        "ticker":       "REMX",         # VanEck Rare Earth & Strategic Metals ETF（含钨矿敞口）
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "REMX"],
        "output_file":  "remx_api_output.txt",
        "prompt_file":  "remx_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/RareEarth",
        "ccy":          "USD",
        "description":  "稀土/战略金属 ETF（含钨；REMX）",
    },
    "OIL": {
        "ticker":       "USO",          # United States Oil Fund ETF
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "USO"],
        "output_file":  "uso_api_output.txt",
        "prompt_file":  "uso_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/Energy",
        "ccy":          "USD",
        "description":  "原油 ETF（USO）",
    },

    # ─────────────── 加密货币 ───────────────
    "BTC": {
        "ticker":       "BTC-USD",
        "type":         "crypto",
        "script":       "btc_analysis.py",
        "script_args":  [],
        "output_file":  "btc_api_output.txt",
        "prompt_file":  "btc_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Crypto",
        "ccy":          "USD",
        "description":  "比特币",
    },

    # ─────────────── 纳斯达克科技股 ───────────────
    "GOOGL": {
        "ticker":       "GOOGL",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "GOOGL"],
        "output_file":  "googl_api_output.txt",
        "prompt_file":  "googl_prompt_output.txt",
        "backtest_dir": "googl_backtest_results",
        "sector":       "Technology/InternetAds",
        "ccy":          "USD",
        "description":  "Alphabet / Google",
    },
    "MSFT": {
        "ticker":       "MSFT",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "MSFT"],
        "output_file":  "msft_api_output.txt",
        "prompt_file":  "msft_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/CloudAI",
        "ccy":          "USD",
        "description":  "Microsoft",
    },
    "NVDA": {
        "ticker":       "NVDA",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "NVDA"],
        "output_file":  "nvda_api_output.txt",
        "prompt_file":  "nvda_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Semiconductors",
        "ccy":          "USD",
        "description":  "NVIDIA",
    },
    "AAPL": {
        "ticker":       "AAPL",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "AAPL"],
        "output_file":  "aapl_api_output.txt",
        "prompt_file":  "aapl_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Consumer",
        "ccy":          "USD",
        "description":  "Apple",
    },
    "META": {
        "ticker":       "META",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "META"],
        "output_file":  "meta_api_output.txt",
        "prompt_file":  "meta_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/SocialMedia",
        "ccy":          "USD",
        "description":  "Meta Platforms",
    },
    "AMZN": {
        "ticker":       "AMZN",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "AMZN"],
        "output_file":  "amzn_api_output.txt",
        "prompt_file":  "amzn_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/CloudEcommerce",
        "ccy":          "USD",
        "description":  "Amazon",
    },
}

# ── 预定义扫描分组（market_scan.py 中使用 --group 选择）──
SCAN_GROUPS: dict[str, list[str]] = {
    "all":        list(ASSET_UNIVERSE.keys()),
    "tech":       ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN"],
    "commodities":["GOLD", "SILVER", "COPPER", "RARE_EARTH", "OIL"],
    "crypto":     ["BTC"],
    "metals":     ["GOLD", "SILVER", "COPPER", "RARE_EARTH"],
    "quick":      ["GOLD", "BTC", "NVDA", "MSFT"],  # 日常快速扫描
}
