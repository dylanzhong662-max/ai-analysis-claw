"""
资产配置注册表 — Single Source of Truth
========================================
所有分析脚本、持仓跟踪器、新闻桥接模块、RAG 采集器共用此注册表。

新增资产只需在 ASSET_UNIVERSE 里加一条记录，其余脚本自动感知，无需逐一修改。

字段说明
--------
asset_key         : 内部唯一标识（大写）
ticker            : yfinance 行情 ticker
type              : equity / commodity / crypto / etf
script            : 调用哪个分析脚本
script_args       : 额外传给脚本的参数列表
output_file       : API 分析结果输出文件
prompt_file       : 提示词输出文件
backtest_dir      : 回测结果目录（None = 尚未实现）
sector            : 所属板块（用于横向对比）
ccy               : 计价货币（USD / USDT）
description       : 简短描述

--- 每日定时任务配置 ---
daily_scan        : True = 纳入 run_daily.sh 每日分析；False = 仅在手动扫描时运行
daily_extra_args  : run_daily.sh 中额外追加给脚本的 CLI 参数（如 --trade --max-usdt 30）

--- 新闻 RAG 配置（news_signal_bridge.py 从此处读取，无需手动维护两份）---
rag_weight        : 该资产在 RAG portfolio.json 中的权重（0.0 表示禁用 RAG 推送）
news_keywords     : RAG 检索时使用的扩展关键词列表（越详细召回率越高）
news_primary_terms: 主标识符（整词匹配，必须命中其一才算相关，防止误判）

--- SEC / 财报采集配置（RAG 采集器从此处读取）---
sec_cik           : SEC EDGAR 公司 CIK 编号（仅 equity 类型需要，ETF/crypto/commodity 留 None）
earnings_tracking : True = 纳入 polygon.py 季报财务采集
insider_tracking  : True = 纳入 form4.py 内部人交易采集
"""

OUTPUTS_DIR = "outputs"   # 所有 LLM 分析输出统一存放目录

ASSET_UNIVERSE: dict[str, dict] = {

    # ─────────────── 贵金属 / 大宗商品 ───────────────
    "GOLD": {
        "ticker":       "GC=F",
        "type":         "commodity",
        "script":       "gold_analysis.py",
        "script_args":  [],
        "output_file":  "outputs/gold_api_output.txt",
        "prompt_file":  "outputs/gold_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/PreciousMetals",
        "ccy":          "USD",
        "description":  "COMEX 黄金期货",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": ["--trade", "--max-usdt", "30"],
        # 新闻 RAG
        "rag_weight":        0.10,
        "news_keywords":     ["gold", "GOLD", "XAU", "precious metals", "safe haven",
                              "gold price", "GLD", "PAXG", "Fed", "FOMC", "inflation",
                              "real yields", "dollar index"],
        "news_primary_terms":["gold", "XAU", "PAXG", "GLD"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },
    "SILVER": {
        "ticker":       "SLV",
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "SLV"],
        "output_file":  "outputs/slv_api_output.txt",
        "prompt_file":  "outputs/slv_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/PreciousMetals",
        "ccy":          "USD",
        "description":  "白银 ETF（SLV）",
        # 每日任务
        "daily_scan":       False,   # 白银波动与黄金高度相关，默认跟随手动扫描
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.05,
        "news_keywords":     ["silver", "SLV", "gold-silver ratio", "precious metals",
                              "industrial demand", "solar panels", "silver price", "XAG",
                              "photovoltaic", "EV battery"],
        "news_primary_terms":["silver", "SLV", "XAG"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },
    "COPPER": {
        "ticker":       "COPX",
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "COPX"],
        "output_file":  "outputs/copx_api_output.txt",
        "prompt_file":  "outputs/copx_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/IndustrialMetals",
        "ccy":          "USD",
        "description":  "铜矿 ETF（COPX）",
        # 每日任务
        "daily_scan":       False,   # 铜价受全球 PMI 驱动，波动相对低频，默认手动扫描
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.05,
        "news_keywords":     ["copper", "COPX", "copper price", "copper miners",
                              "Chile", "Peru", "PMI", "manufacturing",
                              "energy transition", "electric vehicles", "EV"],
        "news_primary_terms":["copper", "COPX"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },
    "RARE_EARTH": {
        "ticker":       "REMX",
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "REMX"],
        "output_file":  "outputs/remx_api_output.txt",
        "prompt_file":  "outputs/remx_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/RareEarth",
        "ccy":          "USD",
        "description":  "稀土/战略金属 ETF（含钨；REMX）",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.05,
        "news_keywords":     ["rare earth", "REMX", "rare earth metals", "tungsten",
                              "neodymium", "China export controls", "EV magnet",
                              "REE", "critical minerals", "strategic metals"],
        "news_primary_terms":["rare earth", "REMX", "tungsten", "neodymium"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },
    "OIL": {
        "ticker":       "USO",
        "type":         "etf",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "USO"],
        "output_file":  "outputs/uso_api_output.txt",
        "prompt_file":  "outputs/uso_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Commodities/Energy",
        "ccy":          "USD",
        "description":  "原油 ETF（USO）",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.05,
        "news_keywords":     ["crude oil", "WTI", "Brent", "OPEC", "OPEC+",
                              "petroleum", "oil price", "barrel", "EIA",
                              "oil inventory", "shale", "refinery", "USO"],
        "news_primary_terms":["crude oil", "WTI", "Brent", "OPEC", "USO"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },

    # ─────────────── 加密货币 ───────────────
    "BTC": {
        "ticker":       "BTC-USD",
        "type":         "crypto",
        "script":       "btc_analysis.py",
        "script_args":  [],
        "output_file":  "outputs/btc_api_output.txt",
        "prompt_file":  "outputs/btc_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Crypto",
        "ccy":          "USD",
        "description":  "比特币",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.15,
        "news_keywords":     ["BTC", "Bitcoin", "bitcoin", "crypto", "cryptocurrency",
                              "ETF approval", "strategic reserve", "Coinbase",
                              "halving", "on-chain", "Lightning Network"],
        "news_primary_terms":["BTC", "Bitcoin"],
        # SEC / 财报
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },

    # ─────────────── 纳斯达克科技股 ───────────────
    "GOOGL": {
        "ticker":       "GOOGL",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "GOOGL"],
        "output_file":  "outputs/googl_api_output.txt",
        "prompt_file":  "outputs/googl_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/InternetAds",
        "ccy":          "USD",
        "description":  "Alphabet / Google",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.15,
        "news_keywords":     ["GOOGL", "Google", "Alphabet", "Sundar Pichai", "Waymo",
                              "Google Cloud", "Gemini", "YouTube", "antitrust", "DOJ",
                              "search monopoly", "Android", "ad revenue", "DMA", "EU fine"],
        "news_primary_terms":["GOOGL", "GOOG", "Google", "Alphabet"],
        # SEC / 财报
        "sec_cik":           "0001652044",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "MSFT": {
        "ticker":       "MSFT",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "MSFT"],
        "output_file":  "outputs/msft_api_output.txt",
        "prompt_file":  "outputs/msft_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/CloudAI",
        "ccy":          "USD",
        "description":  "Microsoft",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.15,
        "news_keywords":     ["MSFT", "Microsoft", "Satya Nadella", "Azure", "Copilot",
                              "OpenAI", "Teams", "Activision", "GitHub", "Office 365",
                              "Windows", "LinkedIn", "cloud revenue", "Bing"],
        "news_primary_terms":["MSFT", "Microsoft"],
        # SEC / 财报
        "sec_cik":           "0000789019",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "NVDA": {
        "ticker":       "NVDA",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "NVDA"],
        "output_file":  "outputs/nvda_api_output.txt",
        "prompt_file":  "outputs/nvda_prompt_output.txt",
        "backtest_dir": "nvda_portfolio_backtest",
        "sector":       "Technology/Semiconductors",
        "ccy":          "USD",
        "description":  "NVIDIA",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.20,
        "news_keywords":     ["NVDA", "Nvidia", "Jensen Huang", "GPU", "H100", "H200",
                              "Blackwell", "CUDA", "data center", "AI chip",
                              "export controls", "Hopper", "NVLink"],
        "news_primary_terms":["NVDA", "Nvidia"],
        # SEC / 财报
        "sec_cik":           "0001045810",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "AAPL": {
        "ticker":       "AAPL",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "AAPL"],
        "output_file":  "outputs/aapl_api_output.txt",
        "prompt_file":  "outputs/aapl_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Consumer",
        "ccy":          "USD",
        "description":  "Apple",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.15,
        "news_keywords":     ["AAPL", "Apple", "Tim Cook", "iPhone", "Mac",
                              "Vision Pro", "App Store", "India manufacturing",
                              "services revenue", "buyback"],
        "news_primary_terms":["AAPL", "Apple"],
        # SEC / 财报
        "sec_cik":           "0000320193",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "META": {
        "ticker":       "META",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "META"],
        "output_file":  "outputs/meta_api_output.txt",
        "prompt_file":  "outputs/meta_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/SocialMedia",
        "ccy":          "USD",
        "description":  "Meta Platforms",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.10,
        "news_keywords":     ["META", "Meta", "Mark Zuckerberg", "Facebook", "Instagram",
                              "WhatsApp", "Llama", "Ray-Ban", "Reality Labs", "Threads",
                              "ad revenue", "CPM", "Reels", "metaverse"],
        "news_primary_terms":["META", "Meta", "Facebook", "Instagram"],
        # SEC / 财报
        "sec_cik":           "0001326801",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "AMZN": {
        "ticker":       "AMZN",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "AMZN"],
        "output_file":  "outputs/amzn_api_output.txt",
        "prompt_file":  "outputs/amzn_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/CloudEcommerce",
        "ccy":          "USD",
        "description":  "Amazon",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.10,
        "news_keywords":     ["AMZN", "Amazon", "Andy Jassy", "AWS", "Bedrock", "Prime",
                              "Alexa", "Amazon Web Services", "cloud revenue",
                              "e-commerce", "fulfillment", "advertising revenue", "Anthropic"],
        "news_primary_terms":["AMZN", "Amazon", "AWS"],
        # SEC / 财报
        "sec_cik":           "0001018724",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },

    # ─────────────── 半导体 / 科技硬件 扩展 ───────────────
    "TSLA": {
        "ticker":       "TSLA",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "TSLA"],
        "output_file":  "outputs/tsla_api_output.txt",
        "prompt_file":  "outputs/tsla_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/EV",
        "ccy":          "USD",
        "description":  "Tesla",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.10,
        "news_keywords":     ["TSLA", "Tesla", "Elon Musk", "Cybertruck", "Full Self-Driving",
                              "FSD", "Giga", "Optimus", "energy storage", "Megapack",
                              "EV delivery", "Model 3", "Model Y", "Model S", "Roadster"],
        "news_primary_terms":["TSLA", "Tesla"],
        # SEC / 财报
        "sec_cik":           "0001318605",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "AMD": {
        "ticker":       "AMD",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "AMD"],
        "output_file":  "outputs/amd_api_output.txt",
        "prompt_file":  "outputs/amd_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Semiconductors",
        "ccy":          "USD",
        "description":  "Advanced Micro Devices",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.10,
        "news_keywords":     ["AMD", "Advanced Micro Devices", "Lisa Su", "MI300", "MI350",
                              "EPYC", "Ryzen", "Instinct", "ROCm", "AI accelerator",
                              "data center GPU", "server CPU"],
        "news_primary_terms":["AMD", "Advanced Micro Devices", "Lisa Su"],
        # SEC / 财报
        "sec_cik":           "0000002488",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "QCOM": {
        "ticker":       "QCOM",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "QCOM"],
        "output_file":  "outputs/qcom_api_output.txt",
        "prompt_file":  "outputs/qcom_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Semiconductors",
        "ccy":          "USD",
        "description":  "Qualcomm",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.08,
        "news_keywords":     ["QCOM", "Qualcomm", "Snapdragon", "X Elite", "modem",
                              "5G", "IoT", "automotive chip", "Oryon", "licensing",
                              "Apple modem", "Samsung"],
        "news_primary_terms":["QCOM", "Qualcomm", "Snapdragon"],
        # SEC / 财报
        "sec_cik":           "0000804328",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "INTC": {
        "ticker":       "INTC",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "INTC"],
        "output_file":  "outputs/intc_api_output.txt",
        "prompt_file":  "outputs/intc_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Semiconductors",
        "ccy":          "USD",
        "description":  "Intel",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.08,
        "news_keywords":     ["INTC", "Intel", "Pat Gelsinger", "Gaudi", "Arc GPU",
                              "18A process", "foundry", "IFS", "Core Ultra", "Xeon",
                              "turnaround", "restructuring", "fab"],
        "news_primary_terms":["INTC", "Intel"],
        # SEC / 财报
        "sec_cik":           "0000050863",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },
    "DELL": {
        "ticker":       "DELL",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "DELL"],
        "output_file":  "outputs/dell_api_output.txt",
        "prompt_file":  "outputs/dell_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/Infrastructure",
        "ccy":          "USD",
        "description":  "Dell Technologies",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.07,
        "news_keywords":     ["DELL", "Dell Technologies", "Dell", "PowerEdge", "ISG",
                              "AI server", "NVIDIA DGX", "storage", "PC shipment",
                              "infrastructure solutions", "Michael Dell"],
        "news_primary_terms":["DELL", "Dell Technologies", "Dell"],
        # SEC / 财报
        "sec_cik":           "0001571996",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },

    # ─────────────── 能源 ───────────────
    "XOM": {
        "ticker":       "XOM",
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "XOM"],
        "output_file":  "outputs/xom_api_output.txt",
        "prompt_file":  "outputs/xom_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Energy/OilGas",
        "ccy":          "USD",
        "description":  "ExxonMobil",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.06,
        "news_keywords":     ["XOM", "ExxonMobil", "Exxon Mobil", "Exxon", "Darren Woods",
                              "Pioneer acquisition", "Permian Basin", "LNG", "refining",
                              "carbon capture", "oil major", "dividend"],
        "news_primary_terms":["XOM", "ExxonMobil", "Exxon"],
        # SEC / 财报
        "sec_cik":           "0000034088",
        "earnings_tracking": True,
        "insider_tracking":  True,
    },

    # ─────────────── 存储半导体（韩国）───────────────
    "HYNIX": {
        "ticker":       "000660.KS",    # SK하이닉스 KOSPI，货币 KRW
        "type":         "equity",
        "script":       "tech_stock_analysis.py",
        "script_args":  ["--ticker", "000660.KS"],
        "output_file":  "outputs/hynix_api_output.txt",
        "prompt_file":  "outputs/hynix_prompt_output.txt",
        "backtest_dir": None,
        "sector":       "Technology/MemorySemiconductors",
        "ccy":          "KRW",
        "description":  "SK 海力士（HBM / DRAM / NAND）",
        # 每日任务
        "daily_scan":       True,
        "daily_extra_args": [],
        # 新闻 RAG
        "rag_weight":        0.08,
        "news_keywords":     ["SK Hynix", "海力士", "HBM3E", "HBM4", "DRAM", "NAND",
                              "memory chip", "high bandwidth memory", "AI memory",
                              "NVIDIA HBM", "Samsung memory competition"],
        "news_primary_terms":["SK Hynix", "Hynix", "HBM", "海力士"],
        # SEC / 财报（韩国上市公司无 SEC CIK）
        "sec_cik":           None,
        "earnings_tracking": False,
        "insider_tracking":  False,
    },
}

# ── 预定义扫描分组（market_scan.py 中使用 --group 选择）──
SCAN_GROUPS: dict[str, list[str]] = {
    "all":        list(ASSET_UNIVERSE.keys()),
    "tech":       ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN",
                   "TSLA", "AMD", "QCOM", "INTC", "DELL"],
    "semis":      ["NVDA", "AMD", "QCOM", "INTC", "HYNIX"],    # 半导体板块
    "commodities":["GOLD", "SILVER", "COPPER", "RARE_EARTH", "OIL"],
    "energy":     ["OIL", "XOM"],
    "crypto":     ["BTC"],
    "metals":     ["GOLD", "SILVER", "COPPER", "RARE_EARTH"],
    "quick":      ["GOLD", "BTC", "NVDA", "MSFT"],
    "mag7":       ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN", "TSLA"],
}


# ──────────────────────────────────────────────────────────────────
# 辅助函数 — 供各模块调用，无需手动过滤 ASSET_UNIVERSE
# ──────────────────────────────────────────────────────────────────

def get_daily_assets() -> list[tuple[str, dict]]:
    """
    返回 daily_scan=True 的资产列表，顺序保持注册顺序。
    run_daily.sh 用此函数动态生成分析命令，无需硬编码。

    Returns: [(asset_key, asset_cfg), ...]
    """
    return [
        (key, cfg)
        for key, cfg in ASSET_UNIVERSE.items()
        if cfg.get("daily_scan", False)
    ]


def get_rag_portfolio() -> dict:
    """
    生成 RAG 系统 portfolio.json 格式的 dict，供 news_signal_bridge.py 使用。
    rag_weight=0 的资产不纳入（相当于禁用 RAG 监控）。

    Returns: { ticker: { asset_type, keywords, weight, sector }, ... }
    """
    portfolio = {}
    for key, cfg in ASSET_UNIVERSE.items():
        weight = cfg.get("rag_weight", 0.0)
        if weight <= 0:
            continue
        portfolio[cfg["ticker"]] = {
            "asset_type": cfg["type"],
            "keywords":   cfg.get("news_keywords", [cfg["ticker"]]),
            "weight":     weight,
            "sector":     cfg.get("sector", "unknown").split("/")[0].lower(),
        }
    return portfolio


def get_sec_tracked_assets() -> dict[str, str]:
    """
    返回需要 SEC 8-K / Form 4 监控的资产 {ticker: cik}。
    sec_edgar.py COMPANY_8K 和 form4.py TICKERS 应从此处读取。

    Returns: { "NVDA": "0001045810", ... }
    """
    result = {}
    for key, cfg in ASSET_UNIVERSE.items():
        cik = cfg.get("sec_cik")
        if cik and cfg.get("insider_tracking", False):
            result[cfg["ticker"]] = cik
    return result


def get_earnings_tracked_tickers() -> list[str]:
    """
    返回需要 Polygon 季报财务采集的 ticker 列表。
    polygon.py EARNINGS_TICKERS 应从此处读取。

    Returns: ["NVDA", "AAPL", "GOOGL", "MSFT", "META", "AMZN"]
    """
    return [
        cfg["ticker"]
        for cfg in ASSET_UNIVERSE.values()
        if cfg.get("earnings_tracking", False)
    ]


def get_news_keywords(ticker: str) -> list[str]:
    """
    返回指定 ticker 的新闻检索关键词列表。
    news_signal_bridge.py 的 _ASSET_KEYWORDS 应从此处读取。

    Returns: keyword list，找不到则返回 [ticker]
    """
    for cfg in ASSET_UNIVERSE.values():
        if cfg["ticker"].upper() == ticker.upper():
            return cfg.get("news_keywords", [ticker])
    return [ticker]


def get_news_primary_terms(ticker: str) -> list[str]:
    """
    返回指定 ticker 的主标识符列表（整词匹配，防止 "Meta"≠"metal" 误判）。
    news_signal_bridge.py 的 _ASSET_PRIMARY_TERMS 应从此处读取。

    Returns: primary terms list，找不到则返回 [ticker]
    """
    for cfg in ASSET_UNIVERSE.values():
        if cfg["ticker"].upper() == ticker.upper():
            return cfg.get("news_primary_terms", [ticker])
    return [ticker]
