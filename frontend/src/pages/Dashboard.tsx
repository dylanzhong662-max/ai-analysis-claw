import { useQuery } from "@tanstack/react-query"
import { fetchSummary, fetchMacro } from "../api/dashboard"
import { fetchAllSignals } from "../api/signals"
import TopBar from "../components/layout/TopBar"
import {
  AlertTriangle, CheckCircle,
  Activity, Zap, Eye
} from "lucide-react"
import type { Signal } from "../types"

function fmt(n?: number | null, digits = 2) {
  if (n == null) return "—"
  return n.toFixed(digits)
}

function PnlBadge({ val }: { val?: number | null }) {
  if (val == null) return <span className="text-muted">—</span>
  const pos = val >= 0
  return (
    <span className={pos ? "text-green-400" : "text-red-400"}>
      {pos ? "+" : ""}{val.toFixed(2)}%
    </span>
  )
}

function BiasBar({ score }: { score?: number | null }) {
  if (score == null) return <span className="text-muted text-xs">—</span>
  const pct = Math.round(score * 100)
  const color = score >= 0.65 ? "bg-green-500" : score >= 0.5 ? "bg-yellow-500" : "bg-slate-600"
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-border rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-muted w-8 text-right">{fmt(score, 2)}</span>
    </div>
  )
}

function ActionBadge({ action }: { action?: string | null }) {
  if (!action) return <span className="text-muted text-xs">—</span>
  const map: Record<string, string> = {
    long: "bg-green-500/20 text-green-400",
    short: "bg-red-500/20 text-red-400",
    no_trade: "bg-slate-700 text-slate-400",
  }
  const labels: Record<string, string> = { long: "做多", short: "做空", no_trade: "观望" }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${map[action] ?? "bg-slate-700 text-slate-400"}`}>
      {labels[action] ?? action}
    </span>
  )
}

function SeverityIcon({ severity }: { severity: string }) {
  if (severity === "error") return <AlertTriangle size={14} className="text-red-400" />
  if (severity === "success") return <CheckCircle size={14} className="text-green-400" />
  return <AlertTriangle size={14} className="text-yellow-400" />
}

const ASSET_GROUPS = [
  { label: "贵金属 / 大宗商品", keys: ["GOLD", "SILVER", "COPPER", "RARE_EARTH", "OIL"] },
  { label: "加密货币", keys: ["BTC"] },
  { label: "科技股", keys: ["GOOGL", "MSFT", "NVDA", "AAPL", "META", "AMZN"] },
]

function strengthColor(s: string) {
  if (s === "Strong") return "text-green-400"
  if (s === "Weak") return "text-red-400"
  return "text-yellow-400"
}

export default function Dashboard() {
  const { data: summary } = useQuery({ queryKey: ["summary"], queryFn: fetchSummary, refetchInterval: 60_000 })
  const { data: macro }   = useQuery({ queryKey: ["macro"],   queryFn: fetchMacro,   refetchInterval: 60_000 })
  const { data: signals } = useQuery({ queryKey: ["signals"], queryFn: fetchAllSignals, refetchInterval: 60_000 })

  const sigMap: Record<string, Signal> = {}
  signals?.forEach((s) => { sigMap[s.asset] = s })

  const sentimentColor = macro?.sentiment?.includes("Off")
    ? "text-red-400" : macro?.sentiment?.includes("On") ? "text-green-400" : "text-yellow-400"

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="仪表盘" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">

        {/* Alert Banner */}
        {(summary?.alerts?.length ?? 0) > 0 && (
          <div className="space-y-2">
            {summary!.alerts.map((a, i) => (
              <div key={i} className={`flex items-center gap-3 px-4 py-3 rounded-lg border text-sm
                ${a.severity === "error"   ? "bg-red-500/10 border-red-500/30 text-red-300" :
                  a.severity === "success" ? "bg-green-500/10 border-green-500/30 text-green-300" :
                                             "bg-yellow-500/10 border-yellow-500/30 text-yellow-300"}`}>
                <SeverityIcon severity={a.severity} />
                <span className="font-medium">{a.asset}</span>
                <span className="text-slate-300">—</span>
                <span>{a.message}</span>
              </div>
            ))}
          </div>
        )}

        {/* KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-surface border border-border rounded-xl p-4">
            <p className="text-muted text-xs mb-1">持仓总值</p>
            <p className="text-xl font-bold text-slate-100">
              ${(summary?.portfolio_value ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}
            </p>
            <p className="text-xs text-muted mt-1">{summary?.open_positions ?? 0} 个持仓</p>
          </div>
          <div className="bg-surface border border-border rounded-xl p-4">
            <p className="text-muted text-xs mb-1">未实现盈亏</p>
            <p className="text-xl font-bold">
              <PnlBadge val={summary?.total_unrealized_pnl_pct} />
            </p>
            <p className="text-xs text-muted mt-1">
              {summary?.total_unrealized_pnl_usd != null
                ? `$${summary.total_unrealized_pnl_usd.toLocaleString()}`
                : "—"}
            </p>
          </div>
          <div className="bg-surface border border-border rounded-xl p-4">
            <p className="text-muted text-xs mb-1">活跃信号</p>
            <p className="text-xl font-bold text-blue-400">{summary?.active_signals ?? "—"}</p>
            <p className="text-xs text-muted mt-1">共 12 个标的</p>
          </div>
          <div className="bg-surface border border-border rounded-xl p-4">
            <p className="text-muted text-xs mb-1">需关注</p>
            <p className={`text-xl font-bold ${(summary?.needs_action ?? 0) > 0 ? "text-red-400" : "text-green-400"}`}>
              {summary?.needs_action ?? 0} 项
            </p>
            <p className="text-xs text-muted mt-1">止损/目标/反转</p>
          </div>
        </div>

        {/* Macro Bar */}
        <div className="bg-surface border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={14} className="text-muted" />
            <span className="text-xs text-muted font-medium uppercase tracking-wider">宏观环境</span>
          </div>
          <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            <div>
              <span className="text-muted text-xs">市场情绪  </span>
              <span className={`font-medium ${sentimentColor}`}>{macro?.sentiment ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted text-xs">VIX  </span>
              <span className={`font-medium ${(macro?.vix ?? 0) > 25 ? "text-red-400" : "text-green-400"}`}>
                {fmt(macro?.vix)}
              </span>
            </div>
            <div>
              <span className="text-muted text-xs">DXY  </span>
              <span className="font-medium text-slate-200">{fmt(macro?.dxy)}</span>
            </div>
            <div>
              <span className="text-muted text-xs">10Y  </span>
              <span className="font-medium text-slate-200">{fmt(macro?.ten_year)}%</span>
            </div>
            {macro?.btc_fear_greed != null && (
              <div>
                <span className="text-muted text-xs">BTC恐惧指数  </span>
                <span className={`font-medium ${macro.btc_fear_greed < 25 ? "text-red-400" : macro.btc_fear_greed > 75 ? "text-green-400" : "text-yellow-400"}`}>
                  {macro.btc_fear_greed} {macro.btc_fear_greed_label ? `(${macro.btc_fear_greed_label})` : ""}
                </span>
              </div>
            )}
            {macro?.last_scan_date && (
              <div>
                <span className="text-muted text-xs">最新扫描  </span>
                <span className="font-medium text-slate-300">{macro.last_scan_date}</span>
              </div>
            )}
          </div>

          {/* Sector Ranking */}
          {(macro?.sector_ranking?.length ?? 0) > 0 && (
            <div className="flex gap-4 mt-3 pt-3 border-t border-border">
              {macro!.sector_ranking.map((sr, i) => (
                <div key={i} className="text-xs">
                  <span className="text-muted">{sr.sector}：</span>
                  <span className={`font-medium ${strengthColor(sr.strength)}`}>{sr.strength}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Top Opportunities + Asset Matrix */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

          {/* Top Opportunities */}
          {(macro?.top_opportunities?.length ?? 0) > 0 && (
            <div className="bg-surface border border-border rounded-xl p-4 xl:col-span-1">
              <div className="flex items-center gap-2 mb-3">
                <Zap size={14} className="text-yellow-400" />
                <span className="text-xs text-muted font-medium uppercase tracking-wider">Top 机会</span>
              </div>
              <div className="space-y-3">
                {macro!.top_opportunities.slice(0, 5).map((op, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <span className="text-muted w-4">#{op.rank}</span>
                    <span className="font-medium text-slate-200 w-16">{op.asset}</span>
                    <ActionBadge action={op.action} />
                    <span className="text-muted text-xs flex-1">{fmt(op.bias_score, 2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Signal Matrix */}
          <div className={`bg-surface border border-border rounded-xl p-4 ${(macro?.top_opportunities?.length ?? 0) > 0 ? "xl:col-span-2" : "xl:col-span-3"}`}>
            <div className="flex items-center gap-2 mb-4">
              <Eye size={14} className="text-muted" />
              <span className="text-xs text-muted font-medium uppercase tracking-wider">信号总览</span>
            </div>
            <div className="space-y-4">
              {ASSET_GROUPS.map((g) => (
                <div key={g.label}>
                  <p className="text-xs text-muted mb-2 font-medium">{g.label}</p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {g.keys.map((key) => {
                      const sig = sigMap[key]
                      return (
                        <div key={key} className="bg-bg border border-border rounded-lg p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium text-sm text-slate-200">{key}</span>
                            <ActionBadge action={sig?.action} />
                          </div>
                          <BiasBar score={sig?.bias_score} />
                          {sig?.analysis_date && (
                            <p className="text-xs text-muted mt-1">{sig.analysis_date}</p>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
