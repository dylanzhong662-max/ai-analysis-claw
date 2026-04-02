import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { fetchAllSignals, refreshSignal, fetchLatestScan, runScan } from "../api/signals"
import TopBar from "../components/layout/TopBar"
import { RefreshCw, Play, ChevronDown, ChevronUp } from "lucide-react"
import type { Signal } from "../types"

const SCAN_GROUPS = ["quick", "tech", "metals", "commodities", "all"]
const SCAN_GROUP_LABELS: Record<string, string> = {
  quick: "快速", tech: "科技股", metals: "贵金属", commodities: "大宗商品", all: "全部"
}

function ActionBadge({ action }: { action?: string | null }) {
  if (!action) return <span className="text-muted text-xs">—</span>
  const map: Record<string, string> = {
    long:     "bg-green-500/20 text-green-400",
    short:    "bg-red-500/20 text-red-400",
    no_trade: "bg-slate-700 text-slate-400",
  }
  const labels: Record<string, string> = { long: "做多", short: "做空", no_trade: "观望" }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${map[action] ?? "bg-slate-700 text-slate-400"}`}>
      {labels[action] ?? action}
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
      <span className="text-xs text-muted tabular-nums">{score.toFixed(2)}</span>
    </div>
  )
}

function SignalCard({ sig, onRefresh }: { sig: Signal; onRefresh: (a: string) => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="font-semibold text-slate-200">{sig.asset}</h3>
            {sig.regime && <p className="text-xs text-muted mt-0.5">{sig.regime}</p>}
          </div>
          <div className="flex items-center gap-2">
            <ActionBadge action={sig.action} />
            <button onClick={() => onRefresh(sig.asset)}
              className="p-1 text-muted hover:text-blue-400 hover:bg-blue-500/10 rounded transition-colors" title="刷新分析">
              <RefreshCw size={12} />
            </button>
          </div>
        </div>

        <BiasBar score={sig.bias_score} />

        {sig.entry_zone && (
          <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
            <div>
              <p className="text-muted">入场区间</p>
              <p className="text-slate-300">{sig.entry_zone}</p>
            </div>
            {sig.stop_loss && (
              <div>
                <p className="text-muted">止损</p>
                <p className="text-red-400">{sig.stop_loss}</p>
              </div>
            )}
            {sig.profit_target && (
              <div>
                <p className="text-muted">目标</p>
                <p className="text-green-400">{sig.profit_target}</p>
              </div>
            )}
          </div>
        )}

        {sig.risk_reward_ratio && (
          <div className="mt-2 text-xs">
            <span className="text-muted">R:R </span>
            <span className={sig.risk_reward_ratio >= 2 ? "text-green-400" : "text-yellow-400"}>
              {sig.risk_reward_ratio.toFixed(1)}
            </span>
            {sig.estimated_holding_weeks && (
              <span className="text-muted ml-3">持仓约 {sig.estimated_holding_weeks} 周</span>
            )}
          </div>
        )}

        {sig.analysis_date && (
          <p className="text-xs text-muted mt-2">{sig.analysis_date}</p>
        )}
      </div>

      {sig.justification && (
        <>
          <button onClick={() => setExpanded((e) => !e)}
            className="w-full flex items-center justify-center gap-1 py-2 text-xs text-muted
                       border-t border-border hover:bg-white/3 transition-colors">
            {expanded ? <><ChevronUp size={12} /> 收起分析</> : <><ChevronDown size={12} /> 查看完整分析</>}
          </button>
          {expanded && (
            <div className="px-4 pb-4 text-xs text-slate-400 leading-relaxed border-t border-border pt-3">
              {typeof sig.justification === "string"
                ? sig.justification
                : JSON.stringify(sig.justification, null, 2)}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function MarketScan() {
  const qc = useQueryClient()
  const [scanGroup, setScanGroup] = useState("quick")
  const [scanMsg, setScanMsg] = useState("")

  const { data: signals } = useQuery({
    queryKey: ["signals"], queryFn: fetchAllSignals, refetchInterval: 60_000,
  })
  const { data: scanData } = useQuery({
    queryKey: ["scan"], queryFn: fetchLatestScan,
  })

  const refreshMut = useMutation({
    mutationFn: refreshSignal,
    onSuccess: (d) => {
      setScanMsg(d.message)
      setTimeout(() => setScanMsg(""), 5000)
    },
  })

  const scanMut = useMutation({
    mutationFn: () => runScan(scanGroup),
    onSuccess: (d) => {
      setScanMsg(d.message)
      setTimeout(() => {
        setScanMsg("")
        qc.invalidateQueries({ queryKey: ["scan"] })
        qc.invalidateQueries({ queryKey: ["signals"] })
      }, 5000)
    },
  })

  const activeSignals = signals?.filter((s) => s.action && s.action !== "no_trade") ?? []
  const noTradeSignals = signals?.filter((s) => !s.action || s.action === "no_trade") ?? []

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="市场扫描" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Control Bar */}
        <div className="flex flex-wrap gap-3 items-center">
          <div className="flex gap-1 bg-surface border border-border rounded-lg p-1">
            {SCAN_GROUPS.map((g) => (
              <button key={g} onClick={() => setScanGroup(g)}
                className={`px-3 py-1.5 rounded text-sm transition-colors
                  ${scanGroup === g ? "bg-blue-600 text-white" : "text-muted hover:text-slate-200"}`}>
                {SCAN_GROUP_LABELS[g]}
              </button>
            ))}
          </div>
          <button onClick={() => scanMut.mutate()} disabled={scanMut.isPending}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50">
            <Play size={13} />
            {scanMut.isPending ? "扫描中…" : "运行扫描"}
          </button>
          {scanMsg && <span className="text-sm text-blue-400">{scanMsg}</span>}
        </div>

        {/* Scan Summary from market_scan_output.json */}
        {scanData && !scanData.message && (
          <div className="bg-surface border border-border rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-medium text-slate-200">最新扫描汇总</h2>
              <span className="text-xs text-muted">{scanData.scan_date}</span>
            </div>

            {scanData.macro_themes?.length > 0 && (
              <div>
                <p className="text-xs text-muted font-medium uppercase tracking-wider mb-2">宏观主题</p>
                <div className="flex flex-wrap gap-2">
                  {scanData.macro_themes.map((t: any, i: number) => (
                    <span key={i} className="px-3 py-1 bg-bg border border-border rounded-full text-xs text-slate-300">
                      {t.theme}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {scanData.correlation_risks?.length > 0 && (
              <div>
                <p className="text-xs text-muted font-medium uppercase tracking-wider mb-2">相关性风险</p>
                {scanData.correlation_risks.map((r: any, i: number) => (
                  <p key={i} className="text-xs text-yellow-400">⚠ {r.risk_note}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Active Signals */}
        {activeSignals.length > 0 && (
          <div>
            <h2 className="text-xs text-muted font-medium uppercase tracking-wider mb-3">
              活跃信号 ({activeSignals.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {activeSignals.map((s) => (
                <SignalCard key={s.asset} sig={s} onRefresh={(a) => refreshMut.mutate(a)} />
              ))}
            </div>
          </div>
        )}

        {/* No Trade Signals */}
        {noTradeSignals.length > 0 && (
          <div>
            <h2 className="text-xs text-muted font-medium uppercase tracking-wider mb-3">
              观望 / 暂无数据 ({noTradeSignals.length})
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
              {noTradeSignals.map((s) => (
                <div key={s.asset} className="bg-surface border border-border rounded-xl p-3 flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm text-slate-400">{s.asset}</p>
                    {s.analysis_date && <p className="text-xs text-muted">{s.analysis_date.slice(0,10)}</p>}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <ActionBadge action={s.action} />
                    <button onClick={() => refreshMut.mutate(s.asset)}
                      className="p-1 text-muted hover:text-blue-400 hover:bg-blue-500/10 rounded transition-colors">
                      <RefreshCw size={11} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
