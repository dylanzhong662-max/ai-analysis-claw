import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { fetchTrades, fetchTradeStats, createTrade, deleteTrade } from "../api/trades"
import TopBar from "../components/layout/TopBar"
import { Plus, X, Trash2 } from "lucide-react"
import type { TradeCreate } from "../types"

const ASSETS = [
  "GOLD","SILVER","COPPER","RARE_EARTH","OIL",
  "BTC","GOOGL","MSFT","NVDA","AAPL","META","AMZN",
]
const TICKER_MAP: Record<string, string> = {
  GOLD:"GC=F", SILVER:"SLV", COPPER:"COPX", RARE_EARTH:"REMX", OIL:"USO",
  BTC:"BTC-USD", GOOGL:"GOOGL", MSFT:"MSFT", NVDA:"NVDA",
  AAPL:"AAPL", META:"META", AMZN:"AMZN",
}

const EXIT_REASONS = [
  { v: "manual",          l: "手动平仓" },
  { v: "stop_loss",       l: "触发止损" },
  { v: "take_profit",     l: "目标达到" },
  { v: "signal_reversed", l: "信号反转" },
  { v: "timeout",         l: "超时" },
]
const EXIT_REASON_LABELS: Record<string, string> = Object.fromEntries(EXIT_REASONS.map((r) => [r.v, r.l]))

function ManualTradeModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [form, setForm] = useState<Partial<TradeCreate>>({
    direction: "long", exit_reason: "manual",
    entry_date: new Date().toISOString().slice(0, 10),
    exit_date: new Date().toISOString().slice(0, 10),
  })
  const [error, setError] = useState("")

  const set = (k: string, v: unknown) => setForm((f) => ({ ...f, [k]: v }))

  const mut = useMutation({
    mutationFn: createTrade,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trades"] })
      qc.invalidateQueries({ queryKey: ["trade-stats"] })
      onClose()
    },
    onError: (e: any) => setError(e.response?.data?.detail ?? "创建失败"),
  })

  const cost = (form.entry_price ?? 0) * (form.quantity ?? 0)
  const pnl = form.entry_price && form.exit_price && form.quantity
    ? form.direction === "long"
      ? (Number(form.exit_price) - Number(form.entry_price)) / Number(form.entry_price) * 100
      : (Number(form.entry_price) - Number(form.exit_price)) / Number(form.entry_price) * 100
    : null

  const submit = () => {
    if (!form.asset || !form.entry_price || !form.exit_price || !form.quantity) {
      setError("请填写必填字段")
      return
    }
    mut.mutate({
      asset: form.asset!,
      ticker: TICKER_MAP[form.asset!] ?? form.asset!,
      direction: form.direction as "long" | "short",
      entry_price: Number(form.entry_price),
      entry_date: form.entry_date!,
      exit_price: Number(form.exit_price),
      exit_date: form.exit_date!,
      quantity: Number(form.quantity),
      cost_basis_usd: cost,
      exit_reason: form.exit_reason,
      notes: form.notes,
    })
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-lg">
        <div className="flex items-center justify-between p-5 border-b border-border">
          <h2 className="font-semibold text-slate-200">手动录入交易记录</h2>
          <button onClick={onClose} className="text-muted hover:text-slate-200"><X size={18} /></button>
        </div>
        <div className="p-5 space-y-4">
          {error && <p className="text-red-400 text-sm bg-red-500/10 px-3 py-2 rounded-lg">{error}</p>}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">资产 *</label>
              <select value={form.asset ?? ""} onChange={(e) => set("asset", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
                <option value="">选择资产</option>
                {ASSETS.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">方向 *</label>
              <div className="flex gap-2">
                {(["long","short"] as const).map((d) => (
                  <button key={d} onClick={() => set("direction", d)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors
                      ${form.direction === d
                        ? d === "long" ? "bg-green-500/20 border-green-500/50 text-green-400"
                                       : "bg-red-500/20 border-red-500/50 text-red-400"
                        : "border-border text-muted"}`}>
                    {d === "long" ? "做多" : "做空"}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">入场价 *</label>
              <input type="number" value={form.entry_price ?? ""} onChange={(e) => set("entry_price", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">出场价 *</label>
              <input type="number" value={form.exit_price ?? ""} onChange={(e) => set("exit_price", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">数量 *</label>
              <input type="number" value={form.quantity ?? ""} onChange={(e) => set("quantity", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">平仓原因</label>
              <select value={form.exit_reason} onChange={(e) => set("exit_reason", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
                {EXIT_REASONS.map((r) => <option key={r.v} value={r.v}>{r.l}</option>)}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">入场日期</label>
              <input type="date" value={form.entry_date ?? ""} onChange={(e) => set("entry_date", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">出场日期</label>
              <input type="date" value={form.exit_date ?? ""} onChange={(e) => set("exit_date", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1 block">备注</label>
            <input type="text" value={form.notes ?? ""} onChange={(e) => set("notes", e.target.value)}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
          </div>

          {pnl != null && (
            <div className="bg-bg rounded-lg p-3 text-sm text-center">
              <span className="text-muted">盈亏：</span>
              <span className={`font-bold text-lg ml-2 ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}%
              </span>
            </div>
          )}
        </div>
        <div className="flex gap-3 p-5 pt-0">
          <button onClick={onClose} className="flex-1 py-2.5 border border-border rounded-lg text-muted text-sm hover:bg-white/5">取消</button>
          <button onClick={submit} disabled={mut.isPending}
            className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white text-sm font-medium disabled:opacity-50">
            {mut.isPending ? "保存中…" : "保存记录"}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function TradeHistory() {
  const qc = useQueryClient()
  const [showModal, setShowModal] = useState(false)
  const [filters, setFilters] = useState({ asset: "", direction: "", exit_reason: "", page: 1 })

  const { data: stats } = useQuery({ queryKey: ["trade-stats"], queryFn: fetchTradeStats })
  const { data, isLoading } = useQuery({
    queryKey: ["trades", filters],
    queryFn: () => fetchTrades({
      asset: filters.asset || undefined,
      direction: filters.direction || undefined,
      exit_reason: filters.exit_reason || undefined,
      page: filters.page,
    }),
  })

  const delMut = useMutation({
    mutationFn: deleteTrade,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trades"] })
      qc.invalidateQueries({ queryKey: ["trade-stats"] })
    },
  })

  const setFilter = (k: string, v: string) => setFilters((f) => ({ ...f, [k]: v, page: 1 }))

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {showModal && <ManualTradeModal onClose={() => setShowModal(false)} />}
      <TopBar title="交易记录" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { l: "总交易", v: stats?.total ?? 0, unit: "笔" },
            { l: "胜率",   v: stats ? `${stats.win_rate}%` : "—", sub: `${stats?.wins ?? 0}胜/${stats?.losses ?? 0}负` },
            { l: "盈利因子", v: stats?.profit_factor?.toFixed(2) ?? "—" },
            { l: "总盈亏",  v: stats ? `$${stats.total_realized_pnl.toLocaleString()}` : "—",
              color: (stats?.total_realized_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400" },
          ].map((c, i) => (
            <div key={i} className="bg-surface border border-border rounded-xl p-4">
              <p className="text-muted text-xs mb-1">{c.l}</p>
              <p className={`text-xl font-bold ${(c as any).color ?? "text-slate-100"}`}>
                {c.v}{(c as any).unit ?? ""}
              </p>
              {(c as any).sub && <p className="text-xs text-muted mt-1">{(c as any).sub}</p>}
            </div>
          ))}
        </div>

        {/* Filters + Add Button */}
        <div className="flex flex-wrap gap-3 items-center">
          <select value={filters.asset} onChange={(e) => setFilter("asset", e.target.value)}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
            <option value="">全部资产</option>
            {ASSETS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <select value={filters.direction} onChange={(e) => setFilter("direction", e.target.value)}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
            <option value="">全部方向</option>
            <option value="long">做多</option>
            <option value="short">做空</option>
          </select>
          <select value={filters.exit_reason} onChange={(e) => setFilter("exit_reason", e.target.value)}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
            <option value="">全部原因</option>
            {EXIT_REASONS.map((r) => <option key={r.v} value={r.v}>{r.l}</option>)}
          </select>
          <div className="flex-1" />
          <button onClick={() => setShowModal(true)}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
            <Plus size={14} />
            手动录入
          </button>
        </div>

        {/* Table */}
        <div className="bg-surface border border-border rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-muted uppercase tracking-wider">
                <th className="text-left px-4 py-3">资产</th>
                <th className="text-left px-4 py-3">方向</th>
                <th className="text-right px-4 py-3">入场价</th>
                <th className="text-right px-4 py-3">出场价</th>
                <th className="text-right px-4 py-3">盈亏%</th>
                <th className="text-right px-4 py-3">盈亏$</th>
                <th className="text-center px-4 py-3">持仓天</th>
                <th className="text-left px-4 py-3">原因</th>
                <th className="text-left px-4 py-3">出场日期</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={10} className="text-center py-10 text-muted">加载中…</td></tr>
              ) : data?.trades.length === 0 ? (
                <tr><td colSpan={10} className="text-center py-10 text-muted">暂无记录</td></tr>
              ) : (
                data?.trades.map((t) => {
                  const win = (t.realized_pnl_pct ?? 0) > 0
                  return (
                    <tr key={t.id} className="border-b border-border hover:bg-white/2 transition-colors">
                      <td className="px-4 py-3 font-medium text-slate-200">{t.asset}</td>
                      <td className="px-4 py-3">
                        <span className={t.direction === "long" ? "text-green-400" : "text-red-400"}>
                          {t.direction === "long" ? "多" : "空"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-slate-300">{t.entry_price}</td>
                      <td className="px-4 py-3 text-right text-slate-300">{t.exit_price}</td>
                      <td className={`px-4 py-3 text-right font-medium ${win ? "text-green-400" : "text-red-400"}`}>
                        {win ? "+" : ""}{t.realized_pnl_pct?.toFixed(2) ?? "—"}%
                      </td>
                      <td className={`px-4 py-3 text-right ${win ? "text-green-400" : "text-red-400"}`}>
                        {t.realized_pnl_usd != null ? `${win ? "+" : ""}$${t.realized_pnl_usd.toFixed(0)}` : "—"}
                      </td>
                      <td className="px-4 py-3 text-center text-muted">{t.holding_days ?? "—"}</td>
                      <td className="px-4 py-3 text-muted">{EXIT_REASON_LABELS[t.exit_reason ?? ""] ?? t.exit_reason ?? "—"}</td>
                      <td className="px-4 py-3 text-muted">{t.exit_date?.slice(0, 10)}</td>
                      <td className="px-4 py-3">
                        <button onClick={() => { if (confirm("确定删除此记录？")) delMut.mutate(t.id) }}
                          className="p-1 text-muted hover:text-red-400 hover:bg-red-500/10 rounded transition-colors">
                          <Trash2 size={13} />
                        </button>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data && data.pages > 1 && (
          <div className="flex items-center justify-center gap-3 text-sm">
            <button disabled={filters.page <= 1}
              onClick={() => setFilters((f) => ({ ...f, page: f.page - 1 }))}
              className="px-3 py-1.5 border border-border rounded-lg text-muted hover:text-slate-200 hover:bg-white/5 disabled:opacity-30">
              上一页
            </button>
            <span className="text-muted">第 {filters.page} / {data.pages} 页 · 共 {data.total} 条</span>
            <button disabled={filters.page >= data.pages}
              onClick={() => setFilters((f) => ({ ...f, page: f.page + 1 }))}
              className="px-3 py-1.5 border border-border rounded-lg text-muted hover:text-slate-200 hover:bg-white/5 disabled:opacity-30">
              下一页
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
