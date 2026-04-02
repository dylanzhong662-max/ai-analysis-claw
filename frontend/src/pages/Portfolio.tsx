import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { fetchPositions, createPosition, deletePosition, closePosition } from "../api/portfolio"
import TopBar from "../components/layout/TopBar"
import { Plus, X, ChevronDown, ChevronUp, Trash2, LogOut } from "lucide-react"
import type { Position, PositionCreate, PositionClose } from "../types"

const ASSETS = [
  "GOLD","SILVER","COPPER","RARE_EARTH","OIL",
  "BTC","GOOGL","MSFT","NVDA","AAPL","META","AMZN",
]
const TICKER_MAP: Record<string, string> = {
  GOLD:"GC=F", SILVER:"SLV", COPPER:"COPX", RARE_EARTH:"REMX", OIL:"USO",
  BTC:"BTC-USD", GOOGL:"GOOGL", MSFT:"MSFT", NVDA:"NVDA",
  AAPL:"AAPL", META:"META", AMZN:"AMZN",
}

function PnlBadge({ val }: { val?: number | null }) {
  if (val == null) return <span className="text-muted">—</span>
  const pos = val >= 0
  return (
    <span className={`font-medium ${pos ? "text-green-400" : "text-red-400"}`}>
      {pos ? "+" : ""}{val.toFixed(2)}%
    </span>
  )
}

function StatusBadge({ status }: { status?: string | null }) {
  const map: Record<string, string> = {
    HOLD:           "bg-slate-700 text-slate-300",
    ADD:            "bg-green-500/20 text-green-400",
    REDUCE:         "bg-yellow-500/20 text-yellow-400",
    EXIT:           "bg-orange-500/20 text-orange-400",
    STOP_TRIGGERED: "bg-red-500/20 text-red-400",
    TARGET_REACHED: "bg-emerald-500/20 text-emerald-400",
    SIGNAL_REVERSED:"bg-purple-500/20 text-purple-400",
  }
  const labels: Record<string, string> = {
    HOLD:"持有", ADD:"加仓", REDUCE:"减仓", EXIT:"平仓",
    STOP_TRIGGERED:"止损触发", TARGET_REACHED:"目标达到", SIGNAL_REVERSED:"信号反转",
  }
  const s = status ?? "HOLD"
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${map[s] ?? "bg-slate-700 text-slate-300"}`}>
      {labels[s] ?? s}
    </span>
  )
}

// ─── Open Position Modal ──────────────────────────────────────────────────
function OpenModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [form, setForm] = useState<Partial<PositionCreate>>({
    direction: "long", trailing_stop: false,
    entry_date: new Date().toISOString().slice(0, 10),
  })
  const [error, setError] = useState("")

  const mut = useMutation({
    mutationFn: createPosition,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["positions"] }); onClose() },
    onError: (e: any) => setError(e.response?.data?.detail ?? "创建失败"),
  })

  const set = (k: string, v: unknown) => setForm((f) => ({ ...f, [k]: v }))

  const handleAsset = (asset: string) => {
    set("asset", asset)
    set("ticker", TICKER_MAP[asset] ?? asset)
  }

  const cost = (form.entry_price ?? 0) * (form.quantity ?? 0)
  const rr = form.stop_loss && form.profit_target && form.entry_price
    ? Math.abs((form.profit_target - form.entry_price) / (form.entry_price - form.stop_loss))
    : null

  const submit = () => {
    if (!form.asset || !form.entry_price || !form.quantity || !form.entry_date) {
      setError("请填写必填字段：资产、入场价、数量、日期")
      return
    }
    mut.mutate({
      asset: form.asset!,
      ticker: form.ticker ?? TICKER_MAP[form.asset!] ?? form.asset!,
      direction: form.direction as "long" | "short",
      entry_price: Number(form.entry_price),
      entry_date: form.entry_date!,
      quantity: Number(form.quantity),
      cost_basis_usd: cost || Number(form.entry_price) * Number(form.quantity),
      stop_loss: form.stop_loss ? Number(form.stop_loss) : null,
      profit_target: form.profit_target ? Number(form.profit_target) : null,
      trailing_stop: form.trailing_stop ?? false,
      exchange: form.exchange,
      symbol: form.symbol,
      notes: form.notes,
    })
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-lg">
        <div className="flex items-center justify-between p-5 border-b border-border">
          <h2 className="font-semibold text-slate-200">新建持仓</h2>
          <button onClick={onClose} className="text-muted hover:text-slate-200"><X size={18} /></button>
        </div>
        <div className="p-5 space-y-4">
          {error && <p className="text-red-400 text-sm bg-red-500/10 px-3 py-2 rounded-lg">{error}</p>}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">资产 *</label>
              <select value={form.asset ?? ""} onChange={(e) => handleAsset(e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
                <option value="">选择资产</option>
                {ASSETS.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">方向 *</label>
              <div className="flex gap-2">
                {(["long", "short"] as const).map((d) => (
                  <button key={d} onClick={() => set("direction", d)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors
                      ${form.direction === d
                        ? d === "long" ? "bg-green-500/20 border-green-500/50 text-green-400"
                                       : "bg-red-500/20 border-red-500/50 text-red-400"
                        : "border-border text-muted hover:text-slate-200"}`}>
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
              <label className="text-xs text-muted mb-1 block">数量 *</label>
              <input type="number" value={form.quantity ?? ""} onChange={(e) => set("quantity", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">止损价</label>
              <input type="number" value={form.stop_loss ?? ""} onChange={(e) => set("stop_loss", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">目标价</label>
              <input type="number" value={form.profit_target ?? ""} onChange={(e) => set("profit_target", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted mb-1 block">入场日期 *</label>
              <input type="date" value={form.entry_date ?? ""} onChange={(e) => set("entry_date", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">交易所</label>
              <input type="text" placeholder="NYSE / Binance" value={form.exchange ?? ""} onChange={(e) => set("exchange", e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
            </div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1 block">备注</label>
            <input type="text" value={form.notes ?? ""} onChange={(e) => set("notes", e.target.value)}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
          </div>

          {/* Summary Row */}
          <div className="bg-bg rounded-lg p-3 text-sm flex gap-6 text-muted">
            <span>成本：<span className="text-slate-200">${cost.toLocaleString("en-US", { maximumFractionDigits: 2 })}</span></span>
            {rr != null && (
              <span>R:R：<span className={rr >= 2 ? "text-green-400" : "text-red-400"}>{rr.toFixed(2)}</span>
                {rr >= 2 ? " ✓" : " ✗ (需≥2.0)"}
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-3 p-5 pt-0">
          <button onClick={onClose} className="flex-1 py-2.5 border border-border rounded-lg text-muted text-sm hover:bg-white/5">取消</button>
          <button onClick={submit} disabled={mut.isPending}
            className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-white text-sm font-medium disabled:opacity-50">
            {mut.isPending ? "创建中…" : "确认建仓"}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Close Modal ──────────────────────────────────────────────────────────
function CloseModal({ pos, onClose }: { pos: Position; onClose: () => void }) {
  const qc = useQueryClient()
  const [form, setForm] = useState<Partial<PositionClose>>({
    exit_date: new Date().toISOString().slice(0, 10),
    exit_reason: "manual",
  })

  const mut = useMutation({
    mutationFn: (data: PositionClose) => closePosition(pos.id, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["positions"] }); onClose() },
  })

  const reasons = [
    { v: "manual",          l: "手动平仓" },
    { v: "stop_loss",       l: "触发止损" },
    { v: "take_profit",     l: "目标达到" },
    { v: "signal_reversed", l: "信号反转" },
    { v: "timeout",         l: "超时平仓" },
  ]

  const pnl = form.exit_price
    ? pos.direction === "long"
      ? ((Number(form.exit_price) - pos.entry_price) / pos.entry_price * 100)
      : ((pos.entry_price - Number(form.exit_price)) / pos.entry_price * 100)
    : null

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-md">
        <div className="flex items-center justify-between p-5 border-b border-border">
          <h2 className="font-semibold text-slate-200">平仓 — {pos.asset}</h2>
          <button onClick={onClose} className="text-muted hover:text-slate-200"><X size={18} /></button>
        </div>
        <div className="p-5 space-y-4">
          <div className="bg-bg rounded-lg p-3 text-sm text-muted">
            <span>入场价：<span className="text-slate-200">{pos.entry_price}</span></span>
            <span className="mx-4">方向：<span className={pos.direction === "long" ? "text-green-400" : "text-red-400"}>{pos.direction === "long" ? "多" : "空"}</span></span>
            <span>数量：<span className="text-slate-200">{pos.quantity}</span></span>
          </div>

          <div>
            <label className="text-xs text-muted mb-1 block">出场价 *</label>
            <input type="number" value={form.exit_price ?? ""} onChange={(e) => setForm((f) => ({ ...f, exit_price: Number(e.target.value) }))}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
          </div>

          <div>
            <label className="text-xs text-muted mb-1 block">出场日期 *</label>
            <input type="date" value={form.exit_date ?? ""} onChange={(e) => setForm((f) => ({ ...f, exit_date: e.target.value }))}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500" />
          </div>

          <div>
            <label className="text-xs text-muted mb-1 block">平仓原因</label>
            <select value={form.exit_reason} onChange={(e) => setForm((f) => ({ ...f, exit_reason: e.target.value }))}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500">
              {reasons.map((r) => <option key={r.v} value={r.v}>{r.l}</option>)}
            </select>
          </div>

          {pnl != null && (
            <div className="bg-bg rounded-lg p-3 text-sm text-center">
              <span className="text-muted">预计盈亏：</span>
              <span className={`font-bold text-lg ml-2 ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}%
              </span>
            </div>
          )}
        </div>
        <div className="flex gap-3 p-5 pt-0">
          <button onClick={onClose} className="flex-1 py-2.5 border border-border rounded-lg text-muted text-sm hover:bg-white/5">取消</button>
          <button onClick={() => form.exit_price && form.exit_date && mut.mutate(form as PositionClose)}
            disabled={!form.exit_price || mut.isPending}
            className="flex-1 py-2.5 bg-red-600 hover:bg-red-700 rounded-lg text-white text-sm font-medium disabled:opacity-50">
            {mut.isPending ? "平仓中…" : "确认平仓"}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Position Row ─────────────────────────────────────────────────────────
function PositionRow({ pos }: { pos: Position }) {
  const qc = useQueryClient()
  const [expanded, setExpanded] = useState(false)
  const [closing, setClosing] = useState(false)

  const delMut = useMutation({
    mutationFn: () => deletePosition(pos.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["positions"] }),
  })

  return (
    <>
      {closing && <CloseModal pos={pos} onClose={() => setClosing(false)} />}
      <div className={`bg-surface border rounded-xl mb-3 overflow-hidden transition-colors
        ${pos.position_status === "STOP_TRIGGERED" ? "border-red-500/50" :
          pos.position_status === "TARGET_REACHED" ? "border-green-500/50" :
          pos.position_status === "SIGNAL_REVERSED" ? "border-yellow-500/50" : "border-border"}`}>

        <div className="flex items-center gap-4 p-4 cursor-pointer" onClick={() => setExpanded((e) => !e)}>
          {/* Asset + Direction */}
          <div className="w-28 shrink-0">
            <p className="font-semibold text-slate-200">{pos.asset}</p>
            <p className={`text-xs ${pos.direction === "long" ? "text-green-400" : "text-red-400"}`}>
              {pos.direction === "long" ? "▲ 多头" : "▼ 空头"}
            </p>
          </div>

          {/* Price */}
          <div className="w-24 shrink-0">
            <p className="text-xs text-muted">入场</p>
            <p className="text-sm text-slate-200">{pos.entry_price}</p>
          </div>
          <div className="w-24 shrink-0">
            <p className="text-xs text-muted">现价</p>
            <p className="text-sm text-slate-200">{pos.current_price ?? "—"}</p>
          </div>

          {/* PnL */}
          <div className="w-24 shrink-0">
            <p className="text-xs text-muted">盈亏</p>
            <PnlBadge val={pos.unrealized_pnl_pct} />
          </div>

          {/* Signal */}
          <div className="w-24 shrink-0 hidden md:block">
            <p className="text-xs text-muted">最新信号</p>
            <p className={`text-xs font-medium ${pos.latest_signal_action === "long" ? "text-green-400" : pos.latest_signal_action === "short" ? "text-red-400" : "text-muted"}`}>
              {pos.latest_signal_action ?? "—"}
              {pos.latest_signal_bias != null ? ` (${pos.latest_signal_bias.toFixed(2)})` : ""}
            </p>
          </div>

          {/* Status */}
          <div className="flex-1 hidden lg:block">
            <StatusBadge status={pos.position_status} />
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 ml-auto" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setClosing(true)}
              className="p-1.5 text-muted hover:text-orange-400 hover:bg-orange-500/10 rounded-lg transition-colors" title="平仓">
              <LogOut size={14} />
            </button>
            <button onClick={() => { if (confirm(`确定删除 ${pos.asset} 持仓？`)) delMut.mutate() }}
              className="p-1.5 text-muted hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors" title="删除">
              <Trash2 size={14} />
            </button>
            {expanded ? <ChevronUp size={14} className="text-muted" /> : <ChevronDown size={14} className="text-muted" />}
          </div>
        </div>

        {/* Expanded detail */}
        {expanded && (
          <div className="border-t border-border px-4 py-3 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-xs text-muted mb-0.5">止损</p>
              <p className="text-slate-200">{pos.stop_loss ?? "—"}</p>
              {pos.distance_to_stop_pct != null && (
                <p className="text-xs text-muted">距离 {pos.distance_to_stop_pct.toFixed(1)}%</p>
              )}
            </div>
            <div>
              <p className="text-xs text-muted mb-0.5">目标</p>
              <p className="text-slate-200">{pos.profit_target ?? "—"}</p>
              {pos.distance_to_target_pct != null && (
                <p className="text-xs text-muted">距离 {pos.distance_to_target_pct.toFixed(1)}%</p>
              )}
            </div>
            <div>
              <p className="text-xs text-muted mb-0.5">数量 / 成本</p>
              <p className="text-slate-200">{pos.quantity}</p>
              <p className="text-xs text-muted">${pos.cost_basis_usd.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs text-muted mb-0.5">入场日期</p>
              <p className="text-slate-200">{pos.entry_date.slice(0, 10)}</p>
              {pos.exchange && <p className="text-xs text-muted">{pos.exchange}</p>}
            </div>
            {pos.notes && (
              <div className="col-span-2 md:col-span-4">
                <p className="text-xs text-muted mb-0.5">备注</p>
                <p className="text-slate-300">{pos.notes}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────
export default function Portfolio() {
  const [openModal, setOpenModal] = useState(false)
  const { data: positions, isLoading } = useQuery({
    queryKey: ["positions"],
    queryFn: fetchPositions,
    refetchInterval: 60_000,
  })

  const totalPnl = positions?.reduce((s, p) => s + (p.unrealized_pnl_usd ?? 0), 0) ?? 0
  const totalCost = positions?.reduce((s, p) => s + p.cost_basis_usd, 0) ?? 0
  const totalPnlPct = totalCost > 0 ? totalPnl / totalCost * 100 : 0
  const alerts = positions?.filter((p) => p.position_status !== "HOLD") ?? []

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {openModal && <OpenModal onClose={() => setOpenModal(false)} />}
      <TopBar title="持仓管理" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Summary + Action */}
        <div className="flex items-center justify-between">
          <div className="flex gap-6 text-sm">
            <span className="text-muted">持仓：<span className="text-slate-200 font-medium">{positions?.length ?? 0}</span></span>
            <span className="text-muted">总成本：<span className="text-slate-200 font-medium">${totalCost.toLocaleString("en-US", { maximumFractionDigits: 0 })}</span></span>
            <span className="text-muted">未实现盈亏：
              <span className={`font-medium ml-1 ${totalPnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                {totalPnl >= 0 ? "+" : ""}{totalPnlPct.toFixed(2)}%
                ({totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(0)})
              </span>
            </span>
            {alerts.length > 0 && (
              <span className="text-yellow-400">⚠ {alerts.length} 项需处理</span>
            )}
          </div>
          <button onClick={() => setOpenModal(true)}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
            <Plus size={14} />
            新建持仓
          </button>
        </div>

        {/* Position list */}
        {isLoading ? (
          <div className="text-center py-16 text-muted">加载中…</div>
        ) : positions?.length === 0 ? (
          <div className="text-center py-16 text-muted">
            <p className="text-4xl mb-4">📭</p>
            <p>暂无持仓，点击「新建持仓」开始</p>
          </div>
        ) : (
          <div>
            {/* Alerts first */}
            {alerts.length > 0 && (
              <div className="mb-2">
                <p className="text-xs text-muted font-medium uppercase tracking-wider mb-2">需要处理</p>
                {alerts.map((p) => <PositionRow key={p.id} pos={p} />)}
              </div>
            )}
            {/* Normal positions */}
            {positions?.filter((p) => p.position_status === "HOLD").map((p) => (
              <PositionRow key={p.id} pos={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
