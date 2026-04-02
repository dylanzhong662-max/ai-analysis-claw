import { RefreshCw } from "lucide-react"
import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"

export default function TopBar({ title }: { title: string }) {
  const qc = useQueryClient()
  const [spinning, setSpinning] = useState(false)

  const handleRefresh = () => {
    setSpinning(true)
    qc.invalidateQueries()
    setTimeout(() => setSpinning(false), 1000)
  }

  return (
    <header className="h-14 border-b border-border bg-surface flex items-center justify-between px-6 shrink-0">
      <h1 className="text-base font-semibold text-slate-200">{title}</h1>
      <button
        onClick={handleRefresh}
        className="flex items-center gap-2 text-muted hover:text-slate-200 text-sm px-3 py-1.5
                   border border-border rounded-lg hover:bg-white/5 transition-colors"
      >
        <RefreshCw size={14} className={spinning ? "animate-spin" : ""} />
        刷新
      </button>
    </header>
  )
}
