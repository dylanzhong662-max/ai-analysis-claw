import { NavLink } from "react-router-dom"
import { LayoutDashboard, BriefcaseBusiness, History, ScanSearch } from "lucide-react"

const links = [
  { to: "/",         icon: LayoutDashboard,  label: "仪表盘" },
  { to: "/scan",     icon: ScanSearch,        label: "市场扫描" },
  { to: "/portfolio",icon: BriefcaseBusiness, label: "持仓管理" },
  { to: "/trades",   icon: History,           label: "交易记录" },
]

export default function Sidebar() {
  return (
    <aside className="w-16 hover:w-48 transition-all duration-200 overflow-hidden
                      bg-surface border-r border-border flex flex-col shrink-0 group">
      <div className="h-14 flex items-center justify-center border-b border-border px-2">
        <span className="text-blue-400 font-bold text-lg whitespace-nowrap">
          <span className="group-hover:hidden">AI</span>
          <span className="hidden group-hover:inline">AI 金融分析</span>
        </span>
      </div>
      <nav className="flex-1 py-4">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 mx-2 rounded-lg mb-1 transition-colors text-sm
               ${isActive
                 ? "bg-blue-500/20 text-blue-400"
                 : "text-muted hover:bg-white/5 hover:text-slate-200"}`
            }
          >
            <Icon size={18} className="shrink-0" />
            <span className="hidden group-hover:block whitespace-nowrap">{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
