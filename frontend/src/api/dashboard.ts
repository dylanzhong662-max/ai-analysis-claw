import { api } from "./client"
import type { DashboardSummary, MacroData } from "../types"

export const fetchSummary = (): Promise<DashboardSummary> =>
  api.get("/api/dashboard/summary").then((r) => r.data)

export const fetchMacro = (): Promise<MacroData> =>
  api.get("/api/dashboard/macro").then((r) => r.data)
