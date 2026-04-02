import { api } from "./client"
import type { Signal } from "../types"

export const fetchAllSignals = (): Promise<Signal[]> =>
  api.get("/api/signals").then((r) => r.data)

export const fetchSignal = (asset: string): Promise<Signal> =>
  api.get(`/api/signals/${asset}`).then((r) => r.data)

export const refreshSignal = (asset: string): Promise<{ status: string; message: string }> =>
  api.post(`/api/signals/refresh/${asset}`).then((r) => r.data)

export const fetchLatestScan = () =>
  api.get("/api/scan/latest").then((r) => r.data)

export const runScan = (group = "quick") =>
  api.post("/api/scan/run", null, { params: { group } }).then((r) => r.data)
