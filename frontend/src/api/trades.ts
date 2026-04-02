import { api } from "./client"
import type { Trade, TradeCreate, TradeStats, TradeListResponse } from "../types"

export const fetchTrades = (params?: {
  asset?: string
  direction?: string
  exit_reason?: string
  start_date?: string
  end_date?: string
  page?: number
  limit?: number
}): Promise<TradeListResponse> =>
  api.get("/api/trades", { params }).then((r) => r.data)

export const createTrade = (data: TradeCreate): Promise<Trade> =>
  api.post("/api/trades", data).then((r) => r.data)

export const deleteTrade = (id: number): Promise<void> =>
  api.delete(`/api/trades/${id}`).then((r) => r.data)

export const fetchTradeStats = (): Promise<TradeStats> =>
  api.get("/api/trades/stats").then((r) => r.data)
