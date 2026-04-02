import { api } from "./client"
import type { Position, PositionCreate, PositionUpdate, PositionClose, Trade } from "../types"

export const fetchPositions = (): Promise<Position[]> =>
  api.get("/api/portfolio/positions").then((r) => r.data)

export const createPosition = (data: PositionCreate): Promise<Position> =>
  api.post("/api/portfolio/positions", data).then((r) => r.data)

export const updatePosition = (id: number, data: PositionUpdate): Promise<Position> =>
  api.put(`/api/portfolio/positions/${id}`, data).then((r) => r.data)

export const deletePosition = (id: number): Promise<void> =>
  api.delete(`/api/portfolio/positions/${id}`).then((r) => r.data)

export const closePosition = (id: number, data: PositionClose): Promise<Trade> =>
  api.post(`/api/portfolio/positions/${id}/close`, data).then((r) => r.data)
