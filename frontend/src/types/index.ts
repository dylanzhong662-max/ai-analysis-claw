export type Direction = "long" | "short"
export type PositionStatus =
  | "HOLD"
  | "ADD"
  | "REDUCE"
  | "EXIT"
  | "STOP_TRIGGERED"
  | "TARGET_REACHED"
  | "SIGNAL_REVERSED"

export interface Position {
  id: number
  asset: string
  ticker: string
  direction: Direction
  entry_price: number
  entry_date: string
  quantity: number
  cost_basis_usd: number
  stop_loss?: number | null
  profit_target?: number | null
  trailing_stop: boolean
  source_signal?: string | null
  exchange?: string | null
  symbol?: string | null
  notes?: string | null
  status: "open" | "closed"
  created_at: string
  updated_at: string
  // computed
  current_price?: number | null
  unrealized_pnl_usd?: number | null
  unrealized_pnl_pct?: number | null
  distance_to_stop_pct?: number | null
  distance_to_target_pct?: number | null
  position_status?: PositionStatus | null
  latest_signal_action?: string | null
  latest_signal_bias?: number | null
}

export interface PositionCreate {
  asset: string
  ticker: string
  direction: Direction
  entry_price: number
  entry_date: string
  quantity: number
  cost_basis_usd: number
  stop_loss?: number | null
  profit_target?: number | null
  trailing_stop?: boolean
  source_signal?: string | null
  exchange?: string | null
  symbol?: string | null
  notes?: string | null
}

export interface PositionUpdate {
  stop_loss?: number | null
  profit_target?: number | null
  trailing_stop?: boolean
  notes?: string | null
  quantity?: number | null
}

export interface PositionClose {
  exit_price: number
  exit_date: string
  exit_reason?: string
  notes?: string | null
}

export interface Trade {
  id: number
  position_id?: number | null
  asset: string
  ticker: string
  direction: Direction
  entry_price: number
  entry_date: string
  exit_price: number
  exit_date: string
  quantity: number
  cost_basis_usd: number
  realized_pnl_usd?: number | null
  realized_pnl_pct?: number | null
  exit_reason?: string | null
  holding_days?: number | null
  notes?: string | null
  created_at: string
}

export interface TradeCreate {
  asset: string
  ticker: string
  direction: Direction
  entry_price: number
  entry_date: string
  exit_price: number
  exit_date: string
  quantity: number
  cost_basis_usd: number
  exit_reason?: string
  notes?: string | null
}

export interface TradeStats {
  total: number
  wins: number
  losses: number
  win_rate: number
  profit_factor: number
  avg_win_pct: number
  avg_loss_pct: number
  total_realized_pnl: number
}

export interface TradeListResponse {
  trades: Trade[]
  total: number
  page: number
  pages: number
}

export interface Signal {
  asset: string
  action?: string | null
  bias_score?: number | null
  regime?: string | null
  entry_zone?: string | null
  stop_loss?: number | null
  profit_target?: number | null
  risk_reward_ratio?: number | null
  estimated_holding_weeks?: number | null
  position_size_pct?: number | null
  invalidation_condition?: string | null
  justification?: string | null
  market_sentiment?: string | null
  analysis_date?: string | null
  raw?: Record<string, unknown>
}

export interface Alert {
  asset: string
  status: string
  message: string
  severity: "error" | "warning" | "success" | "info"
}

export interface DashboardSummary {
  portfolio_value: number
  total_unrealized_pnl_usd: number
  total_unrealized_pnl_pct: number
  active_signals: number
  total_assets: number
  needs_action: number
  open_positions: number
  alerts: Alert[]
}

export interface MacroData {
  sentiment: string
  vix?: number | null
  dxy?: number | null
  ten_year?: number | null
  btc_fear_greed?: number | null
  btc_fear_greed_label?: string | null
  last_scan_date?: string | null
  sector_ranking: SectorRank[]
  top_opportunities: TopOpportunity[]
}

export interface SectorRank {
  sector: string
  strength: "Strong" | "Neutral" | "Weak"
  assets: string[]
  rationale?: string
}

export interface TopOpportunity {
  rank: number
  asset: string
  action: string
  bias_score?: number | null
  entry_zone?: string | null
  rationale?: string
}
