import { BrowserRouter, Routes, Route } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import Sidebar from "./components/layout/Sidebar"
import Dashboard from "./pages/Dashboard"
import MarketScan from "./pages/MarketScan"
import Portfolio from "./pages/Portfolio"
import TradeHistory from "./pages/TradeHistory"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30_000 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex h-screen overflow-hidden bg-bg text-slate-200">
          <Sidebar />
          <main className="flex-1 overflow-hidden flex flex-col">
            <Routes>
              <Route path="/"         element={<Dashboard />} />
              <Route path="/scan"     element={<MarketScan />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/trades"   element={<TradeHistory />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
