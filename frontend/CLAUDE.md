# Frontend — React 仪表盘说明

## 技术栈

| 库 | 版本 | 用途 |
|----|------|------|
| React | 18 | UI 框架 |
| TypeScript | 5 | 类型安全 |
| Vite | 5 | 构建工具 |
| Tailwind CSS | 3 | 样式（深色主题） |
| React Router | 6 | 客户端路由 |
| TanStack Query | 5 | 数据请求 + 缓存 |
| Axios | 1.x | HTTP 客户端 |
| Lucide React | latest | 图标 |

## 目录结构

```
frontend/src/
├── main.tsx                  # 入口
├── App.tsx                   # 根组件：路由 + QueryClientProvider
├── index.css                 # 全局样式（Tailwind 指令 + 滚动条）
├── types/
│   └── index.ts              # 所有 TypeScript 类型定义（与后端 Schema 对齐）
├── api/
│   ├── client.ts             # axios 实例，baseURL 从 VITE_API_URL 读取
│   ├── dashboard.ts          # fetchSummary / fetchMacro
│   ├── portfolio.ts          # fetchPositions / createPosition / updatePosition / closePosition / deletePosition
│   ├── trades.ts             # fetchTrades / createTrade / deleteTrade / fetchTradeStats
│   └── signals.ts            # fetchAllSignals / fetchSignal / refreshSignal / fetchLatestScan / runScan
├── components/
│   └── layout/
│       ├── Sidebar.tsx       # 折叠侧边栏（hover 展开），NavLink 高亮当前页
│       └── TopBar.tsx        # 顶部标题栏 + 全局刷新按钮
└── pages/
    ├── Dashboard.tsx         # 仪表盘（告警横幅/KPI/宏观/Top机会/信号矩阵）
    ├── MarketScan.tsx        # 市场扫描（信号卡片/扫描触发/汇总结果）
    ├── Portfolio.tsx         # 持仓管理（新建/平仓/删除/展开详情）
    └── TradeHistory.tsx      # 交易记录（统计卡片/列表/手动录入/分页）
```

## 路由

| 路径 | 页面 | 说明 |
|------|------|------|
| `/` | Dashboard | 仪表盘，默认页 |
| `/scan` | MarketScan | 市场扫描 |
| `/portfolio` | Portfolio | 持仓管理 |
| `/trades` | TradeHistory | 交易记录 |

## 数据请求模式

所有接口调用通过 TanStack Query，缓存 30 秒（`staleTime: 30_000`），持仓/仪表盘每 60 秒自动轮询刷新（`refetchInterval: 60_000`）。

写操作（新建/更新/删除）用 `useMutation`，成功后调用 `queryClient.invalidateQueries` 使相关缓存失效。

```ts
// 典型模式
const mut = useMutation({
  mutationFn: createPosition,
  onSuccess: () => qc.invalidateQueries({ queryKey: ["positions"] }),
})
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VITE_API_URL` | `http://localhost:8000` | 后端 API 地址 |

部署到服务器前，在 `frontend/` 根目录创建 `.env`：
```
VITE_API_URL=http://你的服务器IP:8000
```
修改后需重新 `npm run build`。

## 主题色板

深色金融终端风格，定义在 `tailwind.config.js`：

| Token | 值 | 用途 |
|-------|-----|------|
| `bg` | `#0f0f12` | 页面背景 |
| `surface` | `#1a1a24` | 卡片/面板背景 |
| `border` | `#2a2a38` | 边框 |
| `muted` | `#94a3b8` | 次级文字 |
| `text-slate-200` | Tailwind 内置 | 主文字 |
| `text-green-400` | Tailwind 内置 | 盈利/做多 |
| `text-red-400` | Tailwind 内置 | 亏损/做空/止损 |
| `text-blue-400/600` | Tailwind 内置 | 主要交互色 |
| `text-yellow-400` | Tailwind 内置 | 警告 |

## 关键组件行为

### Sidebar
- 默认宽度 `w-16`（只显示图标），hover 时过渡到 `w-48`（显示文字）
- 使用 `group` + `group-hover:` 控制文字显隐

### Dashboard 信号矩阵
- 按资产组（贵金属/加密/科技股）排列
- `BiasBar` 组件：bias ≥ 0.65 绿色，≥ 0.5 黄色，< 0.5 灰色

### Portfolio 持仓行
- 颜色边框编码：`STOP_TRIGGERED` 红色，`TARGET_REACHED` 绿色，`SIGNAL_REVERSED` 黄色
- 点击行展开详情（止损距离/目标距离/备注）
- 告警状态的持仓排在列表最前面

### 平仓 Modal
- 实时计算并显示预计盈亏%
- 平仓成功后自动生成 trade 记录（由后端处理）

## 运行与构建

```bash
# 开发模式
npm run dev

# 生产构建
npm run build           # 输出到 dist/
npm run preview         # 本地预览 dist/

# 类型检查
npx tsc --noEmit
```

## 注意事项

- **VITE_API_URL 必须在构建前设置**：Vite 在构建时静态替换环境变量，运行时修改无效
- **跨域**：后端已配置 `allow_origins=["*"]`，本地开发和同服务器部署均无跨域问题
- **App.css**：Vite 默认生成，目前内容无影响，可保留或清空
