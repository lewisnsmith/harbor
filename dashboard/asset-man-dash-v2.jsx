import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Play, Pause, RefreshCw } from 'lucide-react';

export default function AssetManagementDashboard() {
    // Add custom fonts
    useEffect(() => {
        const style = document.createElement('style');
        style.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
      
      body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      
      .mono {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
      }
    `;
        document.head.appendChild(style);
        return () => document.head.removeChild(style);
    }, []);
    const [connectionStatus, setConnectionStatus] = useState('disconnected'); // 'disconnected', 'connecting', 'connected', 'error'
    const [performanceData, setPerformanceData] = useState([]);
    const [tradeLog, setTradeLog] = useState([]);
    const [currentPosition, setCurrentPosition] = useState(null);
    const [metrics, setMetrics] = useState({
        totalReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        currentValue: 100000,
        winRate: 0,
        totalTrades: 0,
        profitFactor: 0,
        avgWin: 0,
        avgLoss: 0
    });
    const [wsUrl, setWsUrl] = useState('');
    const wsRef = React.useRef(null);

    // ===============================================
    // CONNECTION SECTION - REPLACE WITH YOUR ENDPOINT
    // ===============================================
    // Configure your WebSocket URL or API endpoint here
    const DEFAULT_WS_URL = 'ws://localhost:8080'; // Replace with your server
    const DEFAULT_API_URL = 'http://localhost:8080/stream'; // Or REST API endpoint

    // WebSocket connection handler
    const connectToAlgorithm = (url) => {
        setConnectionStatus('connecting');

        try {
            const ws = new WebSocket(url || DEFAULT_WS_URL);

            ws.onopen = () => {
                console.log('Connected to algorithm');
                setConnectionStatus('connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleAlgorithmUpdate(data);
                } catch (err) {
                    console.error('Failed to parse message:', err);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setConnectionStatus('error');
            };

            ws.onclose = () => {
                console.log('Disconnected from algorithm');
                setConnectionStatus('disconnected');
            };

            wsRef.current = ws;
        } catch (err) {
            console.error('Connection failed:', err);
            setConnectionStatus('error');
        }
    };

    // Handle incoming data from algorithm
    const handleAlgorithmUpdate = (data) => {
        // Expected data format:
        // {
        //   timestamp: "2024-01-01T12:00:00Z",
        //   value: 100500,
        //   return: 0.5,
        //   signal: "BUY",
        //   confidence: 75,
        //   trade: { type: "ENTRY", signal: "BUY", price: 100500, size: 1, pnl: 0 }
        // }

        // Handle trade execution
        if (data.trade) {
            const newTrade = {
                ...data.trade,
                id: Date.now(),
                timestamp: data.timestamp || new Date().toLocaleTimeString()
            };
            setTradeLog(prev => [newTrade, ...prev].slice(0, 20));

            if (data.trade.type === 'ENTRY') {
                setCurrentPosition({
                    signal: data.trade.signal,
                    price: data.trade.price,
                    size: data.trade.size,
                    timestamp: data.timestamp
                });
            } else if (data.trade.type === 'EXIT') {
                setCurrentPosition(null);
            }
        }

        const newDataPoint = {
            date: data.timestamp || new Date().toLocaleTimeString(),
            value: data.value,
            return: data.return,
            signal: data.signal,
            confidence: data.confidence
        };

        setPerformanceData(prev => {
            const updated = [...prev, newDataPoint];
            return updated.slice(-50);
        });

        // Update metrics if provided, otherwise calculate
        if (data.metrics) {
            setMetrics(data.metrics);
        } else {
            calculateMetrics(newDataPoint);
        }
    };

    // Calculate metrics from performance data
    const calculateMetrics = (latestPoint) => {
        if (performanceData.length === 0) return;

        const returns = performanceData.map(d => d.return);
        const totalReturn = ((latestPoint.value - 100000) / 100000) * 100;
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const stdDev = Math.sqrt(
            returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
        );
        const sharpeRatio = stdDev !== 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;

        let maxDrawdown = 0;
        let peak = performanceData[0].value;
        performanceData.forEach(point => {
            if (point.value > peak) peak = point.value;
            const drawdown = ((peak - point.value) / peak) * 100;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        });

        const winRate = (returns.filter(r => r > 0).length / returns.length) * 100;

        const completedTrades = tradeLog.filter(t => t.type === 'EXIT');
        const wins = completedTrades.filter(t => t.pnl > 0);
        const losses = completedTrades.filter(t => t.pnl <= 0);
        const totalWins = wins.reduce((sum, t) => sum + t.pnl, 0);
        const totalLosses = Math.abs(losses.reduce((sum, t) => sum + t.pnl, 0));
        const profitFactor = totalLosses !== 0 ? totalWins / totalLosses : 0;
        const avgWin = wins.length > 0 ? totalWins / wins.length : 0;
        const avgLoss = losses.length > 0 ? totalLosses / losses.length : 0;

        setMetrics({
            totalReturn: totalReturn.toFixed(2),
            sharpeRatio: sharpeRatio.toFixed(2),
            maxDrawdown: maxDrawdown.toFixed(2),
            currentValue: latestPoint.value.toFixed(2),
            winRate: winRate.toFixed(1),
            totalTrades: completedTrades.length,
            profitFactor: profitFactor.toFixed(2),
            avgWin: avgWin.toFixed(2),
            avgLoss: avgLoss.toFixed(2)
        });
    };

    // Auto-connect on mount (optional)
    useEffect(() => {
        // Uncomment to auto-connect on page load
        // connectToAlgorithm(DEFAULT_WS_URL);

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    const disconnect = () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setConnectionStatus('disconnected');
    };

    const clearData = () => {
        setPerformanceData([]);
        setTradeLog([]);
        setCurrentPosition(null);
        setMetrics({
            totalReturn: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            currentValue: 100000,
            winRate: 0,
            totalTrades: 0,
            profitFactor: 0,
            avgWin: 0,
            avgLoss: 0
        });
    };

    const latestSignal = performanceData[performanceData.length - 1];

    const getStatusColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'bg-emerald-400/70';
            case 'connecting': return 'bg-yellow-400/70 animate-pulse';
            case 'error': return 'bg-red-400/70';
            default: return 'bg-gray-600';
        }
    };

    const getStatusText = () => {
        switch (connectionStatus) {
            case 'connected': return 'Connected';
            case 'connecting': return 'Connecting';
            case 'error': return 'Error';
            default: return 'Disconnected';
        }
    };

    return (
        <div className="min-h-screen bg-black text-white">
            <div className="max-w-[1400px] mx-auto px-8 py-12">
                {/* Header */}
                <div className="mb-16">
                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-3 mono">Asset Management</p>
                    <h1 className="text-5xl font-light mb-4 tracking-tight bg-gradient-to-r from-cyan-300 via-emerald-300 to-yellow-300 bg-clip-text text-transparent">
                        Performance Dashboard
                    </h1>
                    <p className="text-gray-400 text-lg font-light">Live algorithm monitoring and analytics</p>
                </div>

                {/* Connection Panel */}
                <div className="mb-16">
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-4">
                            <input
                                type="text"
                                value={wsUrl}
                                onChange={(e) => setWsUrl(e.target.value)}
                                placeholder={DEFAULT_WS_URL}
                                className="px-4 py-3 bg-white/5 border border-white/20 rounded-none text-white mono text-sm w-80 focus:outline-none focus:border-cyan-400/50 transition-colors"
                                disabled={connectionStatus === 'connected' || connectionStatus === 'connecting'}
                            />

                            {connectionStatus === 'connected' ? (
                                <button
                                    onClick={disconnect}
                                    className="px-8 py-3 border border-white/20 text-white rounded-none font-medium tracking-wide text-sm hover:border-orange-400/50 hover:text-orange-400/80 transition-all uppercase mono"
                                >
                                    Disconnect
                                </button>
                            ) : (
                                <button
                                    onClick={() => connectToAlgorithm(wsUrl)}
                                    className="px-8 py-3 bg-gradient-to-r from-cyan-400/80 to-emerald-400/80 text-black rounded-none font-medium tracking-wide text-sm hover:from-cyan-400 hover:to-emerald-400 transition-all uppercase mono"
                                    disabled={connectionStatus === 'connecting'}
                                >
                                    {connectionStatus === 'connecting' ? 'Connecting...' : 'Connect'}
                                </button>
                            )}

                            <button
                                onClick={clearData}
                                className="px-8 py-3 border border-white/20 text-white rounded-none font-medium tracking-wide text-sm hover:border-white/40 transition-all uppercase mono"
                            >
                                Clear Data
                            </button>
                        </div>

                        <div className="flex items-center gap-3">
                            <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
                            <span className="text-gray-400 text-sm uppercase tracking-wider mono">{getStatusText()}</span>
                        </div>
                    </div>

                    {latestSignal && (
                        <div className="border-t border-white/10 pt-8">
                            <div className="grid grid-cols-3 gap-12">
                                <div>
                                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-2 mono">Latest Signal</p>
                                    <span className={`text-2xl font-light mono ${latestSignal.signal === 'BUY' ? 'text-emerald-400/80' : 'text-orange-400/80'
                                        }`}>
                                        {latestSignal.signal}
                                    </span>
                                </div>
                                <div>
                                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-2 mono">Confidence</p>
                                    <span className="text-2xl font-light mono text-cyan-400/70">{latestSignal.confidence?.toFixed(1)}%</span>
                                </div>
                                <div>
                                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-2 mono">Position</p>
                                    <span className={`text-2xl font-light mono ${currentPosition ? 'text-yellow-400/80' : 'text-gray-500'}`}>
                                        {currentPosition ? currentPosition.signal : 'NONE'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-4 gap-px bg-white/10 mb-8">
                    <MetricCard
                        title="Current Value"
                        value={`$${Number(metrics.currentValue).toLocaleString()}`}
                        gradient="from-cyan-400/70 to-cyan-300/70"
                    />
                    <MetricCard
                        title="Total Return"
                        value={`${metrics.totalReturn}%`}
                        gradient={metrics.totalReturn >= 0 ? "from-emerald-400/70 to-emerald-300/70" : "from-orange-400/70 to-orange-300/70"}
                    />
                    <MetricCard
                        title="Sharpe Ratio"
                        value={metrics.sharpeRatio}
                        gradient="from-blue-400/70 to-blue-300/70"
                    />
                    <MetricCard
                        title="Win Rate"
                        value={`${metrics.winRate}%`}
                        gradient="from-yellow-400/70 to-yellow-300/70"
                    />
                </div>

                <div className="grid grid-cols-4 gap-px bg-white/10 mb-16">
                    <MetricCard
                        title="Max Drawdown"
                        value={`${metrics.maxDrawdown}%`}
                        gradient="from-pink-400/70 to-pink-300/70"
                    />
                    <MetricCard
                        title="Total Trades"
                        value={metrics.totalTrades}
                        gradient="from-purple-400/70 to-purple-300/70"
                    />
                    <MetricCard
                        title="Profit Factor"
                        value={metrics.profitFactor}
                        gradient="from-green-400/70 to-green-300/70"
                    />
                    <MetricCard
                        title="Avg Win/Loss"
                        value={`${metrics.avgWin}/${metrics.avgLoss}%`}
                        gradient="from-indigo-400/70 to-indigo-300/70"
                    />
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-px bg-white/10 mb-16">
                    {/* Portfolio Value Chart */}
                    <div className="bg-black p-12">
                        <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-6 mono">Portfolio Value</p>
                        <ResponsiveContainer width="100%" height={320}>
                            <AreaChart data={performanceData}>
                                <defs>
                                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#67e8f9" stopOpacity={0.15} />
                                        <stop offset="95%" stopColor="#67e8f9" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                                        <stop offset="0%" stopColor="#67e8f9" stopOpacity={0.8} />
                                        <stop offset="100%" stopColor="#a5b4fc" stopOpacity={0.8} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="0" stroke="#1a1a1a" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#666666"
                                    tick={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                                    tickLine={false}
                                    axisLine={{ stroke: '#1a1a1a' }}
                                />
                                <YAxis
                                    stroke="#666666"
                                    tick={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                                    tickLine={false}
                                    axisLine={{ stroke: '#1a1a1a' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#000000',
                                        border: '1px solid #333333',
                                        borderRadius: '0',
                                        fontSize: '12px',
                                        color: '#ffffff',
                                        fontFamily: 'JetBrains Mono, monospace'
                                    }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="url(#lineGradient)"
                                    strokeWidth={2}
                                    fillOpacity={1}
                                    fill="url(#colorValue)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Returns Chart */}
                    <div className="bg-black p-12">
                        <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-6 mono">Returns Distribution</p>
                        <ResponsiveContainer width="100%" height={320}>
                            <LineChart data={performanceData}>
                                <defs>
                                    <linearGradient id="returnGradient" x1="0" y1="0" x2="1" y2="0">
                                        <stop offset="0%" stopColor="#93c5fd" stopOpacity={0.8} />
                                        <stop offset="100%" stopColor="#c4b5fd" stopOpacity={0.8} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="0" stroke="#1a1a1a" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#666666"
                                    tick={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                                    tickLine={false}
                                    axisLine={{ stroke: '#1a1a1a' }}
                                />
                                <YAxis
                                    stroke="#666666"
                                    tick={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                                    tickLine={false}
                                    axisLine={{ stroke: '#1a1a1a' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#000000',
                                        border: '1px solid #333333',
                                        borderRadius: '0',
                                        fontSize: '12px',
                                        color: '#ffffff',
                                        fontFamily: 'JetBrains Mono, monospace'
                                    }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="return"
                                    stroke="url(#returnGradient)"
                                    strokeWidth={2}
                                    name="Return (%)"
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Trade Log */}
                <div className="mb-16">
                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-6 mono">Recent Trades</p>
                    <div className="border border-white/10">
                        {/* Table Header */}
                        <div className="grid grid-cols-6 gap-4 p-4 border-b border-white/10 bg-white/5">
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">Time</span>
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">Type</span>
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">Signal</span>
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">Price</span>
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">Size</span>
                            <span className="text-xs tracking-wider uppercase text-gray-500 mono">P&L</span>
                        </div>

                        {/* Table Rows */}
                        <div className="max-h-[400px] overflow-y-auto">
                            {tradeLog.length === 0 ? (
                                <div className="p-8 text-center text-gray-500 text-sm mono">
                                    No trades executed yet. Start the algorithm to see trades.
                                </div>
                            ) : (
                                tradeLog.map((trade) => (
                                    <div key={trade.id} className="grid grid-cols-6 gap-4 p-4 border-b border-white/5 hover:bg-white/5 transition-colors">
                                        <span className="text-sm text-gray-400 mono">{trade.timestamp}</span>
                                        <span className="text-sm text-gray-300 mono">{trade.type}</span>
                                        <span className={`text-sm mono ${trade.signal === 'BUY' ? 'text-emerald-400/80' : 'text-orange-400/80'}`}>
                                            {trade.signal}
                                        </span>
                                        <span className="text-sm text-gray-300 mono">${trade.price.toFixed(2)}</span>
                                        <span className="text-sm text-gray-300 mono">{trade.size}</span>
                                        <span className={`text-sm mono ${trade.pnl > 0 ? 'text-emerald-400/80' : trade.pnl < 0 ? 'text-orange-400/80' : 'text-gray-400'}`}>
                                            {trade.pnl ? `${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}%` : '-'}
                                        </span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Integration Instructions */}
                <div className="border-t border-white/10 pt-16">
                    <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-6 mono">Connection Guide</p>
                    <div className="max-w-3xl space-y-8">
                        <div>
                            <h3 className="text-lg font-light text-white mb-4">WebSocket Connection</h3>
                            <p className="text-gray-400 mb-4 font-light leading-relaxed">
                                Connect your algorithm running on cloud servers or terminal to this dashboard via WebSocket. Configure the WebSocket URL at the top of the file or use the input field above.
                            </p>
                            <div className="bg-white/5 p-6 border border-white/10">
                                <pre className="text-gray-300 text-sm mono leading-relaxed">{`// Your server should send messages in this format:
{
  timestamp: "2024-01-01T12:00:00Z",
  value: 100500,           // Portfolio value
  return: 0.5,             // Return %
  signal: "BUY",           // Current signal
  confidence: 75,          // Confidence level 0-100
  
  // Optional: Send trade execution
  trade: {
    type: "ENTRY" | "EXIT",
    signal: "BUY" | "SELL",
    price: 100500,
    size: 1,
    pnl: 0.5              // For EXIT trades
  },
  
  // Optional: Send pre-calculated metrics
  metrics: {
    totalReturn: 5.2,
    sharpeRatio: 1.8,
    // ... other metrics
  }
}`}</pre>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-light text-white mb-4">Example Server Implementation</h3>
                            <div className="bg-white/5 p-6 border border-white/10">
                                <pre className="text-gray-300 text-sm mono leading-relaxed">{`// Python WebSocket Server Example
import asyncio
import websockets
import json

async def algorithm_stream(websocket):
    while True:
        # Your algorithm logic here
        data = {
            "timestamp": datetime.now().isoformat(),
            "value": get_portfolio_value(),
            "return": calculate_return(),
            "signal": get_signal(),
            "confidence": get_confidence()
        }
        
        await websocket.send(json.dumps(data))
        await asyncio.sleep(1)  # Update frequency

async def main():
    async with websockets.serve(algorithm_stream, "localhost", 8080):
        await asyncio.Future()

asyncio.run(main())`}</pre>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-light text-white mb-4">Configuration</h3>
                            <p className="text-gray-400 font-light leading-relaxed">
                                Update <code className="mono bg-white/10 text-gray-300 px-2 py-1 border border-white/15">DEFAULT_WS_URL</code> in the code to your WebSocket endpoint, or use the connection input field to connect dynamically.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function MetricCard({ title, value, gradient = "from-white to-gray-300" }) {
    return (
        <div className="bg-black p-8">
            <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-4 mono">{title}</p>
            <p className={`text-3xl font-light mono bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>{value}</p>
        </div>
    );
}
