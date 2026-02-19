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
  const [isRunning, setIsRunning] = useState(false);
  const [performanceData, setPerformanceData] = useState([]);
  const [metrics, setMetrics] = useState({
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    currentValue: 100000,
    winRate: 0
  });

  // ===============================================
  // YOUR ALGORITHM INTEGRATION SECTION
  // ===============================================
  // Replace this function with your actual algorithm
  const yourAlgorithm = (currentData) => {
    // Example algorithm - replace with your logic
    const randomReturn = (Math.random() - 0.48) * 2;
    const newValue = currentData.value * (1 + randomReturn / 100);

    return {
      value: newValue,
      return: randomReturn,
      signal: randomReturn > 0 ? 'BUY' : 'SELL',
      confidence: Math.abs(randomReturn) * 10
    };
  };

  // Run algorithm simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      const lastDataPoint = performanceData[performanceData.length - 1] || {
        date: 'Start',
        value: 100000,
        return: 0
      };

      const result = yourAlgorithm(lastDataPoint);

      const newDataPoint = {
        date: new Date().toLocaleTimeString(),
        value: result.value,
        return: result.return,
        signal: result.signal,
        confidence: result.confidence
      };

      setPerformanceData(prev => {
        const updated = [...prev, newDataPoint];
        return updated.slice(-50); // Keep last 50 data points
      });

      // Update metrics
      if (performanceData.length > 0) {
        const returns = performanceData.map(d => d.return);
        const totalReturn = ((result.value - 100000) / 100000) * 100;
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

        setMetrics({
          totalReturn: totalReturn.toFixed(2),
          sharpeRatio: sharpeRatio.toFixed(2),
          maxDrawdown: maxDrawdown.toFixed(2),
          currentValue: result.value.toFixed(2),
          winRate: winRate.toFixed(1)
        });
      }
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, [isRunning, performanceData]);

  const toggleAlgorithm = () => {
    setIsRunning(!isRunning);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setPerformanceData([]);
    setMetrics({
      totalReturn: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      currentValue: 100000,
      winRate: 0
    });
  };

  const latestSignal = performanceData[performanceData.length - 1];

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="max-w-[1400px] mx-auto px-8 py-12">
        {/* Header */}
        <div className="mb-16">
          <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-3 mono">Asset Management</p>
          <h1 className="text-5xl font-light mb-4 tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Performance Dashboard
          </h1>
          <p className="text-gray-400 text-lg font-light">Real-time algorithm monitoring and metrics</p>
        </div>

        {/* Control Panel */}
        <div className="mb-16">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <button
                onClick={toggleAlgorithm}
                className="px-8 py-3 bg-cyan-400/60 text-black rounded-none font-medium tracking-wide text-sm hover:bg-cyan-400/80 transition-all uppercase mono"
              >
                {isRunning ? 'Stop' : 'Start'}
              </button>

              <button
                onClick={resetSimulation}
                className="px-8 py-3 border border-white/20 text-white rounded-none font-medium tracking-wide text-sm hover:border-white/40 transition-all uppercase mono"
              >
                Reset
              </button>
            </div>

            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-cyan-400/70' : 'bg-gray-600'}`} />
              <span className="text-gray-400 text-sm uppercase tracking-wider mono">{isRunning ? 'Running' : 'Stopped'}</span>
            </div>
          </div>

          {latestSignal && (
            <div className="border-t border-white/10 pt-8">
              <div className="grid grid-cols-2 gap-12">
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
              </div>
            </div>
          )}
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-5 gap-px bg-white/10 mb-16">
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
            title="Max Drawdown"
            value={`${metrics.maxDrawdown}%`}
            gradient="from-pink-400/70 to-pink-300/70"
          />
          <MetricCard
            title="Win Rate"
            value={`${metrics.winRate}%`}
            gradient="from-yellow-400/70 to-yellow-300/70"
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

        {/* Integration Instructions */}
        <div className="border-t border-white/10 pt-16">
          <p className="text-xs tracking-[0.2em] uppercase text-gray-500 mb-6 mono">Integration Guide</p>
          <div className="max-w-3xl">
            <p className="text-gray-400 mb-6 text-lg font-light leading-relaxed">
              To integrate your algorithm, locate the <code className="mono bg-white/10 text-gray-300 px-2 py-1 border border-white/15">yourAlgorithm</code> function and replace the example logic with your implementation.
            </p>
            <div className="bg-white/5 p-8 border border-white/10">
              <pre className="text-gray-300 text-sm mono leading-relaxed">{`{
  value: number,      // New portfolio value
  return: number,     // Return percentage
  signal: string,     // 'BUY' or 'SELL'
  confidence: number  // Confidence level (0-100)
}`}</pre>
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