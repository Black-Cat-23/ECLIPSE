// LightCurveChart.tsx — placeholder component
// Full D3 light curve visualization would render detrended flux from API
import { motion } from 'framer-motion'

interface Props {
  ticId: number
  sector: number
}

export default function LightCurveChart({ ticId, sector }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
      className="glass-card p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Light Curve — TIC {ticId} · Sector {sector}
        </h2>
        <span className="text-xs text-gray-600 font-mono">PDCSAP Flux</span>
      </div>
      <div className="h-40 flex items-center justify-center text-gray-600 text-sm border border-dashed border-space-700 rounded-lg">
        <div className="text-center">
          <div className="text-2xl mb-2">📊</div>
          <p>Light curve visualization</p>
          <p className="text-xs text-gray-700 mt-1">Connect to backend with fetched light curve data</p>
        </div>
      </div>
    </motion.div>
  )
}
