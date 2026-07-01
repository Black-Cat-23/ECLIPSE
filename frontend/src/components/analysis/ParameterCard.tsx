import { motion } from 'framer-motion'

interface Habitability {
  esi_score: number
  hz_class: string
  tier: number
  rv_amplitude_ms: number | null
}

interface Props {
  result: {
    period: number | null
    period_err: number | null
    duration_days: number | null
    duration_hrs: number | null
    duration_err: number | null
    depth: number | null
    depth_err: number | null
    depth_ppm: number | null
    snr_photometric: number | null
    rp_rearth: number | null
    t_eq_kelvin: number | null
    n_transits: number | null
    snr_tls: number | null
    processing_time_s: number
    habitability?: Habitability | null
  }
}

const HZ_COLORS: Record<string, string> = {
  CONSERVATIVE: '#4ade80', INNER: '#fbbf24', OUTER: '#60a5fa', NONE: '#6b7280'
}


export default function ParameterCard({ result }: Props) {
  const hab = result.habitability
  const hzColor = hab ? (HZ_COLORS[hab.hz_class] || '#6b7280') : '#6b7280'

  const rows = [
    { label: 'Period',       value: result.period,         err: result.period_err,   unit: 'd',    d: 5 },
    { label: 'Duration',     value: result.duration_hrs,   err: result.duration_err, unit: 'hrs',  d: 3 },
    { label: 'Depth',        value: result.depth_ppm,      err: null,                unit: 'ppm',  d: 1 },
    { label: 'Photo SNR',    value: result.snr_photometric, err: null,               unit: 'σ',    d: 2 },
    { label: 'TLS SDE',      value: result.snr_tls,        err: null,                unit: 'σ',    d: 2 },
    { label: 'N transits',   value: result.n_transits,     err: null,                unit: '',     d: 0 },
    { label: 'Rp',           value: result.rp_rearth,      err: null,                unit: 'R⊕',   d: 2 },
    { label: 'Teq',          value: result.t_eq_kelvin,    err: null,                unit: 'K',    d: 0 },
  ]

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Transit Parameters</h2>
        <span className="text-xs text-gray-600 font-mono">{result.processing_time_s.toFixed(1)}s</span>
      </div>

      <div className="space-y-3">
        {rows.map(({ label, value, err, unit, d }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.06 }}
            className="flex justify-between items-baseline"
          >
            <span className="text-gray-500 text-xs">{label}</span>
            <span className="font-mono text-white text-sm">
              {value != null ? (d === 0 ? String(value) : Number(value).toFixed(d)) : '—'}
              {value != null && unit && <span className="text-gray-500 text-xs ml-1">{unit}</span>}
              {err != null && <span className="text-nebula-blue/60 text-xs ml-1">± {Number(err).toFixed(d)}</span>}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Habitability block */}
      {hab && (
        <div className="mt-5 pt-4 border-t border-white/5">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Habitability</div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-xs">ESI</span>
              <span className="font-mono text-sm" style={{ color: hab.esi_score > 0.6 ? '#4ade80' : '#fb923c' }}>
                {(hab.esi_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-xs">HZ Class</span>
              <span className="text-xs px-2 py-0.5 rounded font-mono"
                style={{ background: `${hzColor}20`, color: hzColor, border: `1px solid ${hzColor}40` }}>
                {hab.hz_class}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500 text-xs">Tier</span>
              <span className="font-mono text-sm text-yellow-400">Tier {hab.tier}</span>
            </div>
            {hab.rv_amplitude_ms != null && (
              <div className="flex justify-between items-center">
                <span className="text-gray-500 text-xs">RV K</span>
                <span className="font-mono text-sm text-gray-300">{hab.rv_amplitude_ms.toFixed(3)} m/s</span>
              </div>
            )}
          </div>
        </div>
      )}

      {result.depth != null && (
        <div className="mt-4 pt-4 border-t border-white/5">
          <div className="text-gray-500 text-xs mb-1">Rp / Rs = √depth</div>
          <div className="font-mono text-nebula-cyan text-lg font-semibold">
            {Math.sqrt(result.depth).toFixed(4)}
          </div>
        </div>
      )}
    </div>
  )
}

