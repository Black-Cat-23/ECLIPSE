import { motion } from 'framer-motion'

type ClassKey = 'TRANSIT' | 'EB' | 'BLEND' | 'OTHER'

const CLASS_CONFIG: Record<ClassKey, { label: string; color: string; badgeClass: string; desc: string }> = {
  TRANSIT: { label: 'Planet Transit',         color: '#4CAF50', badgeClass: 'badge-transit', desc: 'Periodic transit by orbiting exoplanet'  },
  EB:      { label: 'Eclipsing Binary',        color: '#FF9800', badgeClass: 'badge-eb',      desc: 'Eclipsing binary star system'           },
  BLEND:   { label: 'Background Blend',        color: '#F44336', badgeClass: 'badge-blend',   desc: 'Transit from background eclipsing source'},
  OTHER:   { label: 'Other / Instrumental',   color: '#9E9E9E', badgeClass: 'badge-other',   desc: 'Stellar variability or noise artifact'  },
}

interface Props {
  result: {
    predicted_class: string
    class_probs: { TRANSIT: number; EB: number; BLEND: number; OTHER: number }
    confidence: number
    tic_id: number
    snr_tls?: number | null
    odd_even_mismatch?: number | null
    centroid_ratio?: number | null
    n_transits?: number | null
  }
}

export default function ClassificationCard({ result }: Props) {
  const cls = result.predicted_class as ClassKey
  const cfg = CLASS_CONFIG[cls] || CLASS_CONFIG.OTHER
  const probs = result.class_probs

  return (
    <div className="glass-card p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">Classification</h2>
        <span className={`px-3 py-1 rounded-full text-xs font-bold ${cfg.badgeClass}`}>
          {cls}
        </span>
      </div>

      {/* Primary class */}
      <div className="text-center mb-6">
        <div className="font-display font-bold text-5xl mb-1" style={{ color: cfg.color }}>
          {(result.confidence * 100).toFixed(1)}%
        </div>
        <div className="text-white font-semibold text-lg">{cfg.label}</div>
        <div className="text-gray-500 text-xs mt-1">{cfg.desc}</div>
      </div>

      {/* Probability bars */}
      <div className="space-y-3">
        {(Object.entries(probs) as [ClassKey, number][])
          .sort((a, b) => b[1] - a[1])
          .map(([c, p]) => (
          <div key={c}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">{CLASS_CONFIG[c].label}</span>
              <span className="font-mono text-gray-300">{(p * 100).toFixed(2)}%</span>
            </div>
            <div className="progress-bar">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${p * 100}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className="progress-fill"
                style={{ background: `linear-gradient(90deg, ${CLASS_CONFIG[c].color}80, ${CLASS_CONFIG[c].color})` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Diagnostic flags */}
      <div className="mt-5 grid grid-cols-2 gap-3 pt-5 border-t border-space-700">
        <div>
          <div className="text-xs text-gray-500">TLS SDE</div>
          <div className="font-mono text-nebula-blue font-semibold">
            {result.snr_tls?.toFixed(2) ?? '—'}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Odd-Even Δ</div>
          <div className="font-mono text-nebula-blue font-semibold">
            {result.odd_even_mismatch?.toFixed(3) ?? '—'}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Centroid Ratio</div>
          <div className="font-mono text-nebula-blue font-semibold">
            {result.centroid_ratio?.toFixed(3) ?? '—'}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">N Transits</div>
          <div className="font-mono text-nebula-blue font-semibold">
            {result.n_transits ?? '—'}
          </div>
        </div>
      </div>
    </div>
  )
}
