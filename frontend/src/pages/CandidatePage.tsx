import { useParams, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { ExternalLink, ArrowLeft, Loader2, AlertTriangle, Globe, Thermometer, Zap, Star } from 'lucide-react'
import axios from 'axios'
import PageTransition from '../components/layout/PageTransition'
// Full detail response from /api/candidate/{tic_id}
interface CandidateDetail {
  tic_id: number
  sector: number
  predicted_class: string
  class_probs: { TRANSIT: number; EB: number; BLEND: number; OTHER: number }
  confidence: number
  conformal_class_set: string[] | null
  period: number | null
  period_err: number | null
  duration_hrs: number | null
  depth: number | null
  depth_ppm: number | null
  snr_tls: number | null
  snr_photometric: number | null
  n_transits: number | null
  odd_even_mismatch: number | null
  rp_rearth: number | null
  t_eq_kelvin: number | null
  stellar: {
    host_name: string | null
    teff: number | null
    logg: number | null
    stellar_mass: number | null
    stellar_radius: number | null
    tmag: number | null
    ra: number | null
    dec: number | null
    distance_pc: number | null
    luminosity_lsun: number | null
  } | null
  habitability: {
    esi_score: number
    hz_class: string
    priority_score: number
    tier: number
    rv_amplitude_ms: number | null
    in_confirmed_catalog: boolean
  } | null
  xai: {
    top_shap_features: { name: string; value: number; shap_value: number }[]
    attention_map_b64: string | null
  } | null
  phase_fold_global: number[] | null
  phase_fold_local: number[] | null
  batman_model: number[] | null
  processing_time_s: number
  error: string | null
}

async function fetchCandidateDetail(tic_id: string, sector: number): Promise<CandidateDetail> {
  const { data } = await axios.get(`/api/candidate/${tic_id}`, { params: { sector } })
  return data
}

const CLASS_COLORS: Record<string, string> = {
  TRANSIT: '#4ade80', EB: '#fb923c', BLEND: '#f87171', OTHER: '#94a3b8'
}

const HZ_LABELS: Record<string, { label: string; color: string }> = {
  CONSERVATIVE: { label: 'Conservative HZ', color: '#4ade80' },
  INNER: { label: 'Inner Optimistic HZ', color: '#fbbf24' },
  OUTER: { label: 'Outer Optimistic HZ', color: '#60a5fa' },
  NONE: { label: 'Outside HZ', color: '#6b7280' },
}

function MiniLightCurve({ data, model, label }: { data: number[]; model?: number[]; label: string }) {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const W = 400, H = 80
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${H - ((v - min) / range) * (H - 4) - 2}`)
  const modelPts = model?.map((v, i) => `${(i / (model.length - 1)) * W},${H - ((v - min) / range) * (H - 4) - 2}`)

  return (
    <div>
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2 font-mono">{label}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded" style={{ background: 'rgba(0,0,0,0.4)' }}>
        <polyline points={pts.join(' ')} fill="none" stroke="#4fc3f7" strokeWidth="1" opacity="0.7" />
        {modelPts && (
          <polyline points={modelPts.join(' ')} fill="none" stroke="#4ade80" strokeWidth="1.5" opacity="0.9" />
        )}
      </svg>
    </div>
  )
}

function ESIGauge({ esi }: { esi: number }) {
  const pct = Math.round(esi * 100)
  const color = esi >= 0.8 ? '#4ade80' : esi >= 0.5 ? '#fbbf24' : '#f87171'
  const r = 38, cx = 50, cy = 50
  const circ = 2 * Math.PI * r
  const dash = (esi * circ).toFixed(2)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
        <circle
          cx={cx} cy={cy} r={r} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: 'stroke-dasharray 1s ease' }}
        />
        <text x="50" y="46" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">{pct}</text>
        <text x="50" y="60" textAnchor="middle" fill="#94a3b8" fontSize="8">ESI</text>
      </svg>
    </div>
  )
}

export default function CandidatePage() {
  const { ticId } = useParams<{ ticId: string }>()
  const [sector] = [1] // default sector — could add UI for sector selection

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['candidate-detail', ticId],
    queryFn: () => fetchCandidateDetail(ticId!, 1),
    enabled: !!ticId,
    retry: 1,
  })

  if (isLoading) return (
    <div className="flex items-center justify-center min-h-screen gap-3">
      <Loader2 className="animate-spin text-[#BAE6FD]" size={24} />
      <span className="text-[#93C5FD]/60 ep-mono">Running ECLIPSE pipeline for TIC {ticId}...</span>
    </div>
  )

  if (isError || !data) return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4 text-center px-6">
      <AlertTriangle size={40} className="text-[#F44336]" />
      <div className="text-[#F44336]/80 ep-mono text-sm">
        {(error as any)?.response?.data?.detail || `TIC ${ticId} not found or pipeline failed`}
      </div>
      <Link to="/catalog" className="text-[#BAE6FD] text-sm hover:underline ep-dsp uppercase tracking-widest font-semibold">← Back to Catalog</Link>
    </div>
  )

  const color = CLASS_COLORS[data.predicted_class] || '#94a3b8'
  const hz = data.habitability ? HZ_LABELS[data.habitability.hz_class] || HZ_LABELS.NONE : null

  return (
    <PageTransition className="relative min-h-screen flex flex-col z-0">
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-20 w-full flex-1">

        {/* Header */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-12">
          <div className="flex justify-between items-center mb-8">
            <Link to="/catalog" className="inline-flex items-center gap-2 text-sm text-white/50 hover:text-white transition-colors ep-dsp uppercase tracking-widest font-semibold drop-shadow-md">
              <ArrowLeft size={16} /> Back to Catalog
            </Link>
            <a
              href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/report/${data.tic_id}`}

              download
              className="inline-flex items-center gap-2 bg-[#1B6EE8]/20 hover:bg-[#1B6EE8]/40 border border-[#1B6EE8]/50 text-[#BAE6FD] px-4 py-2 rounded-lg text-[10px] ep-dsp uppercase tracking-widest font-semibold transition-colors shadow-[0_0_15px_rgba(27,110,232,0.2)] hover:shadow-[0_0_25px_rgba(27,110,232,0.4)]"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
              Download PDF Report
            </a>
          </div>
          <div className="flex flex-wrap items-center gap-6 mb-2">
            <h1 className="ep-dsp font-black text-6xl md:text-7xl drop-shadow-[0_0_30px_rgba(255,255,255,0.4)] text-white">TIC {data.tic_id}</h1>
            <span className="px-3 py-1 rounded-full text-sm font-bold"
              style={{ background: `${color}25`, color, border: `1px solid ${color}60` }}>
              {data.predicted_class}
            </span>
            {data.habitability?.tier && data.predicted_class === 'TRANSIT' && (
              <span className="px-3 py-1 rounded-full text-xs font-bold bg-yellow-500/10 text-yellow-400 border border-yellow-500/30">
                TIER {data.habitability.tier}
              </span>
            )}
            {data.habitability?.in_confirmed_catalog && (
              <span className="px-4 py-1.5 rounded-full text-xs font-bold bg-[#1FAD73]/20 text-[#1FAD73] border border-[#1FAD73]/40 shadow-[0_0_15px_rgba(31,173,115,0.3)]">
                CONFIRMED PLANET
              </span>
            )}
          </div>
          <p className="text-white/60 text-lg ep-dsp tracking-wide drop-shadow-md">
            Sector {data.sector}
            {data.stellar?.host_name && ` · ${data.stellar.host_name}`}
            {data.stellar?.ra != null && data.stellar?.dec != null &&
              ` · RA ${data.stellar.ra.toFixed(4)}° Dec ${data.stellar.dec.toFixed(4)}°`}
          </p>
          <div className="flex items-center gap-6 mt-4">
            <a href={`https://exofop.ipac.caltech.edu/tess/target.php?id=${data.tic_id}`}
              target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-[10px] text-[#BAE6FD] uppercase tracking-[0.2em] font-bold hover:text-white transition-colors ep-dsp drop-shadow-md">
              ExoFOP <ExternalLink size={14} />
            </a>
            <a href={`https://vizier.cds.unistra.fr/viz-bin/VizieR?source=IV/39&target=TIC+${data.tic_id}`}
              target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-[10px] text-[#BAE6FD] uppercase tracking-[0.2em] font-bold hover:text-white transition-colors ep-dsp drop-shadow-md">
              TIC Catalog <ExternalLink size={14} />
            </a>
            <span className="text-xs text-[#93C5FD]/60 ep-mono font-medium drop-shadow-md">
              {data.processing_time_s.toFixed(1)}s processing time
            </span>
          </div>
        </motion.div>

        {/* Error banner */}
        {data.error && (
          <div className="bg-[#F44336]/20 border border-[#F44336] rounded-xl p-6 mb-8 flex gap-4 backdrop-blur-md">
            <AlertTriangle size={24} className="text-[#F44336] shrink-0 mt-0.5" />
            <span className="text-white drop-shadow-md font-medium text-lg">{data.error}</span>
          </div>
        )}

        {/* Top row: Classification + Habitability */}
        <div className="grid lg:grid-cols-3 gap-8 mb-8">

          {/* Classification probabilities */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] lg:col-span-2 relative overflow-hidden">
            {/* Decorative glow */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-[#1B6EE8]/5 rounded-full blur-[80px] pointer-events-none -mr-20 -mt-20"></div>

            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-6 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Classification Profile</h2>
            <div className="flex items-end gap-4 mb-8">
              <div className="text-6xl md:text-7xl font-black ep-dsp drop-shadow-[0_0_20px_rgba(255,255,255,0.2)]" style={{ color }}>
                {(data.confidence * 100).toFixed(1)}<span className="text-3xl text-white/40">%</span>
              </div>
              <div className="pb-2">
                <div className="text-2xl font-bold text-white ep-dsp drop-shadow-md">{data.predicted_class}</div>
                {data.conformal_class_set && (
                  <div className="text-sm text-white/50 mt-1 ep-dsp font-medium drop-shadow-md">
                    Conformal set: [{data.conformal_class_set.join(', ')}]
                  </div>
                )}
              </div>
            </div>
            <div className="space-y-4">
              {Object.entries(data.class_probs)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, prob]) => (
                  <div key={cls}>
                    <div className="flex justify-between text-sm mb-2 drop-shadow-md ep-dsp font-medium">
                      <span className="text-white/70 uppercase tracking-widest">{cls}</span>
                      <span className="text-white">{(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${prob * 100}%` }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                        className="h-full rounded-full"
                        style={{ background: `linear-gradient(90deg, ${CLASS_COLORS[cls]}60, ${CLASS_COLORS[cls]})` }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </motion.div>

          {/* Habitability / ESI */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.1 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)]">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-6 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Habitability</h2>
            {data.habitability && data.predicted_class === 'TRANSIT' ? (
              <>
                <div className="flex justify-center mb-4">
                  <ESIGauge esi={data.habitability.esi_score} />
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">HZ Status</span>
                    <span className="font-mono text-xs px-2 py-0.5 rounded"
                      style={{ background: `${hz?.color}20`, color: hz?.color, border: `1px solid ${hz?.color}40` }}>
                      {hz?.label}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Priority</span>
                    <span className="font-mono" style={{ color: data.habitability.priority_score > 0.6 ? '#4ade80' : '#94a3b8' }}>
                      {(data.habitability.priority_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  {data.habitability.rv_amplitude_ms != null && (
                    <div className="flex justify-between">
                      <span className="text-gray-500 flex items-center gap-1"><Zap size={12} />RV K</span>
                      <span className="font-mono text-gray-300">{data.habitability.rv_amplitude_ms.toFixed(3)} m/s</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-gray-600 text-sm text-center py-6">
                Habitability computed only for<br />TRANSIT-class signals
              </div>
            )}
          </motion.div>
        </div>

        {/* Transit Parameters */}
        {data.period != null && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.15 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-8 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Transit Parameters</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-6">
              {[
                { label: 'Period', value: data.period?.toFixed(5), unit: 'd' },
                { label: '± Err', value: data.period_err?.toFixed(5), unit: 'd' },
                { label: 'Duration', value: data.duration_hrs?.toFixed(3), unit: 'hrs' },
                { label: 'Depth', value: data.depth_ppm?.toFixed(1), unit: 'ppm' },
                { label: 'TLS SDE', value: data.snr_tls?.toFixed(2), unit: 'σ' },
                { label: 'N transits', value: data.n_transits, unit: '' },
                { label: 'Rp', value: data.rp_rearth?.toFixed(2), unit: 'R⊕' },
                { label: 'Teq', value: data.t_eq_kelvin?.toFixed(0), unit: 'K' },
                { label: 'Phot SNR', value: data.snr_photometric?.toFixed(2), unit: 'σ' },
                { label: 'Odd/Even', value: data.odd_even_mismatch?.toFixed(4), unit: '' },
                { label: 'Centroid', value: (data as any).centroid_ratio?.toFixed(4), unit: '' },
                { label: 'Rp/Rs', value: data.depth != null ? Math.sqrt(data.depth).toFixed(4) : null, unit: '' },
              ].map(({ label, value, unit }) => (
                <div key={label} className="bg-[#050B14]/60 rounded-[16px] p-4 border border-[#3B6A9A]/30 shadow-inner">
                  <div className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-widest mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">{label}</div>
                  <div className="font-light ep-dsp text-2xl text-white drop-shadow-md">
                    {value ?? '—'}
                    {value && unit && <span className="text-[#93C5FD]/60 text-sm ml-1 font-medium">{unit}</span>}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Phase fold charts */}
        {(data.phase_fold_local || data.phase_fold_global) && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.2 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-8 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Phase-Folded Light Curve</h2>
            <div className="grid md:grid-cols-2 gap-8">
              {data.phase_fold_local && (
                <MiniLightCurve
                  data={data.phase_fold_local}
                  model={data.batman_model ?? undefined}
                  label="Local view (transit) · green = batman fit"
                />
              )}
              {data.phase_fold_global && (
                <MiniLightCurve
                  data={data.phase_fold_global}
                  label="Global view (full orbit)"
                />
              )}
            </div>
          </motion.div>
        )}

        {/* Stellar Parameters */}
        {data.stellar && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.25 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-8 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)] flex items-center gap-2">
              <Star size={16} /> Host Star
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
              {[
                { label: 'Teff', value: data.stellar.teff?.toFixed(0), unit: 'K' },
                { label: 'log g', value: data.stellar.logg?.toFixed(2), unit: 'cgs' },
                { label: 'Mass', value: data.stellar.stellar_mass?.toFixed(3), unit: 'M☉' },
                { label: 'Radius', value: data.stellar.stellar_radius?.toFixed(3), unit: 'R☉' },
                { label: 'Luminosity', value: data.stellar.luminosity_lsun?.toFixed(3), unit: 'L☉' },
                { label: 'Distance', value: data.stellar.distance_pc?.toFixed(1), unit: 'pc' },
                { label: 'TESS mag', value: data.stellar.tmag?.toFixed(2), unit: '' },
              ].map(({ label, value, unit }) => (
                <div key={label} className="bg-[#050B14]/60 rounded-[16px] p-4 border border-[#3B6A9A]/30 shadow-inner">
                  <div className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-widest mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">{label}</div>
                  <div className="font-light ep-dsp text-2xl text-white drop-shadow-md">
                    {value ?? '—'}
                    {value && unit && <span className="text-[#93C5FD]/60 text-sm ml-1 font-medium">{unit}</span>}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Centroid Validation */}
        {(data as any).centroid_map_b64 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.28 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-6 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">
              Astrometric Centroid Validation
            </h2>
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div>
                <p className="text-sm font-light text-[#BAE6FD] leading-relaxed tracking-wide mb-4">
                  This diagram maps the photometric centroid position during out-of-transit (blue) versus in-transit (green) cadences.
                </p>
                <p className="text-sm font-light text-[#BAE6FD] leading-relaxed tracking-wide">
                  A significant offset would indicate a background eclipsing binary blend rather than a true planetary transit. The tight clustering shown here strongly validates this target.
                </p>
              </div>
              <div className="flex justify-center">
                <img
                  src={`data:image/png;base64,${(data as any).centroid_map_b64}`}
                  alt="Centroid validation map"
                  className="w-full max-w-sm rounded-[24px] opacity-90 border border-[#3B6A9A]/40 shadow-[0_10px_30px_rgba(0,5,15,0.6)]"
                />
              </div>
            </div>
          </motion.div>
        )}

        {/* SHAP Feature Importance */}
        {data.xai?.top_shap_features && data.xai.top_shap_features.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.3 } }}
            className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8">
            <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-8 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">
              SHAP Feature Importance (top {data.xai.top_shap_features.length})
            </h2>
            <div className="space-y-4">
              {data.xai.top_shap_features.map((f, i) => {
                const maxShap = Math.max(...data.xai!.top_shap_features.map(x => Math.abs(x.shap_value)))
                const pct = maxShap > 0 ? (Math.abs(f.shap_value) / maxShap) * 100 : 0
                const positive = f.shap_value >= 0
                return (
                  <div key={i} className="flex items-center gap-3">
                    <div className="text-xs text-gray-400 font-mono w-40 shrink-0 truncate">{f.name}</div>
                    <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.6, delay: i * 0.05 }}
                        className="h-full rounded-full"
                        style={{ background: positive ? '#4ade80' : '#f87171' }}
                      />
                    </div>
                    <div className="text-sm ep-dsp font-medium text-white/70 w-16 text-right drop-shadow-md">
                      {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                    </div>
                  </div>
                )
              })}
            </div>
            {data.xai.attention_map_b64 && (
              <div className="mt-8">
                <div className="text-[10px] text-[#BAE6FD] mb-4 ep-dsp font-semibold uppercase tracking-[0.3em] drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Attention Map</div>
                <img
                  src={`data:image/png;base64,${data.xai.attention_map_b64}`}
                  alt="Attention heatmap"
                  className="w-full rounded-[24px] opacity-90 border border-[#3B6A9A]/40 shadow-[0_10px_30px_rgba(0,5,15,0.6)]"
                />
              </div>
            )}
          </motion.div>
        )}

      </div>
    </PageTransition>
  )
}
