import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { Download, Search, Filter } from 'lucide-react'
import axios from 'axios'
import PageTransition from '../components/layout/PageTransition'
interface Candidate {
  id: number
  tic_id: number
  sector: number
  predicted_class: string
  prob_transit: number | null
  prob_eb: number | null
  prob_blend: number | null
  prob_other: number | null
  confidence: number | null
  period: number | null
  duration: number | null
  depth: number | null
  snr_tls: number | null
}

async function fetchCandidates(params: Record<string, any>) {
  const { data } = await axios.get('/api/candidates', { params })
  return data as { total: number; candidates: Candidate[] }
}

const CLASS_COLORS: Record<string, string> = {
  TRANSIT: 'badge-transit',
  EB:      'badge-eb',
  BLEND:   'badge-blend',
  OTHER:   'badge-other',
}

export default function CatalogPage() {
  const [filterClass, setFilterClass] = useState<string>('')
  const [filterSector, setFilterSector] = useState<string>('')
  const [minSnr, setMinSnr] = useState(0)

  const { data, isLoading } = useQuery({
    queryKey: ['candidates', filterClass, filterSector, minSnr],
    queryFn: () => fetchCandidates({
      predicted_class: filterClass || undefined,
      sector: filterSector ? parseInt(filterSector) : undefined,
      min_snr: minSnr,
      limit: 100
    })
  })

  return (
    <PageTransition className="relative min-h-screen flex flex-col z-0">
      <div className="relative z-10 min-h-screen max-w-7xl mx-auto px-6 py-20 w-full">
      {/* Header */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-12">
        <h1 className="ep-h1 text-white mb-2 drop-shadow-[0_4px_20px_rgba(147,197,253,0.3)]">Candidate Catalog</h1>
        <p className="ep-body text-[#BAE6FD]">All ECLIPSE-PRIME detections from processed TESS sectors</p>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-[#07101E]/40 border border-[#3B6A9A]/30 backdrop-blur-2xl shadow-[0_20px_50px_rgba(0,5,15,0.5)] rounded-2xl p-5 mb-6 flex flex-wrap gap-4 items-end"
      >
        <div>
          <label className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-[0.2em] block mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Class</label>
          <select
            value={filterClass}
            onChange={e => setFilterClass(e.target.value)}
            className="bg-[#050B14]/60 border border-[#3B6A9A]/30 rounded-lg px-3 py-2 text-sm text-white ep-mono
                       focus:outline-none focus:border-[#93C5FD]"
          >
            <option value="">All Classes</option>
            <option value="TRANSIT">TRANSIT</option>
            <option value="EB">EB</option>
            <option value="BLEND">BLEND</option>
            <option value="OTHER">OTHER</option>
          </select>
        </div>
        <div>
          <label className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-[0.2em] block mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Sector</label>
          <input
            type="number" min={1} max={80}
            value={filterSector}
            onChange={e => setFilterSector(e.target.value)}
            placeholder="All"
            className="bg-[#050B14]/60 border border-[#3B6A9A]/30 rounded-lg px-3 py-2 text-sm text-white ep-mono w-24
                       focus:outline-none focus:border-[#93C5FD]"
          />
        </div>
        <div>
          <label className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-[0.2em] block mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Min TLS SDE</label>
          <input
            type="number" min={0} step={1}
            value={minSnr}
            onChange={e => setMinSnr(parseFloat(e.target.value) || 0)}
            className="bg-[#050B14]/60 border border-[#3B6A9A]/30 rounded-lg px-3 py-2 text-sm text-white ep-mono w-20
                       focus:outline-none focus:border-[#93C5FD]"
          />
        </div>
        <div className="ml-auto text-xs text-[#93C5FD]/60 ep-mono self-center font-bold">
          {data?.total ?? 0} total candidates
        </div>
      </motion.div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="bg-[#07101E]/40 border border-[#3B6A9A]/30 backdrop-blur-2xl shadow-[0_20px_50px_rgba(0,5,15,0.5)] rounded-2xl overflow-hidden"
      >
        <table className="w-full text-left">
          <thead className="bg-[#050B14]/80 border-b border-[#3B6A9A]/40 text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em]">
            <tr>
              <th className="py-4 px-6 font-semibold">TIC ID</th>
              <th className="py-4 px-6 font-semibold text-center">Sector</th>
              <th className="py-4 px-6 font-semibold text-center">Class</th>
              <th className="py-4 px-6 font-semibold text-center">Confidence</th>
              <th className="py-4 px-6 font-semibold text-center">Period (d)</th>
              <th className="py-4 px-6 font-semibold text-center">Depth</th>
              <th className="py-4 px-6 font-semibold text-center">TLS SDE</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={7} className="text-center py-12 text-gray-600">
                  <div className="animate-spin w-6 h-6 border-2 border-nebula-blue/30 border-t-nebula-blue rounded-full mx-auto" />
                </td>
              </tr>
            )}
            {!isLoading && data?.candidates.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-12 text-[#93C5FD]/60 ep-mono">
                  No candidates found. Run sector processing first.
                </td>
              </tr>
            )}
            {data?.candidates.map((c, i) => (
              <motion.tr
                key={c.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.02 }}
                className="cursor-pointer border-b border-[#3B6A9A]/20 hover:bg-[#3B6A9A]/10 transition-colors text-white text-sm"
                onClick={() => window.location.href = `/candidate/${c.tic_id}`}
              >
                <td className="font-mono text-[#93C5FD] py-4 px-6">{c.tic_id}</td>
                <td className="text-center">{c.sector}</td>
                <td className="text-center">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${CLASS_COLORS[c.predicted_class] || 'badge-other'}`}>
                    {c.predicted_class}
                  </span>
                </td>
                <td className="text-center font-mono">{c.confidence != null ? `${(c.confidence * 100).toFixed(1)}%` : '—'}</td>
                <td className="text-center font-mono">{c.period?.toFixed(4) ?? '—'}</td>
                <td className="text-center font-mono">{c.depth?.toFixed(6) ?? '—'}</td>
                <td className="text-center font-mono">{c.snr_tls?.toFixed(2) ?? '—'}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </div>
  </PageTransition>
  )
}
