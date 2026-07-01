import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Telescope, Play, Loader2 } from 'lucide-react'
import axios from 'axios'
import PageTransition from '../components/layout/PageTransition'
import { useNavigate } from 'react-router-dom'

interface Candidate {
  tic_id: number;
  sector: number;
  esi_score?: number;
  hz_class?: string;
  tier?: number;
  period?: number;
  rp_rearth?: number;
}

export default function SectorPage() {
  const navigate = useNavigate()
  const [sector, setSector] = useState(1)
  const [maxTIC, setMaxTIC] = useState(500)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [processed, setProcessed] = useState(0)
  const [total, setTotal] = useState(0)
  const [found, setFound] = useState(0)
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (status === 'done') {
      axios.get('/api/candidates').then(res => setCandidates(res.data.candidates || []))
    }
  }, [status])

  const startProcessing = async () => {
    setStatus('running')
    setProgress(0)
    setProcessed(0)
    setFound(0)
    setCandidates([])

    try {
      const { data } = await axios.post('/api/sector/process', { sector, max_tic: maxTIC })
      const jid = data.job_id
      setJobId(jid)

      // Connect WebSocket for live progress
      const WS_URL = import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL.replace('http', 'ws') : 'ws://localhost:8000';
      const ws = new WebSocket(`${WS_URL}/ws/job/${jid}`)

      wsRef.current = ws
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        setProgress(msg.progress ?? 0)
        setProcessed(msg.processed ?? 0)
        setTotal(msg.total ?? 0)
        setFound(msg.found ?? 0)
        if (msg.status === 'done' || msg.status === 'completed') { setStatus('done'); ws.close() }
        if (msg.status === 'error') { setStatus('error'); ws.close() }
      }
    } catch {
      setStatus('error')
    }
  }

  return (
    <PageTransition className="relative min-h-screen flex flex-col z-0">
      <div className="flex-1 w-full max-w-4xl mx-auto px-6 py-20 z-10">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-16 text-center">
          <h1 className="ep-h1 mb-2 text-white drop-shadow-[0_4px_20px_rgba(147,197,253,0.3)]">SECTOR PROCESSING</h1>
          <p className="ep-body mx-auto uppercase tracking-widest text-xs text-[#BAE6FD]">Run ECLIPSE-PRIME batch inference on an entire TESS sector</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-[#07101E]/40 backdrop-blur-2xl p-10 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_40px_100px_rgba(0,5,15,0.8)] flex flex-col gap-10 max-w-2xl mx-auto"
        >
          {/* Subtle bluish ambient glow inside the container */}
          <div className="absolute inset-0 bg-gradient-to-b from-[#1B6EE8]/5 to-transparent pointer-events-none rounded-[32px]"></div>

          <div className="relative z-10 grid sm:grid-cols-2 gap-10">
            <div>
              <label className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-[0.2em] block mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">TESS Sector</label>
              <input
                type="number" min={1} max={80}
                value={sector}
                onChange={e => setSector(parseInt(e.target.value))}
                className="w-full bg-[#050B14]/60 border border-[#3B6A9A]/30 px-6 py-4 text-xl text-white ep-dsp font-medium rounded-xl focus:outline-none focus:border-[#93C5FD] focus:bg-[#07101E]/80 transition-colors shadow-inner"
              />
            </div>
            <div>
              <label className="text-[10px] text-[#BAE6FD] ep-dsp font-semibold uppercase tracking-[0.2em] block mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Max Targets (limit for demo)</label>
              <input
                type="number" step={100}
                value={maxTIC}
                onChange={e => setMaxTIC(parseInt(e.target.value))}
                className="w-full bg-[#050B14]/60 border border-[#3B6A9A]/30 px-6 py-4 text-xl text-white ep-dsp font-medium rounded-xl focus:outline-none focus:border-[#93C5FD] focus:bg-[#07101E]/80 transition-colors shadow-inner"
              />
            </div>
          </div>

          <motion.button
            whileHover={{ backgroundColor: 'rgba(59,106,154,0.2)' }}
            whileTap={{ scale: 0.99 }}
            onClick={startProcessing}
            disabled={status === 'running'}
            className="w-full rounded-xl bg-[#0B172A]/50 border border-[#3B6A9A]/50 text-[#E0E7FF] py-4 flex items-center justify-center gap-3 ep-dsp tracking-[0.15em] uppercase font-semibold text-sm transition-colors shadow-[0_0_15px_rgba(59,106,154,0.1)] hover:border-[#93C5FD]/70 hover:text-white hover:shadow-[0_0_25px_rgba(147,197,253,0.3)] disabled:opacity-50 disabled:hover:bg-[#0B172A]/50"
          >
            {status === 'running' ? (
              <><Loader2 size={18} className="animate-spin text-[#93C5FD]" /> PROCESSING SECTOR {sector}</>
            ) : (
              <><Play size={18} className="text-[#93C5FD]" /> INITIALIZE PIPELINE</>
            )}
          </motion.button>
        </motion.div>

        {/* Progress display */}
        {status !== 'idle' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative bg-[#07101E]/40 backdrop-blur-2xl border border-[#3B6A9A]/30 p-8 rounded-[24px] mt-8 max-w-2xl mx-auto shadow-[0_20px_50px_rgba(0,5,15,0.5)]"
          >
            <div className="flex justify-between items-end mb-6">
              <div>
                <span className="text-[#93C5FD]/60 text-[10px] uppercase tracking-[0.2em] font-semibold ep-dsp block mb-1">STATUS</span>
                <span className="text-white text-xl font-medium ep-dsp tracking-widest uppercase drop-shadow-[0_2px_10px_rgba(255,255,255,0.2)]">SECTOR {sector} PIPELINE</span>
              </div>
              <span className={`ep-dsp text-[10px] px-3 py-1.5 rounded-lg uppercase tracking-widest font-bold ${status === 'done' ? 'bg-[#1FAD73]/20 text-[#4ade80] border border-[#1FAD73]/30' :
                status === 'error' ? 'bg-red-900/40 text-red-400 border border-red-500/30' :
                  'bg-[#1B6EE8]/20 text-[#93C5FD] border border-[#1B6EE8]/30 animate-pulse'
                }`}>
                {status}
              </span>
            </div>

            <div className="h-2 w-full bg-[#050B14] rounded-full overflow-hidden mb-4 border border-[#3B6A9A]/20">
              <motion.div
                className="h-full bg-gradient-to-r from-[#3B6A9A] via-[#93C5FD] to-[#ffffff] shadow-[0_0_10px_rgba(147,197,253,0.5)]"
                animate={{ width: `${progress * 100}%` }}
                transition={{ ease: "linear" }}
              />
            </div>

            <div className="flex justify-between text-[11px] uppercase tracking-[0.1em] text-[#93C5FD]/80 ep-dsp font-semibold">
              <span>{processed} / {total} TIC IDs &nbsp;|&nbsp; <span className="text-white">{found} CANDIDATES</span></span>
              <span className="text-white">{(progress * 100).toFixed(1)}%</span>
            </div>

            {status === 'done' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 pt-6 border-t border-[#3B6A9A]/20"
              >
                <div className="text-center text-[#4ade80] text-[11px] ep-dsp tracking-[0.2em] uppercase font-bold drop-shadow-[0_0_10px_rgba(74,222,128,0.2)] mb-8">
                  PIPELINE EXECUTION COMPLETE.
                </div>

                {candidates.length > 0 && (
                  <div className="text-left">
                    <h3 className="ep-dsp text-[10px] text-[#BAE6FD] font-semibold tracking-[0.2em] uppercase mb-4 drop-shadow-md">
                      Newly Discovered Candidates
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      {candidates.slice(0, 10).map((c, i) => (
                        <div key={i} className="bg-[#050B14]/80 border border-[#1FAD73]/40 p-4 rounded-xl shadow-[0_0_15px_rgba(31,173,115,0.15)] flex flex-col justify-between hover:bg-[#1B6EE8]/10 transition-colors">
                          <div className="flex justify-between items-start mb-3">
                            <span className="text-white ep-dsp font-medium text-sm">TIC {c.tic_id}</span>
                            {c.esi_score ? (
                              <span className="text-[9px] text-[#4ade80] bg-[#1FAD73]/20 px-2 py-1 rounded-md ep-dsp tracking-wider">ESI: {c.esi_score.toFixed(2)}</span>
                            ) : null}
                          </div>
                          <div className="text-[10px] text-gray-400 ep-dsp flex flex-col gap-1.5 mb-4">
                            <span>Period: <span className="text-white">{c.period?.toFixed(2)} days</span></span>
                            <span>Radius: <span className="text-white">{c.rp_rearth?.toFixed(2)} R⊕</span></span>
                            <span className="text-[#93C5FD]">Zone: {c.hz_class || 'N/A'}</span>
                          </div>
                          <button
                            onClick={() => navigate(`/candidate/${c.tic_id}`)}
                            className="w-full text-[9px] ep-dsp uppercase tracking-widest bg-[#1B6EE8]/20 hover:bg-[#1B6EE8]/40 text-[#BAE6FD] py-2.5 rounded-lg transition-colors border border-[#1B6EE8]/30"
                          >
                            View Analysis
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </motion.div>
        )}
      </div>
    </PageTransition>
  )
}
