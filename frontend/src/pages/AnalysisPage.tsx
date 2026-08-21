import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Check, ChevronDown, RefreshCw, ArrowRight, AlertCircle, Sparkles } from 'lucide-react'
import axios from 'axios'
import PageTransition from '../components/layout/PageTransition'
import { GlossaryTerm } from '../components/global/AstroGlossary'

// -- API Interfaces --
interface PredictResponse {
  tic_id: number
  sector: number
  predicted_class: string
  class_probs: { TRANSIT: number; EB: number; BLEND: number; OTHER: number }
  confidence: number
  conformal_class_set: string[]
  in_conformal_90: boolean
  period: number | null
  duration_days: number | null
  depth_ppm: number | null
  t_eq_kelvin: number | null
  rp_rearth: number | null
  habitability: { esi_score: number; tier: number } | null
  error: string | null
}

const FAMOUS_TARGETS = [
  { name: 'π Mensae (TOI-144)', tic: '279741379', sector: 1 },
  { name: 'WASP-18 (Hot Jupiter)', tic: '100100827', sector: 2 },
  { name: 'TOI-700 d (Habitable)', tic: '220397947', sector: 1 },
  { name: 'LHS 3844 b (Rocky)', tic: '391666931', sector: 1 },
  { name: 'TOI-270 b (Mini-Neptune)', tic: '311092062', sector: 3 },
  { name: 'L 98-59 b (Venus-like)', tic: '261136679', sector: 2 },
  { name: 'WASP-121 b (Ultra-Hot)', tic: '238022134', sector: 1 },
  { name: 'TOI-125 b (Sub-Neptune)', tic: '261136246', sector: 1 },
  { name: 'LTT 1445A b (M-Dwarf)', tic: '410153553', sector: 2 },
  { name: 'TOI-216 b (Gas Giant)', tic: '120075081', sector: 1 }
]

const PIPELINE_STAGES = [
  { id: 'ingest', label: 'Ingest', activeText: 'Fetching PDCSAP flux from MAST archive...' },
  { id: 'denoise', label: 'Denoise', activeText: 'Denoising: sigma-clipping 3σ...' },
  { id: 'detect', label: 'Detect', activeText: 'TLS period search: testing trial frequencies...' },
  { id: 'classify', label: 'Classify', activeText: 'Running ECLIPSE-PRIME dual-stream cross-attention...' },
  { id: 'habitability', label: 'Assess', activeText: 'Computing equilibrium temperature and ESI...' }
]

export default function AnalysisPage() {
  const navigate = useNavigate()

  // Form State
  const [ticInput, setTicInput] = useState('')
  const [sector, setSector] = useState(1)
  const [isSectorDropdownOpen, setIsSectorDropdownOpen] = useState(false)

  // Pipeline State: 'setup' | 'analyzing' | 'result'
  const [viewState, setViewState] = useState<'setup' | 'analyzing' | 'result'>('setup')
  const [liveMast, setLiveMast] = useState(false)
  const [currentStageIndex, setCurrentStageIndex] = useState(-1)
  const [apiResult, setApiResult] = useState<PredictResponse | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)

  // Trigger analysis
  const beginAnalysis = async () => {
    if (!ticInput) return
    setViewState('analyzing')
    setApiResult(null)
    setApiError(null)
    setCurrentStageIndex(0)

    const ticNum = parseInt(ticInput)
    let fetchedResult: PredictResponse | null = null
    let fetchedError: string | null = null
    let apiDone = false

    // 1. Kick off API call
    axios.post('/api/predict', { tic_id: ticNum, sector, live_mast: liveMast })
      .then(res => {
        fetchedResult = res.data
        apiDone = true
      })
      .catch(err => {
        fetchedError = err.response?.data?.detail || err.message || 'Analysis failed'
        apiDone = true
      })

    // 2. Step through animated pipeline stages
    for (let i = 0; i < 5; i++) {
      setCurrentStageIndex(i)
      await new Promise(r => setTimeout(r, liveMast ? 2000 : 1200))
    }

    // 3. Wait for API to finish (with 30s timeout)
    const deadline = Date.now() + 30_000
    while (!apiDone && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 200))
    }
    if (!apiDone) {
      fetchedError = 'Pipeline timed out. The target might require live MAST download.'
    }

    setCurrentStageIndex(5)
    if (fetchedError) {
      setApiError(fetchedError)
    } else {
      setApiResult(fetchedResult)
    }

    // 4. Reveal result dashboard
    await new Promise(r => setTimeout(r, 400))
    setViewState('result')
  }

  const resetAnalysis = () => {
    setViewState('setup')
    setApiResult(null)
    setApiError(null)
    setCurrentStageIndex(-1)
  }

  return (
    <PageTransition className="relative min-h-screen flex flex-col z-0">
      <div className="flex-1 w-full max-w-4xl mx-auto px-6 pt-20 pb-24 relative z-10 flex flex-col">

        <AnimatePresence mode="wait">
          {viewState === 'setup' && (
            /* SETUP STATE */
            <motion.div
              key="setup"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20, filter: 'blur(4px)' }}
              transition={{ duration: 0.3 }}
              className="flex-1 flex flex-col justify-center items-center text-center mt-12"
            >
              <h1 className="ep-h2 mb-4 text-white drop-shadow-[0_4px_20px_rgba(147,197,253,0.3)]">
                Run the ECLIPSE pipeline on any TESS target.
              </h1>
              <p className="ep-body max-w-2xl mx-auto mb-10 text-[#BAE6FD] text-lg">
                Enter a TESS Input Catalog ID to detect transit signals, classify them into four astrophysical categories, and estimate orbital parameters with calibrated uncertainty.
              </p>

              {/* Mode Toggle: Fast Benchmark vs Live NASA MAST */}
              <div className="flex items-center justify-center gap-3 mb-6 bg-[#07101E]/60 border border-[#3B6A9A]/30 p-1.5 rounded-full backdrop-blur-xl">
                <button
                  type="button"
                  onClick={() => setLiveMast(false)}
                  className={`px-5 py-1.5 rounded-full text-xs font-semibold ep-dsp tracking-wider uppercase transition-all ${!liveMast ? 'bg-[#1B6EE8] text-white shadow-[0_0_15px_rgba(27,110,232,0.5)]' : 'text-[#BAE6FD]/70 hover:text-white'}`}
                >
                  Fast High-Cadence Mode
                </button>
                <button
                  type="button"
                  onClick={() => setLiveMast(true)}
                  className={`px-5 py-1.5 rounded-full text-xs font-semibold ep-dsp tracking-wider uppercase transition-all ${liveMast ? 'bg-[#1FAD73] text-white shadow-[0_0_15px_rgba(31,173,115,0.5)]' : 'text-[#BAE6FD]/70 hover:text-white'}`}
                >
                  Live NASA MAST Ingestion
                </button>
              </div>

              <div className="w-full max-w-3xl flex flex-col md:flex-row gap-4 mb-6">
                {/* TIC Input */}
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={ticInput}
                    onChange={(e) => setTicInput(e.target.value.replace(/\D/g, ''))}
                    placeholder="Try TIC 261136679 (L 98-59)"
                    className="w-full bg-[#050B14]/60 border border-[#3B6A9A]/30 rounded-xl px-8 py-5 text-white text-lg ep-mono placeholder-[#3B6A9A]/60 focus:outline-none focus:border-[#93C5FD] transition-all text-center md:text-left shadow-[0_10px_40px_rgba(0,5,15,0.5)] backdrop-blur-xl"
                    onKeyDown={e => e.key === 'Enter' && beginAnalysis()}
                  />
                </div>

                {/* Sector Selector */}
                <div className="relative md:w-48">
                  <button
                    onClick={() => setIsSectorDropdownOpen(!isSectorDropdownOpen)}
                    className="w-full h-full min-h-[68px] bg-[#050B14]/60 border border-[#3B6A9A]/30 rounded-xl px-6 flex items-center justify-between text-white ep-mono hover:border-[#93C5FD]/50 transition-all shadow-[0_10px_40px_rgba(0,5,15,0.5)] backdrop-blur-xl"
                  >
                    <div className="flex flex-col text-left">
                      <span className="text-[10px] text-[#BAE6FD] uppercase tracking-[0.15em] font-bold">Sector</span>
                      <span className="text-xl font-bold">{String(sector).padStart(2, '0')}</span>
                    </div>
                    <ChevronDown size={18} className="text-[#93C5FD]/70" />
                  </button>

                  {isSectorDropdownOpen && (
                    <div className="absolute top-[calc(100%+8px)] left-0 w-full bg-[#050B14]/95 border border-[#3B6A9A]/40 rounded-xl max-h-64 overflow-y-auto z-50 backdrop-blur-2xl shadow-[0_20px_50px_rgba(0,5,15,0.8)]">
                      <div className="p-3 border-b border-[#3B6A9A]/40 text-[10px] text-[#93C5FD] font-bold ep-mono text-center tracking-wider bg-[#1B6EE8]/10">
                        Sector 1 recommended for demo
                      </div>
                      {Array.from({ length: 26 }, (_, i) => i + 1).map(s => (
                        <button
                          key={s}
                          onClick={() => { setSector(s); setIsSectorDropdownOpen(false) }}
                          className="w-full text-left px-5 py-3 text-white ep-mono hover:bg-[#1B6EE8]/20 hover:text-[#93C5FD] hover:pl-6 transition-all text-sm border-b border-[#3B6A9A]/20 last:border-0 font-medium"
                        >
                          Sector {String(s).padStart(2, '0')}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Target Pills */}
              <div className="flex flex-wrap justify-center items-center gap-2 mb-10 max-w-3xl">
                {FAMOUS_TARGETS.map((t, i) => (
                  <div key={t.name} className="flex items-center gap-2">
                    <button
                      onClick={() => { setTicInput(t.tic); setSector(t.sector) }}
                      className="text-xs text-[#E0E7FF] font-medium ep-mono border border-[#3B6A9A]/40 bg-[#0B172A]/40 hover:bg-[#1B6EE8]/20 hover:border-[#93C5FD] px-3.5 py-1.5 rounded-full transition-all"
                    >
                      {t.name}
                    </button>
                    {i < FAMOUS_TARGETS.length - 1 && <span className="text-[#3B6A9A]/40">·</span>}
                  </div>
                ))}
              </div>

              {/* Action */}
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <button
                  onClick={beginAnalysis}
                  disabled={!ticInput}
                  className="rounded-xl bg-[#1B6EE8] hover:bg-[#1B6EE8]/80 text-white px-12 py-4 flex items-center justify-center gap-3 ep-dsp tracking-[0.15em] uppercase font-semibold text-sm transition-all shadow-[0_0_25px_rgba(27,110,232,0.4)] disabled:opacity-50 disabled:pointer-events-none"
                >
                  <Sparkles size={16} /> Begin Analysis
                </button>
              </motion.div>
            </motion.div>
          )}

          {viewState === 'analyzing' && (
            /* EXECUTION / ANIMATION STATE */
            <motion.div 
              key="analyzing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="w-full max-w-3xl mx-auto flex flex-col drop-shadow-2xl mt-8"
            >
              {/* Header */}
              <div className="border-b border-[#3B6A9A]/30 pb-4 mb-12 flex justify-between items-center">
                <div className="ep-mono text-[#BAE6FD] text-[11px] uppercase tracking-widest bg-[#0B172A]/60 px-4 py-2 rounded-xl border border-[#3B6A9A]/30 shadow-[0_0_10px_rgba(0,5,15,0.5)]">
                  Target: TIC <span className="text-white font-bold">{ticInput}</span> <span className="text-[#3B6A9A] mx-2">/</span> Sector <span className="text-white font-bold">{String(sector).padStart(2, '0')}</span>
                </div>
                <button onClick={resetAnalysis} className="text-[11px] text-white/70 hover:text-white ep-mono uppercase tracking-widest transition-colors flex items-center gap-2 drop-shadow-md">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_5px_red]"></span>
                  Cancel
                </button>
              </div>

              {/* Pipeline Stepper */}
              <div className="relative ep-pipe mb-12 w-full max-w-5xl mx-auto bg-[#07101E]/40 p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] backdrop-blur-2xl">
                {/* Dynamic Scanning Line */}
                <div className="absolute top-[68px] left-[10%] right-[10%] h-1 bg-[#050B14] rounded-full border border-[#3B6A9A]/20">
                  <motion.div 
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-[#3B6A9A] via-[#93C5FD] to-[#ffffff] shadow-[0_0_10px_rgba(147,197,253,0.5)] rounded-full"
                    initial={{ width: '0%' }}
                    animate={{ width: `${(Math.max(0, currentStageIndex) / (PIPELINE_STAGES.length - 1)) * 100}%` }}
                    transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>

                {PIPELINE_STAGES.map((stage, idx) => {
                  const isComplete = currentStageIndex > idx
                  const isActive = currentStageIndex === idx
                  
                  return (
                    <div 
                      key={stage.id} 
                      className={`ep-pipe__stage transition-all duration-500 ${isComplete || isActive ? 'opacity-100' : 'opacity-40'} ${isActive ? 'scale-110' : 'scale-100'}`}
                    >
                      <div className={`ep-pipe__num ${isComplete ? 'border-[#1FAD73] bg-[#07101E] text-[#4ade80] shadow-[0_0_15px_rgba(31,173,115,0.2)]' : isActive ? 'border-[#93C5FD] bg-[#07101E] text-[#93C5FD] shadow-[0_0_20px_rgba(147,197,253,0.4)]' : 'border-[#3B6A9A]/40 bg-[#050B14] text-[#3B6A9A]'}`}>
                        {isComplete ? <Check size={20} strokeWidth={3} /> : (idx + 1)}
                      </div>
                      <div className={`ep-pipe__title font-bold tracking-widest ${isActive ? 'text-white drop-shadow-[0_2px_10px_rgba(255,255,255,0.3)]' : isComplete ? 'text-[#BAE6FD]' : 'text-[#3B6A9A]'}`}>
                        {stage.label}
                      </div>
                      <div className={`ep-pipe__sub font-mono ${isActive ? 'text-[#93C5FD]' : 'text-[#3B6A9A]/70'}`}>
                        {isComplete ? 'Completed' : isActive ? 'Processing...' : 'Waiting'}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Centered Narrator */}
              <div className="h-24 flex justify-center items-start mb-8">
                <AnimatePresence mode="wait">
                  {currentStageIndex >= 0 && currentStageIndex < PIPELINE_STAGES.length && (
                    <motion.div
                      key={currentStageIndex}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -15 }}
                      transition={{ duration: 0.4 }}
                      className="max-w-2xl text-center bg-[#0B172A]/40 px-8 py-4 rounded-[20px] border border-[#3B6A9A]/30 shadow-[0_10px_30px_rgba(0,5,15,0.5)] backdrop-blur-xl"
                    >
                      <p className="text-base font-light text-[#BAE6FD] leading-relaxed tracking-wide">
                        {currentStageIndex === 0 && <>Retrieved photometric measurements from TESS Sector {String(sector).padStart(2, '0')}.</>}
                        {currentStageIndex === 1 && <>Removed outlier measurements and corrected for systematics. The star's light curve has been isolated.</>}
                        {currentStageIndex === 2 && <>Searching for periodic transit dips using Transit Least Squares (TLS) across trial frequencies.</>}
                        {currentStageIndex === 3 && <>Running ECLIPSE-PRIME dual-stream Cross-Attention model on global and phase-folded views.</>}
                        {currentStageIndex === 4 && <>Evaluating stellar parameters and computing habitability metrics via Earth Similarity Index.</>}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {viewState === 'result' && (
            /* FINAL RESULT DASHBOARD STATE */
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="w-full max-w-4xl mx-auto flex flex-col gap-8 mt-6"
            >
              {/* Header with quick restart */}
              <div className="flex justify-between items-center border-b border-[#3B6A9A]/30 pb-4">
                <div className="ep-mono text-[#BAE6FD] text-sm font-semibold tracking-wider">
                  Analysis Complete: <span className="text-white font-bold">TIC {ticInput}</span> (Sector {sector})
                </div>
                <button
                  onClick={resetAnalysis}
                  className="inline-flex items-center gap-2 text-xs text-[#BAE6FD] hover:text-white bg-[#0B172A]/60 hover:bg-[#1B6EE8]/20 border border-[#3B6A9A]/40 px-4 py-2 rounded-xl transition-all font-semibold uppercase tracking-wider"
                >
                  <RefreshCw size={14} /> Analyze Another Target
                </button>
              </div>

              {apiError ? (
                <div className="bg-[#F44336]/10 border border-[#F44336]/40 p-8 rounded-3xl bg-[#050B14]/80 backdrop-blur-xl text-center flex flex-col items-center gap-4">
                  <AlertCircle size={40} className="text-[#F44336]" />
                  <div className="text-[#F44336] font-mono text-sm uppercase tracking-widest font-bold">Pipeline Error</div>
                  <div className="text-[#E0E7FF] font-sans max-w-lg">{apiError}</div>
                  <button
                    onClick={resetAnalysis}
                    className="mt-4 px-6 py-2 rounded-xl bg-[#1B6EE8] text-white text-xs uppercase tracking-widest font-semibold"
                  >
                    Try Another Target
                  </button>
                </div>
              ) : apiResult ? (
                <div className="flex flex-col gap-8">
                  {/* Main Classification Card */}
                  <div className="flex flex-col gap-10 bg-[#07101E]/60 backdrop-blur-2xl p-10 md:p-12 rounded-[36px] border border-[#3B6A9A]/40 shadow-[0_40px_100px_rgba(0,5,15,0.8)] relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-b from-[#1B6EE8]/10 to-transparent pointer-events-none"></div>
                    <div className={`absolute -top-40 -right-40 w-96 h-96 rounded-full blur-[120px] opacity-25 pointer-events-none ${apiResult.predicted_class === 'TRANSIT' ? 'bg-[#1FAD73]' : 'bg-[#1B6EE8]'}`}></div>

                    {/* Top: Classification & Conformal */}
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 border-b border-[#3B6A9A]/30 pb-8 relative z-10">
                      <div className="flex flex-col">
                        <span className="text-[#93C5FD]/70 ep-dsp text-[10px] uppercase tracking-[0.3em] mb-2 font-semibold">Primary Classification</span>
                        <div className={`text-6xl md:text-7xl ep-dsp font-black tracking-tight drop-shadow-[0_0_30px_rgba(255,255,255,0.3)] ${apiResult.predicted_class === 'TRANSIT' ? 'text-[#1FAD73]' : 'text-white'}`}>
                          {apiResult.predicted_class}
                        </div>
                      </div>
                      
                      <div className="flex flex-col md:items-end md:text-right">
                        <div className="text-4xl font-light text-white drop-shadow-md mb-1 flex items-baseline gap-2 ep-dsp">
                          {(apiResult.confidence * 100).toFixed(1)}% <span className="text-white/40 text-lg font-medium tracking-widest ep-dsp">CONF</span>
                        </div>
                        <div className="text-white/60 ep-dsp text-xs tracking-[0.1em]">
                          Conformal Set: <span className="text-[#BAE6FD] font-semibold">{apiResult.conformal_class_set?.join(', ') || apiResult.predicted_class}</span>
                        </div>
                      </div>
                    </div>

                    {/* Bottom: Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-2 relative z-10">
                      <div className="flex flex-col bg-[#050B14]/40 p-4 rounded-2xl border border-[#3B6A9A]/20">
                        <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-1 font-semibold">Orbital Period</span>
                        <span className="text-2xl md:text-3xl ep-dsp font-light text-white">{apiResult.period ? `${apiResult.period.toFixed(2)}d` : 'N/A'}</span>
                      </div>
                      <div className="flex flex-col bg-[#050B14]/40 p-4 rounded-2xl border border-[#3B6A9A]/20">
                        <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-1 font-semibold">Transit Depth</span>
                        <span className="text-2xl md:text-3xl ep-dsp font-light text-white">{apiResult.depth_ppm ? `${Math.round(apiResult.depth_ppm)} ppm` : 'N/A'}</span>
                      </div>
                      <div className="flex flex-col bg-[#050B14]/40 p-4 rounded-2xl border border-[#3B6A9A]/20">
                        <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-1 font-semibold">Planet Radius</span>
                        <span className="text-2xl md:text-3xl ep-dsp font-light text-white">{apiResult.rp_rearth ? `${apiResult.rp_rearth.toFixed(2)} R⊕` : 'N/A'}</span>
                      </div>
                      <div className="flex flex-col bg-[#050B14]/40 p-4 rounded-2xl border border-[#3B6A9A]/20">
                        <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-1 font-semibold">Earth Sim Idx</span>
                        <span className="text-2xl md:text-3xl ep-dsp font-light text-white">{apiResult.habitability?.esi_score ? apiResult.habitability.esi_score.toFixed(3) : 'N/A'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex justify-end gap-4">
                    <button 
                      onClick={() => navigate(`/candidate/${apiResult.tic_id}`)}
                      className="inline-flex items-center gap-2 rounded-xl bg-[#1B6EE8] hover:bg-[#1B6EE8]/80 text-white px-8 py-3.5 ep-dsp font-bold tracking-widest uppercase text-xs transition-all shadow-[0_0_20px_rgba(27,110,232,0.3)]"
                    >
                      View Full Candidate Profile <ArrowRight size={16} />
                    </button>
                    <button 
                      onClick={() => navigate('/catalog')}
                      className="rounded-xl bg-[#050B14]/60 border border-[#3B6A9A]/40 text-[#BAE6FD] px-8 py-3.5 ep-dsp font-bold tracking-widest uppercase text-xs hover:border-[#93C5FD] hover:text-white transition-all backdrop-blur-md"
                    >
                      Browse Catalog
                    </button>
                  </div>
                </div>
              ) : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </PageTransition>
  )
}

