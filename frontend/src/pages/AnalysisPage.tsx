import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Check, ChevronDown } from 'lucide-react'
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
  { id: 'detect', label: 'Detect', activeText: 'TLS period search: testing 1,200 trial periods...' },
  { id: 'classify', label: 'Classify', activeText: 'Running 1D CNN inference on global and local views...' },
  { id: 'habitability', label: 'Assess', activeText: 'Computing equilibrium temperature and ESI...' }
]

export default function AnalysisPage() {
  const navigate = useNavigate()

  // Form State
  const [ticInput, setTicInput] = useState('')
  const [sector, setSector] = useState(1)
  const [isSectorDropdownOpen, setIsSectorDropdownOpen] = useState(false)

  // Pipeline State
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentStageIndex, setCurrentStageIndex] = useState(-1)
  const [apiResult, setApiResult] = useState<PredictResponse | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [showResult, setShowResult] = useState(false)

  // Trigger analysis
  const beginAnalysis = async () => {
    if (!ticInput) return
    setIsAnalyzing(true)
    setShowResult(false)
    setApiResult(null)
    setApiError(null)
    setCurrentStageIndex(0)

    // 1. Start the API call in the background
    const ticNum = parseInt(ticInput)
    let fetchedResult: PredictResponse | null = null
    let fetchedError: string | null = null

    axios.post('/api/predict', { tic_id: ticNum, sector })
      .then(res => fetchedResult = res.data)
      .catch(err => fetchedError = err.response?.data?.detail || err.message)

    // 2. Artificially step through the 5 stages for the UX narrative
    // Each stage takes 1500ms for demo pacing, except if API takes longer we wait at stage 3/4.
    for (let i = 0; i < 5; i++) {
      setCurrentStageIndex(i)
      await new Promise(r => setTimeout(r, 1800))

      // If we are at the last stage, make sure the API has returned
      if (i === 4) {
        while (!fetchedResult && !fetchedError) {
          await new Promise(r => setTimeout(r, 500))
        }
      }
    }

    setCurrentStageIndex(5) // All complete
    if (fetchedError) {
      setApiError(fetchedError)
    } else {
      setApiResult(fetchedResult)
    }

    // Short pause before expanding result
    await new Promise(r => setTimeout(r, 600))
    setShowResult(true)
  }


  return (
    <PageTransition className="relative min-h-screen flex flex-col z-0">
      <div className="flex-1 w-full max-w-4xl mx-auto px-6 pt-20 pb-24 relative z-10 flex flex-col">
        {/* The 85% random weights situation */}
        <div className="absolute top-0 left-0 w-full flex justify-center mt-6 pointer-events-none z-20">
          <div className="text-[#FF9800] text-xs font-mono tracking-wide">
            Model running on baseline weights. Train the Colab notebook for calibrated results.
          </div>
        </div>

        <AnimatePresence mode="wait">
          {!isAnalyzing ? (
            /* SETUP STATE */
            <motion.div
              key="setup"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40, filter: 'blur(4px)' }}
              transition={{ duration: 0.4 }}
              className="flex-1 flex flex-col justify-center items-center text-center mt-20"
            >
              <h1 className="ep-h2 mb-4 text-white drop-shadow-[0_4px_20px_rgba(147,197,253,0.3)]">
                Run the ECLIPSE pipeline on any TESS target.
              </h1>
              <p className="ep-body max-w-2xl mx-auto mb-12 text-[#BAE6FD] text-lg">
                Enter a TESS Input Catalog ID to detect transit signals, classify them into four astrophysical categories, and estimate orbital parameters with calibrated uncertainty.
              </p>

              <div className="w-full max-w-3xl flex flex-col md:flex-row gap-4 mb-6">
                {/* TIC Input */}
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={ticInput}
                    onChange={(e) => setTicInput(e.target.value.replace(/\D/g, ''))}
                    placeholder="Try TIC 261136679 — host of a known sub-Neptune"
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
                    <div className="absolute top-[calc(100%+8px)] left-0 w-full bg-[#050B14]/90 border border-[#3B6A9A]/40 rounded-xl max-h-64 overflow-y-auto z-50 backdrop-blur-2xl shadow-[0_20px_50px_rgba(0,5,15,0.8)]">
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
              <div className="flex flex-wrap justify-center items-center gap-3 mb-10">
                {FAMOUS_TARGETS.map((t, i) => (
                  <div key={t.name} className="flex items-center gap-3">
                    <button
                      onClick={() => { setTicInput(t.tic); setSector(t.sector) }}
                      className="text-xs text-[#E0E7FF] font-medium ep-mono border border-[#3B6A9A]/40 bg-[#0B172A]/40 hover:bg-[#3B6A9A]/20 hover:border-[#93C5FD]/60 px-4 py-1.5 rounded-full transition-all"
                    >
                      {t.name}
                    </button>
                    {i < FAMOUS_TARGETS.length - 1 && <span className="text-[#3B6A9A] font-bold">·</span>}
                  </div>
                ))}
              </div>

              {/* Action */}
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <button
                  onClick={beginAnalysis}
                  disabled={!ticInput}
                  className="w-full sm:w-auto rounded-xl bg-[#0B172A]/50 border border-[#3B6A9A]/50 text-[#E0E7FF] px-12 py-4 flex items-center justify-center ep-dsp tracking-[0.15em] uppercase font-semibold text-sm transition-colors shadow-[0_0_15px_rgba(59,106,154,0.1)] hover:border-[#93C5FD]/70 hover:text-white hover:shadow-[0_0_25px_rgba(147,197,253,0.3)] disabled:opacity-50 disabled:hover:bg-[#0B172A]/50 mx-auto"
                >
                  Begin Analysis
                </button>
              </motion.div>
            </motion.div>
          ) : (
          /* EXECUTION FLOW */
          <motion.div 
            key="execution"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="w-full max-w-3xl mx-auto flex flex-col drop-shadow-2xl"
          >
            {/* Compressed Header */}
            <div className="border-b border-[#3B6A9A]/30 pb-4 mb-16 flex justify-between items-center">
              <div className="ep-mono text-[#BAE6FD] text-[11px] uppercase tracking-widest bg-[#0B172A]/60 px-4 py-2 rounded-xl border border-[#3B6A9A]/30 shadow-[0_0_10px_rgba(0,5,15,0.5)]">
                Target: TIC <span className="text-white">{ticInput}</span> <span className="text-[#3B6A9A] mx-2">/</span> Sector <span className="text-white">{String(sector).padStart(2, '0')}</span>
              </div>
              <button onClick={() => setIsAnalyzing(false)} className="text-[11px] text-white/70 hover:text-white ep-mono uppercase tracking-widest transition-colors flex items-center gap-2 drop-shadow-md">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_5px_red]"></span>
                Abort
              </button>
            </div>

            {/* Pipeline View */}
            <div className="relative ep-pipe mb-12 w-full max-w-5xl mx-auto bg-[#07101E]/40 p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] backdrop-blur-2xl">
              {/* Dynamic Scanning Line */}
              <div className="absolute top-[68px] left-[10%] right-[10%] h-1 bg-[#050B14] rounded-full border border-[#3B6A9A]/20">
                <motion.div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-[#3B6A9A] via-[#93C5FD] to-[#ffffff] shadow-[0_0_10px_rgba(147,197,253,0.5)] rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${(Math.max(0, currentStageIndex) / (PIPELINE_STAGES.length - 1)) * 100}%` }}
                  transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>

              {PIPELINE_STAGES.map((stage, idx) => {
                const isComplete = currentStageIndex > idx;
                const isActive = currentStageIndex === idx;
                
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
                );
              })}
            </div>

            {/* Centered Narrator */}
            <div className="h-24 flex justify-center items-start mb-8">
              <AnimatePresence mode="wait">
                {!showResult && currentStageIndex >= 0 && currentStageIndex < PIPELINE_STAGES.length && (
                  <motion.div
                    key={currentStageIndex}
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -15 }}
                    transition={{ duration: 0.5 }}
                    className="max-w-2xl text-center bg-[#0B172A]/40 px-8 py-4 rounded-[20px] border border-[#3B6A9A]/30 shadow-[0_10px_30px_rgba(0,5,15,0.5)] backdrop-blur-xl"
                  >
                    <p className="text-lg font-light text-[#BAE6FD] leading-relaxed tracking-wide">
                      {currentStageIndex === 0 && <>Retrieved photometric measurements of <GlossaryTerm termId="PDCSAP">this star</GlossaryTerm>, collected by TESS Sector {String(sector).padStart(2, '0')}.</>}
                      {currentStageIndex === 1 && <>Removed outlier measurements and corrected for telescope systematics. The star's natural brightness variations have been isolated.</>}
                      {currentStageIndex === 2 && <>Searching for periodic dimming events using <GlossaryTerm termId="TLS">Transit Least Squares</GlossaryTerm>. Scanning frequencies for threshold crossings.</>}
                      {currentStageIndex === 3 && <>Analyzing detected signal shape using the deep neural network. Comparing global transit sequence with local ingress/egress profiles.</>}
                      {currentStageIndex === 4 && <>Evaluating stellar parameters and computing habitability metrics via the <GlossaryTerm termId="ESI">Earth Similarity Index</GlossaryTerm>.</>}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Result Reveal */}
            {showResult && (
              <motion.div 
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                className="overflow-hidden drop-shadow-2xl"
              >
                {apiError ? (
                  <div className="bg-[#F44336]/10 border border-[#F44336]/30 p-6 rounded-2xl bg-[#050B14]/80 backdrop-blur-xl">
                    <div className="text-[#F44336] font-mono text-sm uppercase tracking-widest mb-2 font-bold drop-shadow-md">Pipeline Error</div>
                    <div className="text-[#E0E7FF] font-sans font-medium">{apiError}</div>
                  </div>
                ) : apiResult ? (
                  <div className="flex flex-col gap-6">
                    <div className="flex flex-col gap-10 bg-[#07101E]/40 backdrop-blur-2xl p-10 md:p-12 rounded-[40px] border border-[#3B6A9A]/30 shadow-[0_40px_100px_rgba(0,5,15,0.8)] relative overflow-hidden">
                      {/* Decorative Background Glow */}
                      <div className="absolute inset-0 bg-gradient-to-b from-[#1B6EE8]/5 to-transparent pointer-events-none"></div>
                      <div className={`absolute -top-40 -right-40 w-96 h-96 rounded-full blur-[120px] opacity-20 pointer-events-none ${apiResult.predicted_class === 'TRANSIT' ? 'bg-[#1FAD73]' : 'bg-[#93C5FD]'}`}></div>
                      
                      {/* Top: Classification & Conformal */}
                      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 border-b border-[#3B6A9A]/30 pb-10 relative z-10">
                        <div className="flex flex-col">
                          <span className="text-[#93C5FD]/60 ep-dsp text-[10px] uppercase tracking-[0.3em] mb-3 font-semibold">Primary Classification</span>
                          <div className={`text-6xl md:text-8xl ep-dsp font-black tracking-tighter drop-shadow-[0_0_40px_rgba(255,255,255,0.2)] ${apiResult.predicted_class === 'TRANSIT' ? 'text-[#1FAD73]' : 'text-white'}`}>
                            {apiResult.predicted_class}
                          </div>
                        </div>
                        
                        <div className="flex flex-col md:items-end md:text-right mt-4 md:mt-0">
                          <div className="text-4xl font-light text-white drop-shadow-md mb-2 flex items-baseline gap-2 ep-dsp">
                            {(apiResult.confidence * 100).toFixed(1)}% <span className="text-white/40 text-xl font-medium tracking-widest ep-dsp">CONF</span>
                          </div>
                          <div className="text-white/60 ep-dsp text-sm tracking-[0.1em] drop-shadow-md">
                            <GlossaryTerm termId="CONFORMAL">Conformal Set</GlossaryTerm> <span className="text-white/30 mx-2">|</span> 
                            <span className="text-white font-medium ml-1"> {apiResult.conformal_class_set?.join(', ') || apiResult.predicted_class}</span>
                          </div>
                        </div>
                      </div>

                      {/* Bottom: Floating Stats */}
                      <div className="flex flex-wrap justify-between gap-8 pt-2 relative z-10">
                        <div className="flex flex-col">
                          <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-2 font-semibold drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Orbital Period</span>
                          <span className="text-4xl md:text-5xl ep-dsp font-light text-white drop-shadow-[0_2px_10px_rgba(255,255,255,0.3)]">{apiResult.period ? `${apiResult.period.toFixed(2)}d` : 'N/A'}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-2 font-semibold drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Transit Depth</span>
                          <span className="text-4xl md:text-5xl ep-dsp font-light text-white drop-shadow-[0_2px_10px_rgba(255,255,255,0.3)]">{apiResult.depth_ppm ? `${Math.round(apiResult.depth_ppm)} ppm` : 'N/A'}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-2 font-semibold drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Planet Radius</span>
                          <span className="text-4xl md:text-5xl ep-dsp font-light text-white drop-shadow-[0_2px_10px_rgba(255,255,255,0.3)]">{apiResult.rp_rearth ? `${apiResult.rp_rearth.toFixed(2)} R⊕` : 'N/A'}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[#BAE6FD] ep-dsp text-[10px] uppercase tracking-[0.2em] mb-2 font-semibold drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">Earth Sim Idx</span>
                          <span className="text-4xl md:text-5xl ep-dsp font-light text-white drop-shadow-[0_2px_10px_rgba(255,255,255,0.3)]">{apiResult.habitability?.esi_score ? apiResult.habitability.esi_score.toFixed(3) : 'N/A'}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-end gap-4 mt-2">
                      <button 
                        onClick={() => navigate(`/candidate/${apiResult.tic_id}`)}
                        className="rounded-xl bg-[#1B6EE8]/20 border border-[#1B6EE8]/50 text-[#BAE6FD] px-8 py-3 ep-dsp font-bold tracking-widest uppercase text-[11px] hover:bg-[#1B6EE8]/40 hover:text-white transition-all shadow-[0_0_15px_rgba(27,110,232,0.2)]"
                      >
                        View Candidate Profile
                      </button>
                      <button 
                        onClick={() => navigate(`/catalog`)}
                        className="rounded-xl bg-[#050B14]/60 border border-[#3B6A9A]/30 text-[#BAE6FD] px-8 py-3 ep-dsp font-bold tracking-widest uppercase text-[11px] hover:border-[#93C5FD]/50 hover:text-white transition-all shadow-inner backdrop-blur-md"
                      >
                        Add to Catalog
                      </button>
                    </div>
                  </div>
                ) : null}
              </motion.div>
            )}
          </motion.div>
          )}
        </AnimatePresence>
      </div>
    </PageTransition>
  )
}
