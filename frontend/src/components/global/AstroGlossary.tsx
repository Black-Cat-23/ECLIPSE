import React, { createContext, useContext, useState, ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, X, Search, ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

// Glossary Database
export const GLOSSARY: Record<string, { term: string, definition: string, formula?: string }> = {
  ESI: { 
    term: 'Earth Similarity Index (ESI)', 
    definition: 'A measure from 0 to 1 of how physically similar a planet is to Earth, incorporating radius, density, escape velocity, and surface temperature.',
    formula: 'Ref: Schulze-Makuch et al., 2011'
  },
  SNR: {
    term: 'Signal-to-Noise Ratio (SNR)',
    definition: 'The strength of the transit signal divided by the noise level of the light curve. Higher SNR indicates a more robust detection.'
  },
  TCE: {
    term: 'Threshold Crossing Event (TCE)',
    definition: 'A periodic signal found by the pipeline that exceeds a minimum signal-to-noise threshold, requiring further classification.'
  },
  PDCSAP: {
    term: 'PDCSAP Flux',
    definition: 'Pre-search data conditioning — telescope systematics are removed, but natural stellar variability is preserved.'
  },
  TLS: {
    term: 'Transit Least Squares (TLS)',
    definition: 'An algorithm that searches for transit-like dips in light curves using a realistic transit model rather than a simple box shape.'
  },
  SDE: {
    term: 'Signal Detection Efficiency (SDE)',
    definition: 'The significance of the period found by TLS. An SDE > 7 is typically required to consider a signal a genuine TCE.'
  },
  CONFORMAL: {
    term: 'Conformal Prediction',
    definition: 'A statistical framework that outputs a set of possible classifications with a mathematical guarantee (e.g., 90%) that the true class is inside the set.',
    formula: 'Provides calibrated confidence rather than raw softmax probabilities.'
  },
  BATMAN: {
    term: 'Batman Model',
    definition: 'A standard analytical model for transit light curves that calculates exact transit shapes based on orbital parameters and stellar limb darkening.'
  }
}

interface GlossaryContextType {
  openSidebar: () => void;
  closeSidebar: () => void;
  isSidebarOpen: boolean;
}

const GlossaryContext = createContext<GlossaryContextType>({
  openSidebar: () => {}, closeSidebar: () => {}, isSidebarOpen: false
})

export function useGlossary() { return useContext(GlossaryContext) }

export function GlossaryProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')

  const filtered = Object.values(GLOSSARY).filter(g => 
    g.term.toLowerCase().includes(search.toLowerCase()) || 
    g.definition.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => a.term.localeCompare(b.term))

  return (
    <GlossaryContext.Provider value={{ openSidebar: () => setIsOpen(true), closeSidebar: () => setIsOpen(false), isSidebarOpen: isOpen }}>
      {children}
      <AnimatePresence>
        {isOpen && (
          <motion.div key="glossary-wrapper" className="fixed inset-0 z-[100] pointer-events-none">
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/40 pointer-events-auto"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="absolute top-0 right-0 w-full max-w-md h-full bg-[#05070E] border-l border-white/10 p-6 flex flex-col shadow-2xl overflow-hidden pointer-events-auto"
            >
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-2 text-nebula-blue font-mono uppercase tracking-widest text-sm">
                  <BookOpen size={16} /> Astro-Glossary
                </div>
                <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-white transition-colors cursor-pointer z-10 p-2 -mr-2">
                  <X size={20} />
                </button>
              </div>
              
              <div className="relative mb-6">
                <Search size={14} className="absolute left-3 top-3 text-gray-500" />
                <input 
                  type="text" 
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search scientific terms..."
                  className="w-full bg-[#0A0D18] border border-white/10 rounded px-9 py-2 text-sm text-white focus:outline-none focus:border-nebula-blue font-mono placeholder-gray-600"
                />
              </div>

              <div className="flex-1 overflow-y-auto pr-2 space-y-6 scrollbar-thin">
                {filtered.map(g => (
                  <div key={g.term}>
                    <h4 className="text-white font-semibold text-sm mb-1">{g.term}</h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{g.definition}</p>
                    {g.formula && <p className="text-nebula-blue/80 text-xs mt-1 font-mono">{g.formula}</p>}
                  </div>
                ))}
                {filtered.length === 0 && (
                  <div className="text-gray-600 text-sm italic font-mono text-center mt-10">No terms found.</div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </GlossaryContext.Provider>
  )
}

export function GlossaryTerm({ termId, children }: { termId: keyof typeof GLOSSARY, children: ReactNode }) {
  const [isHovered, setIsHovered] = useState(false)
  const [xOffset, setXOffset] = useState('-50%')
  const { openSidebar } = useGlossary()
  const data = GLOSSARY[termId]
  let hoverTimeout: any;

  const onEnter = (e: React.MouseEvent<HTMLSpanElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    // If the element is too close to the right edge, shift the tooltip left
    if (rect.right > window.innerWidth - 150) {
      setXOffset('-90%')
    } else if (rect.left < 150) {
      setXOffset('-10%')
    } else {
      setXOffset('-50%')
    }
    hoverTimeout = setTimeout(() => setIsHovered(true), 300) 
  }
  const onLeave = () => { clearTimeout(hoverTimeout); setIsHovered(false) }

  if (!data) return <span className="border-b border-dotted border-gray-600">{children}</span>

  return (
    <span 
      className="relative inline-block cursor-help border-b border-dotted border-gray-600 hover:border-nebula-blue transition-colors"
      onMouseEnter={onEnter} onMouseLeave={onLeave}
    >
      {children}
      <AnimatePresence>
        {isHovered && (
          <motion.span
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            style={{ transform: `translateX(${xOffset})` }}
            className="absolute bottom-full left-1/2 mb-2 w-64 bg-[#0A0D18] border border-white/10 p-4 shadow-2xl z-50 text-left rounded block"
            onClick={(e) => e.stopPropagation()}
          >
            <span className="block text-white font-semibold text-sm mb-1">{data.term}</span>
            <span className="block text-gray-300 text-sm leading-relaxed mb-3 font-sans">{data.definition}</span>
            {data.formula && <span className="block text-nebula-blue/80 text-[10px] font-mono mb-2">{data.formula}</span>}
            <button 
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setIsHovered(false); openSidebar(); }}
              className="text-nebula-blue text-xs flex items-center gap-1 hover:underline"
            >
              See in full glossary <ArrowRight size={10} />
            </button>
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  )
}
