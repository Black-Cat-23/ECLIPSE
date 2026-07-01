import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Telescope, Database, Command, X, BookOpen } from 'lucide-react'
import { useGlossary } from './AstroGlossary'

export default function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const navigate = useNavigate()
  const { openSidebar } = useGlossary()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen(true)
      }
      if (e.key === 'Escape') setIsOpen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  if (!isOpen) return null

  const getResults = () => {
    const q = query.toLowerCase()
    const results = []
    
    // Exact TIC match
    if (/^\d{6,10}$/.test(q)) {
      results.push({ id: 'analyze-tic', icon: Telescope, label: `Analyze TIC ${q}`, action: () => navigate(`/analysis?tic=${q}`) })
      results.push({ id: 'view-tic', icon: Database, label: `View Candidate Profile TIC ${q}`, action: () => navigate(`/candidate/${q}`) })
    }
    
    if ('sector'.includes(q) || /^\d{1,2}$/.test(q)) {
      const num = q.match(/\d+/) ? q.match(/\d+/)?.[0] : '01'
      results.push({ id: 'sector', icon: Database, label: `Process Sector ${num}`, action: () => navigate(`/sector`) })
    }

    if ('catalog'.includes(q)) {
      results.push({ id: 'cat-all', icon: Database, label: 'View all TRANSIT candidates', action: () => navigate('/catalog') })
    }

    if ('esi'.includes(q) || 'snr'.includes(q) || 'tce'.includes(q) || 'tls'.includes(q) || 'conformal'.includes(q)) {
      results.push({ id: 'glossary', icon: BookOpen, label: `Search '${q}' in Astro-Glossary`, action: openSidebar })
    }

    if (results.length === 0 && q.length > 0) {
      results.push({ id: 'search-cat', icon: Search, label: `Search catalog for '${q}'`, action: () => navigate('/catalog') })
    }

    return results
  }

  const results = getResults()

  return (
    <AnimatePresence>
      <div className="fixed inset-0 bg-black/60 z-[100] flex items-start justify-center pt-[15vh]">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: -20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -20 }}
          transition={{ duration: 0.15 }}
          className="w-full max-w-2xl bg-[#0A0D18] border border-white/10 rounded-lg shadow-2xl overflow-hidden flex flex-col"
        >
          <div className="flex items-center px-4 py-3 border-b border-white/10">
            <Search size={18} className="text-gray-500 mr-3" />
            <input
              autoFocus
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search targets, pages, or terms..."
              className="flex-1 bg-transparent border-none text-white font-mono placeholder-gray-600 focus:outline-none"
            />
            <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono bg-white/5 px-2 py-1 rounded">
              <Command size={10} /> K
            </div>
            <button onClick={() => setIsOpen(false)} className="ml-4 text-gray-500 hover:text-white">
              <X size={16} />
            </button>
          </div>
          
          {query.length > 0 && (
            <div className="max-h-[60vh] overflow-y-auto p-2">
              {results.map(r => (
                <button
                  key={r.id}
                  onClick={() => { setIsOpen(false); r.action(); }}
                  className="w-full flex items-center gap-3 px-3 py-3 hover:bg-white/5 rounded text-left text-gray-300 hover:text-white transition-colors"
                >
                  <r.icon size={16} className="text-nebula-blue" />
                  <span className="font-mono text-sm">{r.label}</span>
                </button>
              ))}
              {results.length === 0 && (
                <div className="px-3 py-8 text-center text-gray-500 font-mono text-sm">No commands found</div>
              )}
            </div>
          )}
          {query.length === 0 && (
            <div className="px-4 py-6 text-center text-gray-600 font-mono text-xs uppercase tracking-widest">
              Type to search
            </div>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
