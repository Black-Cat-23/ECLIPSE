import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Compass, X } from 'lucide-react'

interface Annotation {
  id: string
  title: string
  description: string
  x: number // percentage
  y: number // percentage
}

export default function MissionBriefing({ annotations }: { annotations: Annotation[] }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <button 
        onClick={() => setIsOpen(true)}
        className="fixed top-24 right-6 z-40 bg-[#05070E] border border-white/10 p-2 rounded-full text-gray-500 hover:text-nebula-blue hover:border-nebula-blue transition-colors shadow-lg"
        title="Mission Briefing"
      >
        <Compass size={18} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <div 
            className="fixed inset-0 z-50 overflow-hidden"
            onClick={() => setIsOpen(false)}
          >
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-[#05070E]/80 backdrop-blur-sm"
            />
            
            <div className="absolute inset-0 pointer-events-none">
              {annotations.map((ann, i) => (
                <motion.div
                  key={ann.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ delay: i * 0.1 }}
                  className="absolute max-w-[280px]"
                  style={{ left: `${ann.x}%`, top: `${ann.y}%`, transform: 'translate(-50%, -50%)' }}
                >
                  {/* The dot */}
                  <div className="w-2 h-2 rounded-full bg-nebula-blue absolute -top-4 left-1/2 -translate-x-1/2 shadow-[0_0_10px_rgba(79,195,247,0.8)]" />
                  
                  {/* The card */}
                  <div className="bg-[#0A0D18] border border-nebula-blue/30 rounded p-4 shadow-2xl relative mt-2">
                    <div className="absolute -top-[1px] left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[6px] border-b-nebula-blue/30" />
                    <h4 className="text-white font-semibold text-sm mb-1">{ann.title}</h4>
                    <p className="text-gray-400 text-xs leading-relaxed font-sans">{ann.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="absolute bottom-10 left-1/2 -translate-x-1/2 bg-white/10 text-white px-4 py-2 rounded-full font-mono text-xs uppercase tracking-widest backdrop-blur border border-white/20"
            >
              Click anywhere to dismiss
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}
