import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Navigation from './components/layout/Navigation'
import StarfieldCanvas from './components/layout/StarfieldCanvas'
import HomePage from './pages/HomePage'
import AnalysisPage from './pages/AnalysisPage'
import CatalogPage from './pages/CatalogPage'
import SectorPage from './pages/SectorPage'
import CandidatePage from './pages/CandidatePage'
import { GlossaryProvider } from './components/global/AstroGlossary'
import CommandPalette from './components/global/CommandPalette'

export default function App() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <GlossaryProvider>
      <div className="ep-page relative">
        {/* Animated starfield background - running at 30% opacity on all pages */}
        <StarfieldCanvas opacity={isHome ? 1 : 0.3} />

        {/* Global Persistent Cinematic Background (Eliminates Route Flashing) */}
        <AnimatePresence>
          {!isHome && (
            <motion.div
              initial={{ opacity: 0, scale: 1.05, filter: 'brightness(1.1)' }}
              animate={{ opacity: 1, scale: 1, filter: 'brightness(1)' }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 2, ease: [0.16, 1, 0.3, 1] }}
              className="ep-page-bg"
            >
              <img src="/16.jpg" alt="" className="ep-page-bg__img" />
              <div className="ep-page-bg__overlay"></div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Navigation */}
        {!isHome && <Navigation />}

        {/* Global Command Palette */}
        <CommandPalette />

        {/* Page routing */}
        <main className={`relative z-10 ${!isHome ? 'pt-16' : ''}`}>
          <AnimatePresence mode="wait">
            <Routes location={location} key={location.pathname}>
              <Route path="/"           element={<HomePage />} />
              <Route path="/analysis"   element={<AnalysisPage />} />
              <Route path="/catalog"    element={<CatalogPage />} />
              <Route path="/sector"     element={<SectorPage />} />
              <Route path="/candidate/:ticId" element={<CandidatePage />} />
            </Routes>
          </AnimatePresence>
        </main>
      </div>
    </GlossaryProvider>
  )
}
