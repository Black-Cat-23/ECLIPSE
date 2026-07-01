import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navigation() {
  const [navScrolled, setNavScrolled] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const handleScroll = () => {
      setNavScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    // Initial check
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <motion.nav 
      initial={{ y: -60 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={`ep-nav ${navScrolled ? 'ep-nav--scrolled' : ''}`}
    >
      <div className="ep-nav__inner">
        <Link className="ep-logo" to="/" style={{ position: 'relative' }}>
          <img src="/logo.png" alt="ECLIPSE" style={{ height: '80px', width: 'auto', display: 'block', transform: 'translateY(6px)' }} />
        </Link>
        <div className="ep-nav__right-group" style={{ display: 'flex', alignItems: 'center', gap: '32px', marginLeft: 'auto' }}>
          <ul className="ep-nav__links" style={{ marginLeft: 0 }}>
            <li><Link to="/analysis" className={location.pathname === '/analysis' ? 'ep-nav-active' : ''}>Pipeline</Link></li>
            <li><Link to="/sector" className={location.pathname === '/sector' ? 'ep-nav-active' : ''}>Sectors</Link></li>
            <li><Link to="/catalog" className={location.pathname === '/catalog' ? 'ep-nav-active' : ''}>Catalog</Link></li>
          </ul>
          <div className="ep-nav__right" style={{ marginLeft: 0 }}>
            <Link to="/analysis" className="ep-btn ep-btn--outline" style={{ fontSize: '13px', padding: '9px 20px' }}>
              Launch Analysis
            </Link>
          </div>
        </div>
      </div>
    </motion.nav>
  )
}
