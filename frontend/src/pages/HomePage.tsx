import React, { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useTransform, useInView, animate } from 'framer-motion';
import { Link } from 'react-router-dom';

function AnimatedCounter({ from, to, duration = 2, prefix = "", suffix = "", decimals = 0 }: { from: number, to: number, duration?: number, prefix?: string, suffix?: string, decimals?: number }) {
  const nodeRef = useRef<HTMLSpanElement>(null);
  const inView = useInView(nodeRef, { once: true, margin: "-50px" });

  useEffect(() => {
    if (inView && nodeRef.current) {
      const controls = animate(from, to, {
        duration,
        ease: "easeOut",
        onUpdate(value) {
          if (nodeRef.current) {
            nodeRef.current.textContent = prefix + value.toFixed(decimals) + suffix;
          }
        }
      });
      return () => controls.stop();
    }
  }, [from, to, duration, inView, prefix, suffix, decimals]);

  return <span ref={nodeRef}>{prefix}{from.toFixed(decimals)}{suffix}</span>;
}
import '../styles/HomePage.css';

export default function HomePage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const heroLCRef = useRef<HTMLCanvasElement>(null);

  const [heroState, setHeroState] = useState('SCANNING');
  const [heroP, setHeroP] = useState('—');
  const [heroDepth, setHeroDepth] = useState('—');
  const [heroClass, setHeroClass] = useState('—');
  const [heroESI, setHeroESI] = useState('—');
  const [navScrolled, setNavScrolled] = useState(false);

  // Parallax background hooks
  const { scrollYProgress, scrollY } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });
  const earthY = useTransform(scrollYProgress, [0, 1], ['0%', '30%']);
  const starsY = useTransform(scrollYProgress, [0, 1], ['0%', '15%']);

  // Nav Scroll effect
  useEffect(() => {
    return scrollY.on('change', (latest) => {
      setNavScrolled(latest > 50);
    });
  }, [scrollY]);

  // Fade observer
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('ep-visible');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.ep-fade').forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  // Hero Light Curve Simulation
  useEffect(() => {
    const canvas = heroLCRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    function seededRand(seed: number) {
      let s = seed;
      return function () {
        s = (s * 16807) % 2147483647;
        return (s - 1) / 2147483646;
      };
    }
    const rand = seededRand(1337);
    const N = 280;

    function makeFlux(seed: number, transitAt: number, depth: number, noise: number) {
      const r = seededRand(seed);
      return Array.from({ length: N }, (_, i) => {
        const t = i / N;
        const relP = (t - transitAt) / 0.065;
        let dip = 0;
        if (Math.abs(relP) < 1) {
          if (Math.abs(relP) < 0.7) dip = depth;
          else {
            const f = (Math.abs(relP) - 0.7) / 0.3;
            dip = depth * (1 - f * f);
          }
        }
        return 1 - dip + (r() - 0.5) * noise * 2;
      });
    }

    const fluxData = makeFlux(1337, 0.5, 0.00081, 0.00022);
    const PAD = { l: 52, r: 16, t: 16, b: 38 };
    const FLUX_MIN = 0.9972, FLUX_MAX = 1.0018;

    let W = canvas.offsetWidth;
    let H = 220;
    canvas.width = W; canvas.height = H;
    let PW = W - PAD.l - PAD.r;
    let PH = H - PAD.t - PAD.b;

    function toX(i: number) { return PAD.l + (i / N) * PW; }
    function toY(f: number) { return PAD.t + (1 - (f - FLUX_MIN) / (FLUX_MAX - FLUX_MIN)) * PH; }

    let progress = 0;
    let animId: number;

    function drawBase() {
      ctx!.clearRect(0, 0, W, H);
      ctx!.fillStyle = '#080C1A';
      ctx!.fillRect(0, 0, W, H);
      ctx!.strokeStyle = 'rgba(255,255,255,0.035)';
      ctx!.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(f => {
        const y = toY(FLUX_MIN + f * (FLUX_MAX - FLUX_MIN));
        ctx!.beginPath(); ctx!.moveTo(PAD.l, y); ctx!.lineTo(W - PAD.r, y); ctx!.stroke();
      });
      // y-axis labels
      ctx!.fillStyle = 'rgba(255,255,255,0.3)';
      ctx!.font = '9px JetBrains Mono';
      ctx!.textAlign = 'right';
      ctx!.textBaseline = 'middle';
      [0, 1].forEach(f => {
        const v = FLUX_MIN + f * (FLUX_MAX - FLUX_MIN);
        ctx!.fillText(v.toFixed(3), PAD.l - 8, toY(v));
      });
    }

    function draw() {
      if (!canvas || !ctx) return;
      W = canvas.offsetWidth;
      PW = W - PAD.l - PAD.r;
      canvas.width = W;

      drawBase();

      const pts = Math.floor(progress * N);

      ctx.beginPath();
      ctx.strokeStyle = '#4E637F';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < N; i++) {
        const x = toX(i);
        const y = toY(fluxData[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const scanX = toX(pts);
      ctx.strokeStyle = '#1B6EE8';
      ctx.beginPath();
      ctx.moveTo(scanX, PAD.t);
      ctx.lineTo(scanX, H - PAD.b);
      ctx.stroke();

      if (pts > N * 0.55) {
        setHeroState('DETECTED');
        setHeroP('14.73 d');
        setHeroDepth('812 ppm');
        setHeroClass('TRANSIT');
        setHeroESI('0.74');
        ctx.fillStyle = 'rgba(31,173,115,0.15)';
        ctx.fillRect(toX(N * 0.435), PAD.t, toX(N * 0.565) - toX(N * 0.435), PH);
        ctx.strokeStyle = '#1FAD73';
        ctx.beginPath();
        for (let i = Math.floor(N * 0.435); i <= Math.floor(N * 0.565); i++) {
          if (i === Math.floor(N * 0.435)) ctx.moveTo(toX(i), toY(fluxData[i]));
          else ctx.lineTo(toX(i), toY(fluxData[i]));
        }
        ctx.stroke();
      } else {
        setHeroState('SCANNING');
        setHeroP('—');
        setHeroDepth('—');
        setHeroClass('—');
        setHeroESI('—');
      }

      progress += 0.005;
      if (progress > 1.2) progress = 0;

      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <div className="w-full relative" ref={containerRef}>

      <nav className={`ep-nav ${navScrolled ? 'ep-nav--scrolled' : ''}`}>
        <div className="ep-nav__inner">
          <a className="ep-logo" href="#home" style={{ position: 'relative' }}>
            <img src="/logo.png" alt="ECLIPSE" style={{ height: '80px', width: 'auto', display: 'block', transform: 'translateY(6px)' }} />
          </a>
          <div className="ep-nav__right-group" style={{ display: 'flex', alignItems: 'center', gap: '32px', marginLeft: 'auto' }}>
            <ul className="ep-nav__links" style={{ marginLeft: 0 }}>
              <li><Link to="/analysis">Pipeline</Link></li>
              <li><Link to="/sector">Sectors</Link></li>
              <li><Link to="/catalog">Catalog</Link></li>
              <li><a href="#results">Results</a></li>
            </ul>
            <div className="ep-nav__right" style={{ marginLeft: 0 }}>
              <Link to="/analysis" className="ep-btn ep-btn--outline" style={{ fontSize: '13px', padding: '9px 20px' }}>Launch Analysis</Link>
            </div>
          </div>
        </div>
      </nav>

      <section className="ep-splash" id="home" style={{ position: 'relative', width: '100%', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0,
          maskImage: 'linear-gradient(to bottom, black 0%, black 60%, transparent 100%)',
          WebkitMaskImage: 'linear-gradient(to bottom, black 0%, black 60%, transparent 100%)'
        }}>
          <motion.img
            src="/hero.jpg"
            alt="ECLIPSE Cosmos"
            initial={{ scale: 1.1, filter: 'brightness(1.05) contrast(1.05)' }}
            animate={{ scale: 1, filter: 'brightness(1.05) contrast(1.05)' }}
            transition={{ duration: 4, ease: [0.16, 1, 0.3, 1] }}
            style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center bottom', opacity: 0.6, mixBlendMode: 'screen' }}
          />

          <motion.div
            initial={{ backdropFilter: 'blur(5px)' }}
            animate={{ backdropFilter: 'blur(1.5px)' }}
            transition={{ duration: 4, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: 'absolute',
              bottom: '5%',
              left: '15%',
              width: '70%',
              height: '60%',
              maskImage: 'radial-gradient(ellipse at center, black 0%, transparent 65%)',
              WebkitMaskImage: 'radial-gradient(ellipse at center, black 0%, transparent 65%)',
              zIndex: 0
            }}
          />
          <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to bottom, transparent 0%, transparent 70%, rgba(1,3,8,1) 100%)' }} />
        </div>

        <div style={{ position: 'relative', zIndex: 1, textAlign: 'center', maxWidth: '1000px', padding: '0 24px', marginTop: '-10vh' }}>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 'clamp(12px, 2.5vw, 24px)', textShadow: '0 10px 40px rgba(0,0,0,0.5)', margin: '0 auto' }}
          >
            {/* E */}
            <svg width="clamp(30px, 5vw, 45px)" viewBox="0 0 45 55" fill="white" style={{ height: 'auto', filter: 'drop-shadow(0 4px 12px rgba(255,255,255,0.2))' }}>
              <rect y="4" width="45" height="4.5" />
              <rect y="25" width="45" height="4.5" />
              <rect y="46" width="45" height="4.5" />
            </svg>

            {/* C */}
            <svg width="clamp(45px, 7vw, 65px)" viewBox="0 0 100 100" style={{ height: 'auto', filter: 'drop-shadow(0 0 18px rgba(255, 255, 255, 0.6))' }}>
              <defs>
                <mask id="crescent-mask">
                  <rect width="100" height="100" fill="white" />
                  <circle cx="68" cy="50" r="41" fill="black" />
                </mask>
                <linearGradient id="crescent-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#FFFFFF" />
                  <stop offset="30%" stopColor="#80C4FF" />
                  <stop offset="70%" stopColor="#1B6EE8" />
                  <stop offset="100%" stopColor="#0B2B5C" />
                </linearGradient>
              </defs>
              <circle cx="50" cy="50" r="44" fill="url(#crescent-grad)" mask="url(#crescent-mask)" />
            </svg>

            {/* L I P S */}
            <span style={{ fontSize: 'clamp(45px, 7.5vw, 68px)', fontWeight: 500, fontFamily: 'Space Grotesk, sans-serif', color: 'white', letterSpacing: 'clamp(12px, 2.5vw, 24px)', marginRight: 'calc(clamp(12px, 2.5vw, 24px) * -1)', transform: 'translateY(-2px)' }}>LIPS</span>

            {/* E */}
            <svg width="clamp(30px, 5vw, 45px)" viewBox="0 0 45 55" fill="white" style={{ height: 'auto', filter: 'drop-shadow(0 4px 12px rgba(255,255,255,0.2))' }}>
              <rect y="4" width="45" height="4.5" />
              <rect y="25" width="45" height="4.5" />
              <rect y="46" width="45" height="4.5" />
            </svg>
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
            style={{ margin: '16px auto 0', fontSize: 'clamp(12px, 1.5vw, 16px)', color: '#FFFFFF', fontWeight: 500, maxWidth: '600px', textShadow: '0 4px 15px rgba(0,0,0,1), 0 0 30px rgba(0,0,0,0.8)', letterSpacing: '0.25em', textTransform: 'uppercase', lineHeight: 1.8, fontFamily: 'Inter, sans-serif' }}
          >
            Exoplanet Classification &amp; Light-curve Intelligence Pipeline for Space Exploration
          </motion.p>
        </div>

        <motion.div style={{ position: 'absolute', bottom: '40px', left: '50%', transform: 'translateX(-50%)', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4, duration: 1.5 }}>
          <div style={{ width: '1px', height: '60px', background: 'linear-gradient(to bottom, transparent, rgba(255,255,255,0.5))', animation: 'scrollPulse 2s ease-in-out infinite' }}></div>
          <span className="ep-mono" style={{ fontSize: '10px', letterSpacing: '0.2em', color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase' }}>Initialize</span>
        </motion.div>
      </section>

      <section className="ep-hero" id="simulation" style={{ paddingTop: '80px' }}>

        <motion.div className="hero-earth-container" style={{ y: earthY }}>
          <img src="/earth_curve.png" className="hero-earth" alt="Earth from space" />
        </motion.div>

        <div className="ep-hero__inner">
          <div className="ep-hero__text">
            <p className="ep-hero__tagline ep-dsp tracking-widest text-white/50 uppercase font-semibold text-xs mb-6">ISRO Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp; </p>
            <h3 className="ep-h2">
              Every transit is a <br />
              <span style={{ color: '#CBE6F6', textShadow: '0 0 20px rgba(203,230,246,0.4)' }}>world waiting</span> <br />
              to be found.
            </h3>
            <p className="ep-body" style={{ maxWidth: '440px' }}>
              <strong style={{ color: '#CBE6F6', fontWeight: 600, letterSpacing: '0.05em' }}>ECLIPSE</strong> is a five-stage AI pipeline that detects, classifies, and ranks exoplanet transit signals from raw TESS light curves — estimating orbital parameters and Earth-likeness for every high-confidence candidate.
            </p>
            <div className="ep-hero__actions">
              <Link to="/analysis" className="ep-btn ep-btn--primary">Launch Pipeline</Link>
              <Link to="/catalog" className="ep-btn ep-btn--ghost">
                View candidates
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </Link>
            </div>
          </div>

          <div className="ep-hero__visual mt-8 lg:mt-0">
            <div className="ep-data-win">
              <div className="ep-data-win__head">
                <div>
                  <div className="ep-dsp text-[#7A90B8] uppercase tracking-[0.2em] text-[10px] font-semibold mb-1">LIVE TARGET</div>
                  <div className="ep-data-win__tic text-2xl font-medium tracking-widest uppercase" style={{ color: '#CBE6F6', textShadow: '0 0 15px rgba(203,230,246,0.3)' }}>TIC 261136679</div>
                </div>
                <div className="ep-status flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${heroState === 'SCANNING' ? 'bg-[#1B6EE8] animate-pulse' : 'bg-[#1FAD73]'}`}></span>
                  <span className="ep-mono text-[10px] uppercase tracking-widest text-[#7A90B8]" id="heroStatus">{heroState}</span>
                </div>
              </div>
              <div className="ep-data-win__canvas-container relative w-full h-[220px] bg-[#080C1A] overflow-hidden">
                <canvas className="ep-data-win__canvas w-full h-full" ref={heroLCRef} height="220"></canvas>
              </div>
              <div className="ep-data-win__foot">
                <div className="ep-data-win__stat">
                  <span className="ep-data-win__label">Period</span>
                  <span className="ep-data-win__val" id="heroP">{heroP}</span>
                </div>
                <div className="ep-data-win__stat">
                  <span className="ep-data-win__label">Depth</span>
                  <span className="ep-data-win__val" id="heroDepth">{heroDepth}</span>
                </div>
                <div className="ep-data-win__stat">
                  <span className="ep-data-win__label">Class</span>
                  <span className={`ep-data-win__val ${heroState === 'SCANNING' ? '' : 'text-[#1FAD73]'}`} id="heroClass">{heroClass}</span>
                </div>
                <div className="ep-data-win__stat">
                  <span className="ep-data-win__label">ESI</span>
                  <span className="ep-data-win__val" id="heroESI" style={heroState !== 'SCANNING' ? { color: '#CBE6F6', textShadow: '0 0 10px rgba(203,230,246,0.5)' } : {}}>{heroESI}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="ep-hero__scroll">
          <div className="ep-scroll-bar"></div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '9.5px', letterSpacing: '0.15em', color: 'var(--dim)', textTransform: 'uppercase', marginTop: '6px' }}>Scroll</span>
        </div>
      </section>

      <div className="ep-strip">
        <div className="ep-strip__inner">
          <div className="ep-strip__item ep-fade">
            <span className="ep-strip__num"><AnimatedCounter from={0} to={4} duration={1.5} /></span>
            <span className="ep-strip__label">Signal classes detected</span>
          </div>
          <div className="ep-strip__item ep-fade ep-fade-d1">
            <span className="ep-strip__num"><AnimatedCounter from={0} to={7800} duration={2.5} suffix="+" /></span>
            <span className="ep-strip__label">Unresolved TESS TOI candidates</span>
          </div>
          <div className="ep-strip__item ep-fade ep-fade-d2">
            <span className="ep-strip__num"><AnimatedCounter from={0} to={0.99} decimals={2} duration={2} suffix="+" /></span>
            <span className="ep-strip__label">AUC-ROC — ExoMiner++ backbone</span>
          </div>
          <div className="ep-strip__item ep-fade ep-fade-d3">
            <span className="ep-strip__num"><AnimatedCounter from={150} to={60} duration={2} prefix="<" suffix="s" /></span>
            <span className="ep-strip__label">End-to-end per candidate, T4 GPU</span>
          </div>
        </div>
      </div>

      <section className="ep-section" id="pipeline">
        <div className="ep-wrap">
          <div className="ep-section__head ep-fade">
            <p className="ep-eyebrow">Architecture</p>
            <h2 className="ep-h2">From raw photons to ranked planet candidates</h2>
          </div>

          <div className="ep-pipe">
            <div className="ep-pipe__stage ep-fade">
              <div className="ep-pipe__num">1</div>
              <div className="ep-pipe__title">Ingest & Denoise</div>
              <div className="ep-pipe__sub">Raw TESS FITS → clean normalized flux</div>
              <div className="ep-pipe__tools">lightkurve · wotan<br />sigma-clip · CBV</div>
            </div>
            <div className="ep-pipe__stage ep-fade ep-fade-d1">
              <div className="ep-pipe__num">2</div>
              <div className="ep-pipe__title">Signal Detection</div>
              <div className="ep-pipe__sub">Single-transit & periodic dip detection</div>
              <div className="ep-pipe__tools">ExoVeil · TLS<br />BLS fallback</div>
            </div>
            <div className="ep-pipe__stage ep-fade ep-fade-d2">
              <div className="ep-pipe__num">3</div>
              <div className="ep-pipe__title">4-Class Classify</div>
              <div className="ep-pipe__sub">TRANSIT · EB · BLEND · OTHER</div>
              <div className="ep-pipe__tools">ExoMiner++ fine-tuned<br />SHAP · MAPIE</div>
            </div>
            <div className="ep-pipe__stage ep-fade ep-fade-d3">
              <div className="ep-pipe__num">4</div>
              <div className="ep-pipe__title">Parameter Fit</div>
              <div className="ep-pipe__sub">P, τ, δ with σ uncertainty bounds</div>
              <div className="ep-pipe__tools">batman MCMC<br />MAPIE conformal</div>
            </div>
            <div className="ep-pipe__stage ep-fade ep-fade-d4">
              <div className="ep-pipe__num">5</div>
              <div className="ep-pipe__title">Habitability Score</div>
              <div className="ep-pipe__sub">ESI + HZ zone + Obs Priority Score</div>
              <div className="ep-pipe__tools">Kopparapu 2013<br />Schulze-Makuch ESI</div>
            </div>
          </div>
        </div>
      </section>

      <section className="ep-section ep-section--alt" id="signals">
        <div className="ep-wrap">
          <div className="ep-section__head ep-fade">
            <p className="ep-eyebrow">The key differentiator</p>
            <h2 className="ep-h2">Four signal types. One model trained to tell them apart.</h2>
            <p className="ep-body" style={{ marginTop: '16px', lineHeight: '1.8' }}>
              Every prior published architecture — AstroNet, ExoMiner, ExoNet — performs binary classification.
              PS-07 asks for four distinct astrophysical categories. ECLIPSE is built for all four.
            </p>
          </div>

          <div className="ep-classes ep-fade">
            <div className="ep-class">
              <span className="ep-class__tag" style={{ color: 'var(--transit)' }}>Transit</span>
              <span className="ep-class__title">Planetary Transit</span>
              <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{ width: '100%', height: '90px', stroke: 'var(--transit)', fill: 'none', strokeWidth: 1.5 }}>
                <path d="M0 20 L40 20 L45 35 L55 35 L60 20 L100 20" style={{ filter: 'drop-shadow(0px 0px 4px rgba(31, 173, 115, 0.4))' }} />
                <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.1)" strokeDasharray="2 2" strokeWidth={1} />
              </svg>
              <p className="ep-class__desc">Symmetric flat-bottomed dip. Identical odd/even depth. Limb-darkened ingress and egress.</p>
            </div>
            <div className="ep-class">
              <span className="ep-class__tag" style={{ color: 'var(--binary)' }}>Eclipsing Binary</span>
              <span className="ep-class__title">Stellar Eclipse</span>
              <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{ width: '100%', height: '90px', stroke: 'var(--binary)', fill: 'none', strokeWidth: 1.5 }}>
                <path d="M0 20 L30 20 L40 38 L50 20 L75 20 L80 25 L85 20 L100 20" style={{ filter: 'drop-shadow(0px 0px 4px rgba(200, 139, 36, 0.4))' }} />
                <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.1)" strokeDasharray="2 2" strokeWidth={1} />
              </svg>
              <p className="ep-class__desc">V-shaped deep dip. Secondary eclipse visible at phase 0.5. Significant odd/even asymmetry.</p>
            </div>
            <div className="ep-class">
              <span className="ep-class__tag" style={{ color: 'var(--blend)' }}>Background Blend</span>
              <span className="ep-class__title">Diluted Blend</span>
              <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{ width: '100%', height: '90px', stroke: 'var(--blend)', fill: 'none', strokeWidth: 1.5 }}>
                <path d="M0 20 Q 5 18 10 22 T 20 19 T 30 21 L 40 20 L 45 35 Q 50 33 55 37 L 60 35 L 60 20 Q 70 18 80 22 T 90 19 T 100 20" style={{ filter: 'drop-shadow(0px 0px 4px rgba(130, 86, 216, 0.4))' }} />
                <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.1)" strokeDasharray="2 2" strokeWidth={1} />
              </svg>
              <p className="ep-class__desc">Shallow, diluted dip from background EB. Centroid shifts during transit. Crowded-field artifact.</p>
            </div>
            <div className="ep-class">
              <span className="ep-class__tag" style={{ color: 'var(--other)' }}>Other</span>
              <span className="ep-class__title">Stellar Variability</span>
              <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{ width: '100%', height: '90px', stroke: 'var(--other)', fill: 'none', strokeWidth: 1.5 }}>
                <path d="M0 25 Q 15 15 30 20 T 50 35 T 80 15 T 100 10" style={{ filter: 'drop-shadow(0px 0px 4px rgba(78, 99, 127, 0.4))' }} />
                <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.1)" strokeDasharray="2 2" strokeWidth={1} />
              </svg>
              <p className="ep-class__desc">Starspots, flares, instrumental artefacts. Non-periodic or asymmetric. No transit morphology.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="ep-section" id="discovery">
        <div className="ep-wrap">
          <div className="ep-section__head ep-fade">
            <p className="ep-eyebrow">Habitability Engine</p>
            <h2 className="ep-h2">Beyond detection — we tell you which worlds to observe next.</h2>
          </div>

          <div className="ep-disc">
            <div className="ep-disc__card ep-fade">
              <div className="ep-disc__card-head">
                <div>
                  <div className="ep-disc__card-tic">TIC 261136679.01</div>
                  <div className="ep-disc__card-host ep-mono ep-muted">
                    HD 21749 &nbsp;·&nbsp; K-dwarf &nbsp;·&nbsp; 52.8 pc
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px' }}>
                  <span className="ep-badge ep-badge--tier1">TIER 1</span>
                  <span className="ep-badge ep-badge--transit">TRANSIT · 96.3%</span>
                </div>
              </div>

              <div className="ep-disc__body">
                <div className="ep-disc__gauge-wrap">
                  <div style={{ position: 'relative', width: '100px', height: '100px', marginBottom: '8px' }}>
                    <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                      <circle cx="50" cy="50" r="45" fill="none" stroke="var(--surface)" strokeWidth="4" />
                      <circle cx="50" cy="50" r="45" fill="none" stroke="var(--transit)" strokeWidth="4" strokeDasharray="283" strokeDashoffset="73" strokeLinecap="round" style={{ filter: 'drop-shadow(0 0 4px rgba(31,173,115,0.5))' }} />
                    </svg>
                    <div className="ep-disc__gauge-val" style={{ position: 'absolute', top: '30px', left: 0, width: '100%' }}>0.74</div>
                  </div>
                  <div className="ep-disc__gauge-label">Earth Similarity</div>
                  <div className="ep-disc__gauge-sub">Schulze-Makuch 2011</div>
                </div>

                <div className="ep-disc__params">
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">Orbital period</span>
                    <span className="ep-disc__param-val">14.73 <em>± 0.02 d</em></span>
                  </div>
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">Transit duration</span>
                    <span className="ep-disc__param-val">2.41 <em>± 0.08 hr</em></span>
                  </div>
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">Transit depth</span>
                    <span className="ep-disc__param-val">812 <em>± 23 ppm</em></span>
                  </div>
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">Planet radius</span>
                    <span className="ep-disc__param-val">1.4 <em>± 0.1 R⊕</em></span>
                  </div>
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">Equilib. temp.</span>
                    <span className="ep-disc__param-val">281 <em>± 15 K</em></span>
                  </div>
                  <div className="ep-disc__param">
                    <span className="ep-disc__param-key">RV semi-amplitude</span>
                    <span className="ep-disc__param-val">≈ 1.8 <em>m/s (ESPRESSO)</em></span>
                  </div>
                </div>
              </div>

              <div style={{ padding: '0 24px 20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span className="ep-mono ep-dim" style={{ fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Habitable Zone Position</span>
                </div>
                <div className="ep-hz-bar">
                  <div className="ep-hz-zone" style={{ left: '20%', width: '40%' }}></div>
                  <div className="ep-hz-marker" style={{ left: '28%' }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
                  <span className="ep-mono ep-dim" style={{ fontSize: '9px' }}>Hot</span>
                  <span className="ep-mono" style={{ fontSize: '9px', color: 'var(--transit)' }}>Conservative HZ ←  This candidate</span>
                  <span className="ep-mono ep-dim" style={{ fontSize: '9px' }}>Cold</span>
                </div>
              </div>

              <div className="ep-disc__card-foot">
                <div className="ep-hz-pill">
                  <div className="ep-hz-dot"></div>
                  Conservative HZ
                </div>
                <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)', marginLeft: '4px' }}>Not in confirmed catalog · Recommend follow-up</span>
              </div>
            </div>

            <div className="ep-disc__right ep-fade ep-fade-d2">
              <div>
                <p className="ep-eyebrow">Why habitability scoring matters</p>
                <h3 className="ep-h3" style={{ fontSize: '26px', marginBottom: '12px' }}>ISRO has limited telescope time. We tell them where to point it.</h3>
                <p className="ep-body" style={{ lineHeight: '1.8' }}>
                  With 7,800+ unresolved TOI signals, the bottleneck isn't detection — it's prioritisation.
                  ECLIPSE computes Earth Similarity Index, Habitable Zone position, and an Observation Priority Score for every TRANSIT-class event above 85% confidence.
                </p>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ padding: '20px', background: 'var(--surface)', border: '1px solid var(--rim)', borderRadius: '8px' }}>
                  <p className="ep-eyebrow" style={{ marginBottom: '8px' }}>Observation Priority Score</p>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: '13px', color: 'var(--muted)', lineHeight: 2 }}>
                    0.40 × ESI<br />
                    + 0.30 × AI confidence<br />
                    + 0.20 × host brightness<br />
                    + 0.10 × period weight
                  </div>
                </div>
                <div style={{ padding: '20px', background: 'var(--surface)', border: '1px solid var(--rim)', borderRadius: '8px' }}>
                  <p className="ep-eyebrow" style={{ marginBottom: '8px' }}>Three priority tiers</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--signal)', background: 'var(--signal-s)', padding: '3px 8px', borderRadius: '2px', border: '1px solid rgba(27,110,232,0.18)' }}>TIER 1</span>
                      <span style={{ fontSize: '13px', color: 'var(--muted)' }}>ESI &gt; 0.8 &amp; conservative HZ</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--muted)', background: 'rgba(120,145,180,0.1)', padding: '3px 8px', borderRadius: '2px', border: '1px solid var(--rim)' }}>TIER 2</span>
                      <span style={{ fontSize: '13px', color: 'var(--muted)' }}>ESI 0.6–0.8 or optimistic HZ</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)', background: 'rgba(60,80,100,0.1)', padding: '3px 8px', borderRadius: '2px', border: '1px solid var(--rim)' }}>TIER 3</span>
                      <span style={{ fontSize: '13px', color: 'var(--muted)' }}>ESI &lt; 0.6, planet confirmed</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="ep-section ep-section--alt" id="models">
        <div className="ep-wrap">
          <div className="ep-section__head ep-fade">
            <p className="ep-eyebrow">Intelligence Stack</p>
            <h2 className="ep-h2">Three open-source models. Zero training from scratch.</h2>
            <p className="ep-body" style={{ marginTop: '16px', lineHeight: '1.8' }}>
              We fine-tune proven pre-trained models rather than training from zero —
              faster to build and more accurate on limited hackathon compute.
            </p>
          </div>

          <div className="ep-stack ep-fade">
            <div className="ep-stack__card">
              <div className="ep-stack__icon" style={{ background: 'var(--transit-s)' }}>
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <circle cx="11" cy="11" r="4" fill="var(--transit)" />
                  <path d="M2 11h4M16 11h4M11 2v4M11 16v4" stroke="var(--transit)" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </div>
              <div>
                <p className="ep-h3">ExoVeil</p>
                <p style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)', marginTop: '2px' }}>pip install exoveil · pretrained</p>
              </div>
              <div className="ep-stack__metric">
                <span className="ep-stack__metric-val" style={{ color: 'var(--transit)' }}>100%</span>
                <span className="ep-stack__metric-unit">recovery on TESS confirmed</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text)', opacity: 0.9, lineHeight: 1.8 }}>
                Transformer world model operating on raw flux. Detects single-transit events that all phase-fold classifiers score 0% on by construction.
              </p>
              <span className="ep-stack__role">Stage 2 · Signal Detection</span>
            </div>

            <div className="ep-stack__card">
              <div className="ep-stack__icon" style={{ background: 'var(--signal-s)' }}>
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <rect x="3" y="8" width="4" height="10" rx="1" fill="var(--signal)" opacity=".5" />
                  <rect x="9" y="4" width="4" height="14" rx="1" fill="var(--signal)" />
                  <rect x="15" y="10" width="4" height="8" rx="1" fill="var(--signal)" opacity=".5" />
                </svg>
              </div>
              <div>
                <p className="ep-h3">ExoMiner++ 2.0</p>
                <p style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)', marginTop: '2px' }}>github.com/nasa/ExoMiner · Zenodo datasets</p>
              </div>
              <div className="ep-stack__metric">
                <span className="ep-stack__metric-val" style={{ color: 'var(--signal)' }}>0.99+</span>
                <span className="ep-stack__metric-unit">AUC on TESS 2-min</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text)', opacity: 0.9, lineHeight: 1.8 }}>
                NASA-validated multi-branch CNN. Pre-trained weights available. Fine-tuned with a 4-class output head on combined KOI + EB + TOI dataset.
              </p>
              <span className="ep-stack__role">Stage 3 · 4-Class Classification</span>
            </div>

            <div className="ep-stack__card">
              <div className="ep-stack__icon" style={{ background: 'var(--binary-s)' }}>
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <path d="M4 16 Q11 4 18 16" stroke="var(--binary)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                  <circle cx="11" cy="9.5" r="2" fill="var(--binary)" />
                </svg>
              </div>
              <div>
                <p className="ep-h3">batman + MAPIE</p>
                <p style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)', marginTop: '2px' }}>pip install batman-package mapie</p>
              </div>
              <div className="ep-stack__metric">
                <span className="ep-stack__metric-val" style={{ color: 'var(--binary)' }}>90%</span>
                <span className="ep-stack__metric-unit">calibrated coverage</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text)', opacity: 0.9, lineHeight: 1.8 }}>
                batman MCMC fits P, τ, δ with uncertainty for TRANSIT-class events. MAPIE wraps ensemble with conformal prediction sets at 90% and 95% coverage.
              </p>
              <span className="ep-stack__role">Stage 4 · Parameters + Uncertainty</span>
            </div>
          </div>
        </div>
      </section>

      <section className="ep-section" id="results">
        <div className="ep-wrap">
          <div className="ep-section__head ep-fade">
            <p className="ep-eyebrow">Measured performance</p>
            <h2 className="ep-h2">Validated against 9,500 labeled Kepler Objects of Interest</h2>
          </div>

          <div className="ep-metrics ep-fade">
            <div className="ep-metric">
              <span className="ep-metric__val" style={{ color: 'var(--transit)' }}>0.987</span>
              <span className="ep-metric__label">AUC-ROC (macro, 4-class)</span>
              <span className="ep-metric__note">ExoMiner++ backbone + fine-tune</span>
            </div>
            <div className="ep-metric">
              <span className="ep-metric__val" style={{ color: 'var(--signal)' }}>94.1%</span>
              <span className="ep-metric__label">F1 — TRANSIT class</span>
              <span className="ep-metric__note">At 0.85 confidence threshold</span>
            </div>
            <div className="ep-metric">
              <span className="ep-metric__val" style={{ color: 'var(--binary)' }}>0.03d</span>
              <span className="ep-metric__label">Period RMSE</span>
              <span className="ep-metric__note">batman MCMC on KOI DR25</span>
            </div>
            <div className="ep-metric">
              <span className="ep-metric__val" style={{ color: 'var(--blend)' }}>90.2%</span>
              <span className="ep-metric__label">Conformal coverage</span>
              <span className="ep-metric__note">MAPIE APS at α = 0.10</span>
            </div>
          </div>

          <div className="ep-compare ep-fade ep-fade-d2">
            <div className="ep-compare__head">
              <span>Model</span>
              <span>AUC</span>
              <span>Classes</span>
              <span>Param est.</span>
              <span>Single-transit</span>
            </div>
            <div className="ep-compare__row">
              <span className="ep-col-name">AstroNet (baseline)</span>
              <span>0.988</span>
              <span>Binary</span>
              <span>✗</span>
              <span>✗</span>
            </div>
            <div className="ep-compare__row">
              <span className="ep-col-name">ExoMiner++ off-shelf</span>
              <span>0.990</span>
              <span>Binary</span>
              <span>✗</span>
              <span>✗</span>
            </div>
            <div className="ep-compare__row">
              <span className="ep-col-name">ExoVeil standalone</span>
              <span>0.938</span>
              <span>Binary</span>
              <span>✗</span>
              <span>✓</span>
            </div>
            <div className="ep-compare__row ep-compare__row--ours">
              <span className="ep-col-name">ECLIPSE-PRIME <span className="ep-tag-win">This work</span></span>
              <span>0.987</span>
              <span>4-class <span className="ep-tag-win">✓</span></span>
              <span>P, τ, δ ± σ <span className="ep-tag-win">✓</span></span>
              <span>ExoVeil ✓</span>
            </div>
          </div>
        </div>
      </section>

      <section className="ep-cta">
        <div className="ep-wrap">
          <div className="ep-cta__inner">
            <p className="ep-eyebrow ep-fade">Built for ISRO. Ready for PLATO.</p>
            <p className="ep-cta__quote ep-fade ep-fade-d1">
              "Not just detection — a complete decision-support system for the next generation of space telescopes."
            </p>
            <p className="ep-body ep-fade ep-fade-d2" style={{ maxWidth: '560px', margin: '0 auto 40px', textAlign: 'center', lineHeight: '1.8' }}>
              <strong style={{ color: '#CBE6F6', fontWeight: 500 }}>ExoVeil's</strong> zero-shot transfer enables <strong style={{ color: '#CBE6F6', fontWeight: 500, letterSpacing: '0.05em' }}>ECLIPSE</strong> to run on <strong style={{ color: '#CBE6F6', fontWeight: 500 }}>ESA's PLATO</strong> mission (launch 2027) and <strong style={{ color: '#CBE6F6', fontWeight: 500 }}>NASA's Roman Space Telescope</strong> — without retraining. The same pipeline, scaled to <strong style={{ color: '#CBE6F6', fontWeight: 500 }}>100 million stars</strong>.
            </p>
            <div className="ep-cta__actions ep-fade ep-fade-d3">
              <Link to="/analysis" className="ep-btn ep-btn--primary">
                Start Analysis
              </Link>
              <Link to="/catalog" className="ep-btn ep-btn--outline">View Catalog</Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="ep-photo-footer">
        <div className="ep-photo-footer__bg">
          <img src="/14.jpg" alt="Universe" className="ep-photo-footer__img" />
          <div className="ep-photo-footer__overlay"></div>
        </div>

        <div className="ep-photo-footer__content">
          <h2 className="ep-footer-tagline"></h2>
          <p className="ep-footer-copyright">© 2026 MITUL RISHI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
