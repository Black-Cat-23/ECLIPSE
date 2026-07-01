import { useEffect, useRef } from 'react'

/**
 * Three.js animated star field as the fixed background canvas.
 * 3000 stars with parallax drift effect.
 */
export default function StarfieldCanvas({ opacity = 0.9 }: { opacity?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    let width = window.innerWidth
    let height = window.innerHeight
    canvas.width = width
    canvas.height = height

    // Generate stars
    const N_STARS = 1500
    const stars = Array.from({ length: N_STARS }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 1.2 + 0.1,
      alpha: Math.random() * 0.9 + 0.1,
      twinkleSpeed: Math.random() * 0.03 + 0.005,
      twinkleOffset: Math.random() * Math.PI * 2,
      color: Math.random() > 0.92 ? '#CE93D8' : Math.random() > 0.85 ? '#80DEEA' : '#E8EAF6'
    }))

    // Shooting stars state
    interface ShootingStar {
      x: number;
      y: number;
      length: number;
      speed: number;
      angle: number;
      opacity: number;
    }
    let shootingStars: ShootingStar[] = [];

    // Nebula "blobs" — large diffuse glows
    const nebulae = [
      { x: 0.15, y: 0.25, r: 250, color: 'rgba(79, 195, 247, 0.03)' },
      { x: 0.75, y: 0.60, r: 300, color: 'rgba(206, 147, 216, 0.025)' },
      { x: 0.50, y: 0.85, r: 200, color: 'rgba(128, 222, 234, 0.02)' },
    ]

    let frame = 0
    let animId: number

    function draw() {
      frame++
      ctx.clearRect(0, 0, width, height)

      // Deep space background
      const bg = ctx.createLinearGradient(0, 0, width, height)
      bg.addColorStop(0, '#020408')
      bg.addColorStop(0.5, '#050A1C')
      bg.addColorStop(1, '#020408')
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, width, height)

      // Nebulae
      for (const neb of nebulae) {
        const grd = ctx.createRadialGradient(
          neb.x * width, neb.y * height, 0,
          neb.x * width, neb.y * height, neb.r
        )
        grd.addColorStop(0, neb.color)
        grd.addColorStop(1, 'transparent')
        ctx.fillStyle = grd
        ctx.fillRect(0, 0, width, height)
      }

      // Stars
      for (const star of stars) {
        // Enhanced twinkle with larger amplitude
        const twinkle = Math.sin(frame * star.twinkleSpeed + star.twinkleOffset) * 0.5 + 0.5
        ctx.beginPath()
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2)
        ctx.fillStyle = star.color
        ctx.globalAlpha = star.alpha * twinkle
        ctx.fill()
      }
      ctx.globalAlpha = 1.0

      // Shooting Stars
      if (Math.random() < 0.02) { // Random spawn chance
        shootingStars.push({
          x: Math.random() * width * 1.5, // Can spawn further right
          y: Math.random() * (height / 2) - 100, // Spawn in upper region
          length: Math.random() * 100 + 40,
          speed: Math.random() * 15 + 15,
          angle: (Math.PI / 4) + (Math.random() * 0.1 - 0.05), // roughly 45 deg down-left
          opacity: Math.random() * 0.5 + 0.5 // Initial brightness
        });
      }

      for (let i = shootingStars.length - 1; i >= 0; i--) {
        const ss = shootingStars[i];
        
        ctx.beginPath();
        // Calculate the tail coordinates (behind the star)
        const tailX = ss.x - Math.cos(ss.angle) * ss.length;
        const tailY = ss.y - Math.sin(ss.angle) * ss.length;
        
        const grad = ctx.createLinearGradient(ss.x, ss.y, tailX, tailY);
        grad.addColorStop(0, `rgba(255, 255, 255, ${ss.opacity})`);
        grad.addColorStop(1, `rgba(255, 255, 255, 0)`);
        
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.moveTo(ss.x, ss.y);
        ctx.lineTo(tailX, tailY);
        ctx.stroke();

        // Move the star
        ss.x += Math.cos(ss.angle) * ss.speed;
        ss.y += Math.sin(ss.angle) * ss.speed;
        ss.opacity -= 0.01; // Fade out as it flies

        // Remove if fully faded
        if (ss.opacity <= 0) {
          shootingStars.splice(i, 1);
        }
      }

      animId = requestAnimationFrame(draw)
    }

    draw()

    const handleResize = () => {
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = width
      canvas.height = height
      for (const s of stars) {
        s.x = Math.random() * width
        s.y = Math.random() * height
      }
    }
    window.addEventListener('resize', handleResize)
    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{ opacity, transition: 'opacity 0.5s ease' }}
    />
  )
}
