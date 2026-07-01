import React from 'react';
import { motion } from 'framer-motion';
import { PredictResponse } from '../types';

export const TransitVisualizer: React.FC<{ data: PredictResponse }> = ({ data }) => {
  if (!data.stellar || !data.rp_rearth || !data.period) return null;

  // Real Physics Calculations
  const starTemp = data.stellar.teff || 5700;
  const starMass = data.stellar.stellar_mass || 1.0; // Solar masses
  const starRadius = data.stellar.stellar_radius || 1.0; // Solar radii
  const periodDays = data.period;
  
  // Kepler's Third Law to find Semi-Major Axis (a) in AU
  // a^3 = M_star * P^2 (if P in years and M in solar masses)
  const periodYears = periodDays / 365.25;
  const semiMajorAxisAU = Math.cbrt(starMass * Math.pow(periodYears, 2));
  const distanceKm = semiMajorAxisAU * 149597870.7; // AU to km

  // Orbital Velocity: v ≈ 2 * pi * a / P
  const circumferenceKm = 2 * Math.PI * distanceKm;
  const periodSeconds = periodDays * 24 * 3600;
  const velocityKmS = circumferenceKm / periodSeconds;

  // Visual scaling
  let starColor = '#FDB813'; 
  if (starTemp > 7500) starColor = '#9DB4FF'; 
  else if (starTemp < 4000) starColor = '#FF7B25'; 

  const orbitDuration = Math.max(3, Math.min(12, periodDays / 2)); // Map real period to 3-12 sec animation

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.15 } }}
      className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8 overflow-hidden relative">
      <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">
        3D Orbital Mechanics Engine
      </h2>
      <p className="text-xs text-white/50 mb-8 ep-dsp tracking-wide">
        Live animation governed by Kepler's Third Law derived from the AI's predicted transit parameters.
      </p>

      {/* Real Physics Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 border-b border-white/5 pb-6">
        <div>
          <div className="text-[9px] text-[#93C5FD]/60 ep-mono uppercase tracking-widest">Planet Radius</div>
          <div className="text-sm text-white ep-dsp font-medium">{data.rp_rearth.toFixed(2)} R⊕</div>
        </div>
        <div>
          <div className="text-[9px] text-[#93C5FD]/60 ep-mono uppercase tracking-widest">Orbital Velocity</div>
          <div className="text-sm text-white ep-dsp font-medium">{velocityKmS.toFixed(1)} km/s</div>
        </div>
        <div>
          <div className="text-[9px] text-[#93C5FD]/60 ep-mono uppercase tracking-widest">Star Distance (a)</div>
          <div className="text-sm text-white ep-dsp font-medium">{semiMajorAxisAU.toFixed(4)} AU</div>
        </div>
        <div>
          <div className="text-[9px] text-[#93C5FD]/60 ep-mono uppercase tracking-widest">Star Radius</div>
          <div className="text-sm text-white ep-dsp font-medium">{starRadius.toFixed(2)} R☉</div>
        </div>
      </div>

      <div className="relative w-full h-72 flex items-center justify-center">
        {/* The Star (unrotated, stays spherical) */}
        <div 
          className="rounded-full absolute z-10"
          style={{
            width: 140, height: 140,
            background: `radial-gradient(circle at 30% 30%, ${starColor}, #000)`,
            boxShadow: `0 0 80px ${starColor}40, inset -10px -10px 40px rgba(0,0,0,0.8)`
          }}
        />

        {/* 3D Orbit System */}
        <div 
          className="absolute flex items-center justify-center z-20"
          style={{ width: 400, height: 400, transformStyle: 'preserve-3d', transform: 'rotateX(75deg)' }}
        >
          {/* Highlighted Orbit Line */}
          <div className="absolute w-full h-full border-[1.5px] border-[#BAE6FD]/30 rounded-full" />

          {/* Spinning Armature */}
          <motion.div
            className="absolute w-full h-full"
            animate={{ rotateZ: [0, 360] }}
            transition={{ duration: orbitDuration, repeat: Infinity, ease: "linear" }}
          >
            {/* The Planet */}
            <div
              className="absolute left-1/2 bg-[#0a0a0a] border border-[#4ade80] rounded-full shadow-[0_0_15px_rgba(74,222,128,0.6)]"
              style={{
                width: 18,
                height: 18,
                top: -9, // half of height to center on the line
                transform: 'translateX(-50%) rotateX(-75deg)', // Counter-rotate so it stays a perfect sphere
              }}
            />
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};
