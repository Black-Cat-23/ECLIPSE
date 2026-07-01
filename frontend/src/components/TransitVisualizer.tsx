import React from 'react';
import { motion } from 'framer-motion';
import { PredictResponse } from '../types';

export const TransitVisualizer: React.FC<{ data: PredictResponse }> = ({ data }) => {
  if (!data.stellar || !data.rp_rearth) return null;

  // Star parameters
  const starTemp = data.stellar.teff || 5700;
  let starColor = '#FDB813'; // Sun-like
  if (starTemp > 7500) starColor = '#9DB4FF'; // Hot star
  else if (starTemp < 4000) starColor = '#FF7B25'; // M-dwarf

  // Scale the sizes for visualization
  // Earth is ~0.009 Solar radii. Let's exaggerate the planet so it's visible.
  const starBaseSize = 180; 
  const planetBaseSize = Math.max(8, Math.min(40, data.rp_rearth * 3)); 
  
  // Calculate a visual orbit duration (in seconds, mapped from actual period)
  const orbitDuration = Math.max(3, Math.min(10, (data.period || 5) / 2));

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.15 } }}
      className="bg-[#07101E]/40 backdrop-blur-2xl p-8 rounded-[32px] border border-[#3B6A9A]/30 shadow-[0_20px_50px_rgba(0,5,15,0.5)] mb-8 overflow-hidden relative">
      <h2 className="text-[10px] ep-dsp font-semibold uppercase tracking-[0.3em] text-[#BAE6FD] mb-2 drop-shadow-[0_1px_5px_rgba(186,230,253,0.2)]">
        Live Transit Simulation
      </h2>
      <p className="text-xs text-white/50 mb-8 ep-dsp tracking-wide">
        Scale visualization generated from actual AI predictions (Teff: {starTemp}K, Rp: {data.rp_rearth.toFixed(2)} R⊕)
      </p>

      <div className="relative w-full h-64 flex items-center justify-center">
        {/* The Star */}
        <div 
          className="rounded-full absolute z-10"
          style={{
            width: starBaseSize,
            height: starBaseSize,
            background: `radial-gradient(circle at 30% 30%, ${starColor}, #000)`,
            boxShadow: `0 0 60px ${starColor}40, inset -10px -10px 20px rgba(0,0,0,0.5)`
          }}
        />

        {/* Orbit Path */}
        <div 
          className="absolute border border-white/10 rounded-[100%] z-0"
          style={{
            width: '80%',
            height: '40%',
            transform: 'rotateX(75deg)'
          }}
        />

        {/* The Planet (Animated) */}
        <motion.div
          className="absolute z-20 rounded-full bg-black border border-white/20"
          style={{
            width: planetBaseSize,
            height: planetBaseSize,
            boxShadow: 'inset -2px -2px 5px rgba(255,255,255,0.2)'
          }}
          animate={{
            x: ['-200px', '200px', '-200px'],
            y: [0, 10, 0],
            scale: [1, 1.2, 0.8],
            zIndex: [20, 20, 0]
          }}
          transition={{
            duration: orbitDuration,
            repeat: Infinity,
            ease: "linear",
            times: [0, 0.5, 1]
          }}
        />
      </div>

      <div className="flex justify-between items-center mt-6 text-[10px] text-white/40 ep-mono uppercase">
        <span>Planet Size: {data.rp_rearth.toFixed(2)} R⊕</span>
        <span>Orbit Period: {data.period?.toFixed(2)} Days</span>
        <span>Star Temp: {starTemp} K</span>
      </div>
    </motion.div>
  );
};
