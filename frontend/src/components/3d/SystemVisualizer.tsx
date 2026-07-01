import { useRef, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Environment, Float, Sphere, Trail } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette, Noise } from '@react-three/postprocessing'
import * as THREE from 'three'

// Premium custom shader for the Host Star (Plasma / Corona effect)
const StarMaterial = () => {
  const shaderArgs = useMemo(() => ({
    uniforms: {
      time: { value: 0 },
      colorA: { value: new THREE.Color('#FFD54F') },
      colorB: { value: new THREE.Color('#FF8F00') },
    },
    vertexShader: `
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;
      void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float time;
      uniform vec3 colorA;
      uniform vec3 colorB;
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      // Simple noise function
      float noise(vec3 p) {
        return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
      }

      void main() {
        float intensity = pow(0.65 - dot(vNormal, vec3(0, 0, 1.0)), 2.0);
        float n = noise(vNormal * 10.0 + time * 0.5);
        vec3 color = mix(colorA, colorB, n * intensity);
        // Add extreme brightness for the bloom to pick up
        gl_FragColor = vec4(color * 2.5 + vec3(intensity * 1.5), 1.0);
      }
    `
  }), [])

  useFrame((state) => {
    if (shaderArgs.uniforms) {
      shaderArgs.uniforms.time.value = state.clock.elapsedTime
    }
  })

  return <shaderMaterial attach="material" args={[shaderArgs]} />
}

// The orbiting exoplanet that crosses the star to create a transit
const Exoplanet = ({ orbitRadius = 4, speed = 0.5, size = 0.3 }) => {
  const planetRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    const t = state.clock.elapsedTime * speed
    if (planetRef.current) {
      planetRef.current.position.x = Math.cos(t) * orbitRadius
      planetRef.current.position.z = Math.sin(t) * orbitRadius
      planetRef.current.rotation.y += 0.01
    }
  })

  return (
    <group>
      {/* Invisible orbit ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[orbitRadius - 0.02, orbitRadius + 0.02, 64]} />
        <meshBasicMaterial color="#4FC3F7" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>
      
      {/* The Planet itself */}
      <mesh ref={planetRef} castShadow receiveShadow>
        <sphereGeometry args={[size, 64, 64]} />
        <meshStandardMaterial 
          color="#002233" 
          roughness={0.8} 
          metalness={0.2} 
        />
        {/* Atmosphere glow on the dark planet */}
        <mesh>
           <sphereGeometry args={[size * 1.05, 32, 32]} />
           <meshBasicMaterial color="#80DEEA" transparent opacity={0.15} blending={THREE.AdditiveBlending} />
        </mesh>
      </mesh>
    </group>
  )
}

const SystemScene = () => {
  return (
    <>
      <color attach="background" args={['#030914']} />
      
      {/* Lighting */}
      <ambientLight intensity={0.1} color="#051024" />
      <pointLight position={[0, 0, 0]} intensity={50} color="#FFD54F" castShadow distance={20} decay={2} />
      
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      
      {/* Host Star */}
      <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.2}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[1.5, 64, 64]} />
          <StarMaterial />
        </mesh>
      </Float>

      {/* Orbiting Exoplanets */}
      <Exoplanet orbitRadius={4} speed={0.8} size={0.2} />
      <Exoplanet orbitRadius={7} speed={0.4} size={0.35} />
      
      {/* Controls */}
      <OrbitControls 
        enablePan={false} 
        enableZoom={false} 
        maxPolarAngle={Math.PI / 2 + 0.2} 
        minPolarAngle={Math.PI / 2 - 0.2}
        autoRotate
        autoRotateSpeed={0.5}
      />
      
      {/* Post Processing: This creates the ultra-realistic cinematic look */}
      <EffectComposer enableNormalPass={false}>
        <Bloom luminanceThreshold={1} mipmapBlur intensity={1.5} radius={0.8} />
        <Noise opacity={0.03} />
        <Vignette eskil={false} offset={0.1} darkness={1.1} />
      </EffectComposer>
    </>
  )
}

export default function SystemVisualizer() {
  return (
    <div className="absolute inset-0 w-full h-full z-0 pointer-events-auto">
      <Canvas shadows camera={{ position: [0, 1.5, 12], fov: 45 }}>
        <SystemScene />
      </Canvas>
    </div>
  )
}
