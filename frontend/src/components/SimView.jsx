import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function SimView({ caseData, onReset }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const distalNodeRef = useRef(null)
  const baseDistalPos = useRef(new THREE.Vector3())
  const baseDistalRot = useRef(new THREE.Euler())
  const animFrameRef = useRef(null)

  const [selectedImplant, setSelectedImplant] = useState(caseData.recommendations?.options?.[0]?.id)
  const [loading, setLoading] = useState(false)
  const [assembly, setAssembly] = useState(null)
  const [error, setError] = useState('')
  const [audience, setAudience] = useState('surgeon')
  const [displacement, setDisplacement] = useState(100) // 100% = displaced fracture, 0% = anatomically reduced
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)

  const activeGlbUrl = assembly?.glb_url || caseData.preview_glb_url
  const fractureInfo = caseData.analysis?.fracture || {}
  const fractureDetected = fractureInfo.detected !== false
  const dispMm = fractureInfo.displacement_mm ?? 6.0
  const angleDeg = fractureInfo.angle_deg ?? 18.0
  const angulationDeg = fractureInfo.angulation_deg ?? 3.5

  useEffect(() => {
    if (!containerRef.current || !activeGlbUrl) return
    const container = containerRef.current
    container.innerHTML = ''
    const width = container.clientWidth
    const height = container.clientHeight
    const scene = new THREE.Scene()
    sceneRef.current = scene
    scene.background = new THREE.Color('#020617')

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    const light1 = new THREE.DirectionalLight(0xffffff, 1.4)
    light1.position.set(120, 200, 260)
    scene.add(light1)

    const light2 = new THREE.DirectionalLight(0x67e8f9, 0.7)
    light2.position.set(-150, -80, -100)
    scene.add(light2)

    scene.add(new THREE.AmbientLight(0x475569, 1.6))

    const loader = new GLTFLoader()
    loader.load(
      activeGlbUrl,
      (gltf) => {
        scene.add(gltf.scene)
        distalNodeRef.current = null

        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true
            child.receiveShadow = true
            if (child.name.toLowerCase().includes('distal')) {
              distalNodeRef.current = child
              baseDistalPos.current.copy(child.position)
              baseDistalRot.current.copy(child.rotation)
            }
          }
        })

        const box = new THREE.Box3().setFromObject(gltf.scene)
        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z, 120)

        camera.position.set(center.x, center.y + maxDim * 0.15, center.z + maxDim * 1.7)
        camera.lookAt(center)
        controls.target.copy(center)

        // If newly assembled, default to reduced (0%), else fractured view (100%)
        if (assembly) {
          setDisplacement(0)
        } else {
          setDisplacement(100)
        }
      },
      undefined,
      (err) => console.error('Failed to load GLB:', err)
    )

    let animId
    const animate = () => {
      animId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!container) return
      camera.aspect = container.clientWidth / container.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', handleResize)
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
      renderer.dispose()
    }
  }, [activeGlbUrl])

  // Update distal node displacement based on slider
  useEffect(() => {
    if (!distalNodeRef.current) return
    const ratio = displacement / 100.0

    // Only apply displacement offset if assembly is loaded (base is reduced)
    // If assembly is active, base is at reduced 0, displaced is at 1.0
    if (assembly) {
      const offsetX = dispMm * ratio
      const offsetY = 1.5 * ratio
      const offsetZ = -Math.abs(dispMm * 0.3) * ratio
      distalNodeRef.current.position.set(
        baseDistalPos.current.x + offsetX,
        baseDistalPos.current.y + offsetY,
        baseDistalPos.current.z + offsetZ
      )
      const angRad = THREE.MathUtils.degToRad(angulationDeg * ratio)
      distalNodeRef.current.rotation.set(
        baseDistalRot.current.x,
        baseDistalRot.current.y + angRad,
        baseDistalRot.current.z
      )
    }
  }, [displacement, dispMm, angulationDeg, assembly])

  // Auto-play reduction / displacement animation loop
  useEffect(() => {
    if (!isAutoPlaying) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      return
    }

    let direction = 1
    let current = displacement
    const step = () => {
      current += direction * 0.8
      if (current >= 100) {
        current = 100
        direction = -1
      } else if (current <= 0) {
        current = 0
        direction = 1
      }
      setDisplacement(current)
      animFrameRef.current = requestAnimationFrame(step)
    }
    animFrameRef.current = requestAnimationFrame(step)

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [isAutoPlaying, displacement])

  const assemble = async () => {
    if (!selectedImplant) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`/api/cases/${caseData.case_id}/assemble`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ implant_id: selectedImplant, audience })
      })
      let data = null
      try {
        data = await res.json()
      } catch {}
      if (!res.ok) throw new Error(data?.detail || `Assembly failed (${res.status} ${res.statusText || 'Server error'}).`)
      setAssembly(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const options = caseData.recommendations?.options || []
  const selected = options.find((o) => o.id === selectedImplant)

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 px-6 py-3.5 flex items-center justify-between bg-slate-900/50 backdrop-blur">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] font-medium text-cyan-400">FemurFix Studio</p>
          <p className="text-sm text-slate-300">
            Case <span className="font-mono text-cyan-300">{caseData.case_id}</span>
            {fractureDetected
              ? <> · {fractureInfo.region || 'shaft'} fracture · AO/OTA {fractureInfo.ao_family || '32'}</>
              : <> · no fracture detected</>}
          </p>
        </div>
        <button onClick={onReset} className="px-4 py-1.5 rounded-lg border border-slate-700 text-sm hover:bg-slate-800 transition">
          New Case
        </button>
      </header>

      <main className="grid lg:grid-cols-[400px,1fr] flex-1 overflow-hidden">
        {/* Sidebar Controls */}
        <aside className="border-r border-slate-800 overflow-y-auto p-5 space-y-5 bg-slate-950/80">
          {/* Analysis Card with Anatomical Metrics */}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Fracture Morphology</h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-2.5 text-sm shadow-sm">
              <div className="flex justify-between pb-1 border-b border-slate-800/80">
                <span className="text-slate-400">Region & Classification</span>
                <span className="font-medium text-cyan-300 capitalize">
                  {fractureDetected ? `${fractureInfo.region} (AO/OTA ${fractureInfo.ao_family})` : 'Not determined by model'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Pattern</span>
                <span className="text-slate-200 text-right text-xs max-w-[210px]">
                  {fractureInfo.pattern || (fractureDetected ? 'Fracture detected' : 'No fracture detected')}
                </span>
              </div>
              {fractureDetected && (
                <>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Fracture Plane Inclination</span>
                    <span className="font-mono text-amber-300">{Math.abs(angleDeg).toFixed(1)}° {angleDeg >= 0 ? '(Oblique)' : '(Reverse)'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Fragment Displacement</span>
                    <span className="font-mono text-amber-300">{Math.abs(dispMm).toFixed(1)} mm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Angular Deformity</span>
                    <span className="font-mono text-amber-300">{Math.abs(angulationDeg).toFixed(1)}°</span>
                  </div>
                </>
              )}
              <div className="flex justify-between pt-1 border-t border-slate-800/80 text-xs">
                <span className="text-slate-500">Detector Confidence</span>
                <span className={`font-medium ${fractureDetected ? 'text-emerald-400' : 'text-slate-500'}`}>
                  {fractureDetected ? `${((fractureInfo.confidence ?? 0) * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
            </div>
            {!fractureDetected && (
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-relaxed text-amber-200">
                <strong>No fracture detected.</strong> This is not evidence of an intact femur. Review the radiograph or try a different image.
              </div>
            )}
          </section>

          {/* Implant Recommendations */}
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Recommended Fixation Constructs</h2>
            {fractureDetected ? (
              <div className="space-y-2">
                {options.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setSelectedImplant(opt.id)}
                  className={`w-full text-left rounded-xl border p-3.5 space-y-1.5 transition ${
                    selectedImplant === opt.id
                      ? 'border-cyan-400 bg-cyan-950/40 shadow-sm shadow-cyan-950'
                      : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-sm text-slate-100">{opt.name}</p>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-400/10 text-cyan-300 font-mono">
                      {(opt.score * 100).toFixed(0)}% fit
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 line-clamp-2">
                    {audience === 'surgeon' ? opt.surgeon_text : opt.patient_text}
                  </p>
                </button>
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3 text-xs leading-relaxed text-slate-400">
                Fixation recommendations are withheld because no fracture was detected.
              </div>
            )}
          </section>

          {/* Assembly Generation */}
          {fractureDetected && options.length > 0 && (
          <section className="space-y-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Surgical Simulation</h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3.5">
              <div className="flex gap-2">
                <button
                  onClick={() => setAudience('surgeon')}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition ${
                    audience === 'surgeon' ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  Surgeon Register
                </button>
                <button
                  onClick={() => setAudience('patient')}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition ${
                    audience === 'patient' ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  Patient Register
                </button>
              </div>

              <button
                onClick={assemble}
                disabled={loading || !selectedImplant}
                className="w-full py-2.5 rounded-lg bg-cyan-400 text-slate-950 text-sm font-semibold disabled:opacity-50 hover:bg-cyan-300 transition shadow-md shadow-cyan-950"
              >
                {loading ? 'Synthesizing Fixation...' : 'Assemble Fixation Construct'}
              </button>
              {error && <p className="text-xs text-rose-400">{error}</p>}
            </div>
          </section>
          )}

          {/* Assembly Narrative & Details */}
          {assembly && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Fixation Outcome</h2>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2.5 text-xs text-slate-300 leading-relaxed">
                <p>
                  <strong className="text-slate-100">Construct:</strong> {assembly.device.name} ({assembly.device.insertion} insertion)
                </p>
                <p className="text-slate-400">{assembly.narrative?.text}</p>
                <div className="pt-2 flex items-center justify-between">
                  <a
                    href={assembly.glb_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 transition"
                  >
                    Download 3D GLB Asset
                  </a>
                </div>
              </div>
            </section>
          )}
        </aside>

        {/* 3D Viewport with Interactive Reduction Controls */}
        <section className="relative w-full h-full bg-slate-950">
          <div ref={containerRef} className="w-full h-full" />

          {/* Top disclaimer badge */}
          <div className="absolute top-4 left-4 rounded-lg border border-slate-800/80 bg-slate-950/85 backdrop-blur px-3 py-1.5 text-xs text-slate-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            Estimated 3D Anatomical Reconstruction · Educational simulation only
          </div>

          {/* Fracture Reduction / Displacement Interactive Controls */}
          {assembly && (
            <div className="absolute bottom-6 left-6 right-6 max-w-lg mx-auto rounded-2xl border border-slate-700/80 bg-slate-900/90 backdrop-blur p-4 shadow-2xl space-y-2.5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Surgical Reduction & Fixation Simulation
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    {displacement > 50 ? 'Displaced Fractured State' : 'Anatomically Reduced with Fixation Hardware'}
                  </p>
                </div>
                <button
                  onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition ${
                    isAutoPlaying ? 'bg-rose-500 text-white' : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
                  }`}
                >
                  {isAutoPlaying ? 'Pause Loop' : '▶ Auto Animate'}
                </button>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-[11px] text-emerald-400 font-mono w-14">0% Reduced</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={displacement}
                  onChange={(e) => {
                    setIsAutoPlaying(false)
                    setDisplacement(parseFloat(e.target.value))
                  }}
                  className="flex-1 accent-cyan-400 cursor-pointer"
                />
                <span className="text-[11px] text-amber-400 font-mono w-16 text-right">100% Broken</span>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
