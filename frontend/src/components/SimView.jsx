import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { gsap } from 'gsap'

export default function SimView({ caseData, onReset, onReAnalyze }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const distalNodeRef = useRef(null)
  const baseDistalPos = useRef(new THREE.Vector3())
  const baseDistalRot = useRef(new THREE.Euler())
  const mixerRef = useRef(null)
  const actionRef = useRef(null)
  const clipDurRef = useRef(0)
  const scrubObjRef = useRef({ v: 100 })
  const stagingRef = useRef(null)
  const introTweenRef = useRef(null)

  const [selectedImplant, setSelectedImplant] = useState(caseData.recommendations?.options?.[0]?.id)
  const [loading, setLoading] = useState(false)
  const [assembly, setAssembly] = useState(null)
  const [error, setError] = useState('')
  const [audience, setAudience] = useState('surgeon')
  const [displacement, setDisplacement] = useState(100) // 100% = displaced fracture, 0% = anatomically reduced
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)
  const [sourceImgFailed, setSourceImgFailed] = useState(false)
  const [regionBusy, setRegionBusy] = useState(false)

  const activeGlbUrl = assembly?.glb_url || caseData.preview_glb_url
  const fractureInfo = caseData.analysis?.fracture || {}
  const fractureDetected = fractureInfo.detected !== false
  const weakSignal = !fractureDetected && fractureInfo.weak_signal ? fractureInfo.weak_signal : null
  const zoneClass = fractureInfo.model_class && fractureInfo.model_class !== 'fracture'
    ? fractureInfo.model_class.replace(/_/g, ' ')
    : null
  const viewHint = fractureInfo.view_hint
    ? `${fractureInfo.view_hint.class_name.replace(/_/g, ' ')} (${Math.round(fractureInfo.view_hint.score * 100)}%)`
    : null
  const viewContext = viewHint
    ? <> The anatomical-zone model reads this film as <em> {viewHint} </em>
        — view context only, it does not detect fractures.</>
    : null
  const regionSourceLabel = {
    zone_model: 'anatomical-zone model',
    model: 'detector class',
    user_override: 'surgeon override',
    geometric: 'silhouette geometry',
  }[fractureInfo.region_source] || null
  const dispMm = fractureInfo.displacement_mm ?? 6.0
  const angleDeg = fractureInfo.angle_deg ?? 18.0
  const angulationDeg = fractureInfo.angulation_deg ?? 3.5
  // Grounded detection evidence: show the actual upload whenever it is a
  // browser-renderable radiograph, with the detector box overlaid on it.
  const sourceImgUrl =
    caseData.kind === 'xray' && /\.(jpe?g|png|webp)$/i.test(caseData.sourceFilename || '')
      ? `/api/cases/${caseData.case_id}/source-image`
      : null
  const bbox = fractureDetected ? fractureInfo.bbox_norm || null : null

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
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    container.appendChild(renderer.domElement)

    // Neutral studio reflections so metal implants read as metal, not grey plastic.
    const pmrem = new THREE.PMREMGenerator(renderer)
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04)
    scene.environment = envTex

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    const light1 = new THREE.DirectionalLight(0xffffff, 1.15)
    light1.castShadow = true
    light1.shadow.mapSize.set(2048, 2048)
    light1.shadow.bias = -0.0004
    light1.position.set(120, 200, 260)
    scene.add(light1)

    const light2 = new THREE.DirectionalLight(0x67e8f9, 1.1)
    light2.position.set(-150, -80, -100)
    scene.add(light2)

    scene.add(new THREE.AmbientLight(0x64748b, 2.0))

    const loader = new GLTFLoader()
    loader.load(
      activeGlbUrl,
      (gltf) => {
        scene.add(gltf.scene)
        distalNodeRef.current = null
        mixerRef.current = null
        actionRef.current = null
        clipDurRef.current = 0

        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true
            child.receiveShadow = true
            if (!child.material) return
            const isImplant = child.name.toLowerCase().startsWith('implant_')
            // Guarantee a readable look even when the env map is weak/absent:
            // lit metal (not pure chrome) and completely matte bone.
            if (isImplant) {
              child.material.metalness = Math.min(child.material.metalness ?? 1, 0.6)
              child.material.roughness = Math.max(child.material.roughness ?? 0.25, 0.3)
              child.material.envMapIntensity = 1.3
            } else {
              child.material.metalness = 0
              child.material.roughness = 0.85
              child.material.envMapIntensity = 0.6
            }
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

        // Fit the shadow frustum to the model so shadows are crisp.
        light1.position.set(center.x + maxDim * 0.5, center.y + maxDim * 1.1, center.z + maxDim * 0.8)
        light1.target.position.copy(center)
        scene.add(light1.target)
        const shadowCam = light1.shadow.camera
        shadowCam.left = -maxDim
        shadowCam.right = maxDim
        shadowCam.top = maxDim
        shadowCam.bottom = -maxDim
        shadowCam.near = 1
        shadowCam.far = maxDim * 6
        shadowCam.updateProjectionMatrix()

        // Baked clip from the backend takes over pose control.
        if (gltf.animations?.length) {
          const mixer = new THREE.AnimationMixer(gltf.scene)
          const clip = gltf.animations[0]
          const action = mixer.clipAction(clip)
          action.paused = true
          action.play()
          mixerRef.current = mixer
          actionRef.current = action
          clipDurRef.current = clip.duration || 3.0
        } else if (assembly) {
          // Fallback staging for GLBs without a baked clip: retract implants,
          // then the gsap sequence below inserts them in order.
          const retracts = new Map()
          gltf.scene.traverse((child) => {
            if (!child.isMesh || !child.name.toLowerCase().startsWith('implant_')) return
            const childBox = new THREE.Box3().setFromObject(child)
            const childSize = childBox.getSize(new THREE.Vector3())
            let offset
            if (child.name.toLowerCase().includes('nail') && !child.name.toLowerCase().includes('screw')) {
              offset = new THREE.Vector3(0, 0, Math.max(childSize.z * 1.1, 180))
            } else {
              offset = new THREE.Vector3(-Math.min(Math.max(childSize.x * 2.5, 70), 160), 0, 0)
            }
            retracts.set(child, { offset, base: child.position.clone() })
            child.position.add(offset)
          })
          stagingRef.current = retracts
        } else {
          stagingRef.current = null
        }

        // AP-style framing: the reconstructed bone's long axis is +z, so look
        // in perpendicular to it (from the lateral side) with z as screen-up.
        camera.up.set(0, 0, 1)
        camera.position.set(center.x + maxDim * 0.8, center.y + maxDim * 1.9, center.z + maxDim * 0.3)
        camera.lookAt(center)
        controls.target.copy(center)

        // Assembled scenes begin displaced and animate into reduction.
        setDisplacement(100)
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
      if (introTweenRef.current) introTweenRef.current.kill()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
      envTex.dispose()
      pmrem.dispose()
      renderer.dispose()
    }
  }, [activeGlbUrl])

  // Intro: on each assembly load, play displacement 100 -> 0 (reduction) with easing.
  useEffect(() => {
    if (!assembly || !activeGlbUrl) return
    if (introTweenRef.current) introTweenRef.current.kill()
    const scrub = scrubObjRef.current
    scrub.v = 100
    introTweenRef.current = gsap.to(scrub, {
      v: 0,
      duration: 4.2,
      ease: 'power1.inOut',
      onUpdate: () => setDisplacement(scrub.v),
    })
    return () => {
      if (introTweenRef.current) introTweenRef.current.kill()
    }
  }, [assembly, activeGlbUrl])

  // Apply the displacement: scrub the baked clip, or move the distal node.
  useEffect(() => {
    const action = actionRef.current
    if (action) {
      action.time = (1 - displacement / 100.0) * clipDurRef.current
      mixerRef.current?.update(0)
      return
    }
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

    // Fallback insertion staging: hardware starts retracted and is inserted
    // after reduction, so slider 100% (broken) = retracted and 0% (reduced)
    // = fully inserted. Implanted parts stagger so the nail leads the screws.
    if (assembly && stagingRef.current?.size) {
      const total = stagingRef.current.size
      const insertProgress = 1 - displacement / 100.0
      let i = 0
      for (const [child, info] of stagingRef.current.entries()) {
        const local = Math.min(1, Math.max(0, insertProgress * total - i))
        const ease = local * local * (3 - 2 * local)
        child.position.copy(info.base).addScaledVector(info.offset, 1 - ease)
        i += 1
      }
    }
  }, [displacement, dispMm, angulationDeg, assembly])

  // Auto-play reduction / displacement animation loop (eased ping-pong)
  useEffect(() => {
    if (!isAutoPlaying) {
      return
    }
    const scrub = scrubObjRef.current
    scrub.v = displacement
    const tween = gsap.to(scrub, {
      v: 0,
      duration: 2.2,
      ease: 'power2.inOut',
      repeat: -1,
      yoyo: true,
      onUpdate: () => setDisplacement(scrub.v),
      onComplete: () => setDisplacement(scrub.v),
    })
    return () => {
      tween.kill()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAutoPlaying])

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
              : weakSignal
                ? <> · weak fracture signal below threshold</>
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
                <span className="font-medium text-cyan-300 capitalize text-right">
                  {fractureDetected ? (
                    <>
                      {`${fractureInfo.region} (AO/OTA ${fractureInfo.ao_family})`}
                      {regionSourceLabel && (
                        <span className="block text-[11px] font-normal text-slate-400 normal-case">
                          {zoneClass ? `zone: ${zoneClass} · ` : ''}
                          {fractureInfo.region_score != null && `${Math.round(fractureInfo.region_score * 100)}% · `}
                          via {regionSourceLabel}
                        </span>
                      )}
                    </>
                  ) : (
                    <>
                      Not determined by model
                      {viewHint && (
                        <span className="block text-[11px] font-normal text-slate-400 normal-case">
                          film reads as {viewHint} — view context only
                        </span>
                      )}
                    </>
                  )}
                </span>
              </div>
              {caseData.kind === 'xray' && onReAnalyze && (
                <div
                  className="flex items-center justify-between gap-2"
                  title={`Region source: ${regionSourceLabel || 'not determined'}. A surgeon override re-runs analysis with your chosen region.`}
                >
                  <span className="text-slate-400 text-xs">Region review</span>
                  <div className="flex gap-1">
                    {[['proximal', '31'], ['shaft', '32'], ['distal', '33']].map(([r, ao]) => (
                      <button
                        key={r}
                        disabled={regionBusy || fractureDetected && fractureInfo.region === r}
                        onClick={async () => {
                          setRegionBusy(true)
                          try {
                            setAssembly(null)
                            onReAnalyze(r)
                          } finally {
                            setRegionBusy(false)
                          }
                        }}
                        className={`px-2 py-0.5 rounded-md text-[11px] font-medium transition border ${
                          fractureDetected && fractureInfo.region === r
                            ? 'bg-cyan-950/60 border-cyan-500/60 text-cyan-300'
                            : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                        } disabled:opacity-60`}
                      >
                        {r} <span className="text-slate-500">{ao}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
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
                <span className="text-slate-500">
                  {fractureDetected && fractureInfo.confidence_basis?.includes('calibration')
                    ? 'Fracture Probability'
                    : 'Detector Confidence'}
                </span>
                <span
                  className={`font-medium ${fractureDetected ? 'text-emerald-400' : weakSignal ? 'text-amber-300' : 'text-slate-500'}`}
                  title={
                    fractureDetected && fractureInfo.confidence_basis?.includes('calibration')
                      ? 'Calibrated on FracAtlas validation; the raw detector score is shown for reference.'
                      : undefined
                  }
                >
                  {fractureDetected
                    ? `${((fractureInfo.confidence ?? 0) * 100).toFixed(0)}%`
                    : weakSignal
                      ? `${((weakSignal.max_confidence ?? 0) * 100).toFixed(0)}% (weak)`
                      : '—'}
                  {fractureDetected && fractureInfo.confidence_basis?.includes('calibration') && fractureInfo.detector_confidence != null && (
                    <span className="ml-1 text-[10px] font-normal text-slate-500">
                      score {(fractureInfo.detector_confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </span>
              </div>
            </div>
            {!fractureDetected && (
              <div className={`rounded-xl border p-3 text-xs leading-relaxed ${weakSignal ? "border-amber-500/50 bg-amber-500/10 text-amber-200" : "border-amber-500/30 bg-amber-500/10 text-amber-200"}`}>
                {weakSignal ? (
                  <>
                    <strong>Weak fracture signal.</strong> The model produced {weakSignal.candidates} faint
                    candidate {weakSignal.candidates === 1 ? "line" : "lines"} (strongest raw score {((weakSignal.max_confidence ?? 0) * 100).toFixed(0)}%
                    {weakSignal.calibrated_probability != null
                      ? `, ≈${(weakSignal.calibrated_probability * 100).toFixed(0)}% calibrated probability`
                      : ""}),
                    below its {((weakSignal.gate_conf ?? 0.15) * 100).toFixed(0)}% raw-score gate.
                    This is not a confirmation of a fracture, and a flat negative is not evidence of an
                    intact femur — review the radiograph or consult a specialist.
                    {viewContext}
                  </>
                ) : (
                  <>
                    <strong>No fracture detected.</strong> This is not evidence of an intact femur. Review the radiograph or try a different image.
                    {viewContext}
                  </>
                )}
              </div>
            )}
          </section>

          {/* Grounded detection evidence: the actual upload with the model box. */}
          {sourceImgUrl && !sourceImgFailed && (
            <section className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Detection Evidence</h2>
              <div className="relative rounded-xl border border-slate-800 overflow-hidden bg-black shadow-sm">
                <img
                  src={sourceImgUrl}
                  alt="Uploaded radiograph"
                  className="block w-full"
                  onError={() => setSourceImgFailed(true)}
                />
                {bbox && (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                  >
                    <rect
                      x={bbox[0] * 100}
                      y={bbox[1] * 100}
                      width={(bbox[2] - bbox[0]) * 100}
                      height={(bbox[3] - bbox[1]) * 100}
                      fill="rgba(251,191,36,0.10)"
                      stroke="#fbbf24"
                      strokeWidth="1.5"
                      vectorEffect="non-scaling-stroke"
                    />
                  </svg>
                )}
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                {fractureDetected ? (
                  <>
                    Amber box = the model-detected fracture region ({((fractureInfo.confidence ?? 0) * 100).toFixed(0)}%{' '}
                    {fractureInfo.confidence_basis?.includes('calibration') ? 'calibrated probability' : 'raw confidence'}).
                    Educational estimate — verify the box against the original film.
                  </>
                ) : (
                  <>
                    No detection box: the model found nothing above its{' '}
                    {((weakSignal?.gate_conf ?? 0.15) * 100).toFixed(0)}% raw-score gate
                    {weakSignal ? `, but flagged ${weakSignal.candidates} weak candidate${weakSignal.candidates === 1 ? '' : 's'}` : ''}.
                  </>
                )}
              </p>
            </section>
          )}

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
                    {displacement > 97
                      ? 'Displaced Fractured State'
                      : displacement > 5
                        ? 'Reducing the fracture…'
                        : 'Fracture contact — fragment settling / implant seating'}
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
                    introTweenRef.current?.kill()
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
