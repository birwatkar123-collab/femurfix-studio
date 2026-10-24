import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function SimView({ caseData, onReset }) {
  const containerRef = useRef(null)
  const [selectedImplant, setSelectedImplant] = useState(caseData.recommendations?.options?.[0]?.id)
  const [loading, setLoading] = useState(false)
  const [assembly, setAssembly] = useState(null)
  const [error, setError] = useState('')
  const [audience, setAudience] = useState('surgeon')
  const [showAnimation, setShowAnimation] = useState(true)

  const activeGlbUrl = assembly?.glb_url || caseData.preview_glb_url

  useEffect(() => {
    if (!containerRef.current || !activeGlbUrl) return
    const container = containerRef.current
    container.innerHTML = ''
    const width = container.clientWidth
    const height = container.clientHeight
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#020617')
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    const light1 = new THREE.DirectionalLight(0xffffff, 1.2)
    light1.position.set(100, 200, 250)
    scene.add(light1)
    const light2 = new THREE.DirectionalLight(0xa5f3fc, 0.8)
    light2.position.set(-100, -100, -100)
    scene.add(light2)
    scene.add(new THREE.AmbientLight(0x64748b, 1.5))
    const loader = new GLTFLoader()
    loader.load(activeGlbUrl, (gltf) => {
      scene.add(gltf.scene)
      const box = new THREE.Box3().setFromObject(gltf.scene)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z, 100)
      camera.position.set(center.x, center.y + maxDim * 0.2, center.z + maxDim * 1.8)
      camera.lookAt(center)
      controls.target.copy(center)
    }, undefined, (err) => console.error('Failed to load GLB:', err))
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
      } catch {
        // Non-JSON response
      }
      if (!res.ok) throw new Error(data?.detail || `Assembly failed (${res.status} ${res.statusText || 'Server error'}).`)
      setAssembly(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const options = caseData.recommendations?.options || []
  const selected = options.find(o => o.id === selectedImplant)

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-cyan-400">FemurFix Studio</p>
          <p className="text-sm text-slate-400">Case {caseData.case_id} · {caseData.analysis?.fracture?.region} fracture · AO/OTA {caseData.analysis?.fracture?.ao_family}</p>
        </div>
        <button onClick={onReset} className="px-4 py-1.5 rounded-lg border border-slate-700 text-sm hover:bg-slate-800">New Case</button>
      </header>
      <main className="grid lg:grid-cols-[380px,1fr] gap-0 h-[calc(100vh-73px)]">
        <aside className="border-r border-slate-800 overflow-y-auto p-6 space-y-6">
          <section className="space-y-3">
            <h2 className="text-lg font-medium">Analysis</h2>
            <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 space-y-2 text-sm">
              <p><span className="text-slate-400">Region:</span> {caseData.analysis?.fracture?.region}</p>
              <p><span className="text-slate-400">Pattern:</span> {caseData.analysis?.fracture?.pattern}</p>
              <p><span className="text-slate-400">Confidence:</span> {(caseData.analysis?.fracture?.confidence*100).toFixed(0)}%</p>
              <p className="text-xs text-slate-500">Reconstruction: {caseData.analysis?.imaging?.modality || 'estimated'}</p>
            </div>
          </section>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">Implant Recommendations</h2>
            <div className="space-y-2">
              {options.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setSelectedImplant(opt.id)}
                  className={`w-full text-left rounded-lg border p-4 space-y-2 transition ${selectedImplant === opt.id ? 'border-cyan-400 bg-cyan-400/10' : 'border-slate-800 bg-slate-900/60 hover:border-slate-600'}`}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-medium">{opt.name}</p>
                    <p className="text-xs text-cyan-300">{(opt.score*100).toFixed(0)}%</p>
                  </div>
                  <p className="text-xs text-slate-400">{opt.surgeon_text}</p>
                </button>
              ))}
            </div>
          </section>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">Assemble</h2>
            <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 space-y-3">
              <div className="flex gap-2">
                <button onClick={() => setAudience('surgeon')} className={`flex-1 px-3 py-1.5 rounded-lg text-sm ${audience === 'surgeon' ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800'}`}>Surgeon</button>
                <button onClick={() => setAudience('patient')} className={`flex-1 px-3 py-1.5 rounded-lg text-sm ${audience === 'patient' ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800'}`}>Patient</button>
              </div>
              <button onClick={assemble} disabled={loading || !selectedImplant} className="w-full px-4 py-2 rounded-lg bg-cyan-400 text-slate-950 font-medium disabled:opacity-50 hover:bg-cyan-300">
                {loading ? 'Assembling...' : 'Generate Animated Assembly'}
              </button>
              {error && <p className="text-sm text-rose-400">{error}</p>}
            </div>
          </section>
          {assembly && (
            <section className="space-y-3">
              <h2 className="text-lg font-medium">Assembly Result</h2>
              <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 space-y-2 text-sm">
                <p><span className="text-slate-400">Device:</span> {assembly.device.name}</p>
                <p><span className="text-slate-400">Insertion:</span> {assembly.device.insertion}</p>
                <p className="text-slate-300">{assembly.narrative.text}</p>
                <a href={assembly.glb_url} target="_blank" rel="noreferrer" className="inline-flex px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700">Download/ Open GLB</a>
                <p className="text-xs text-amber-400">{assembly.disclaimer}</p>
              </div>
            </section>
          )}
        </aside>
        <section className="relative">
          <div ref={containerRef} className="w-full h-full" />
          <div className="absolute top-4 left-4 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-1.5 text-xs text-slate-400">Educational simulation only · Not medical advice</div>
        </section>
      </main>
    </div>
  )
}
