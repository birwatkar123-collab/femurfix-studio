import React from 'react'

const DISCLAIMER = "FemurFix Studio is an educational simulation only. It is not medical advice, diagnosis, or a substitute for professional clinical judgement. Do not use for clinical decision making."

export default function DisclaimerGate({ onAccept }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
      <div className="max-w-2xl w-full space-y-6">
        <header className="space-y-2 text-center">
          <p className="text-xs uppercase tracking-[0.4em] text-cyan-400">FemurFix Studio</p>
          <h1 className="text-3xl md:text-4xl font-semibold">Educational Femur Fracture Fixation Simulator</h1>
          <p className="text-slate-400">Upload a femur X-ray/CT/MRI to explore fracture analysis, implant ideas and an animated assembly.</p>
        </header>
        <div className="rounded-xl border border-amber-500/60 bg-amber-500/10 p-6 space-y-4">
          <h2 className="text-lg font-medium text-amber-300">Important Disclaimer</h2>
          <p className="text-slate-200 leading-relaxed">{DISCLAIMER}</p>
          <p className="text-xs text-amber-200/80">By continuing you acknowledge this is for educational purposes only.</p>
        </div>
        <div className="flex flex-col md:flex-row gap-3 justify-center">
          <button
            onClick={onAccept}
            className="px-6 py-2 rounded-lg bg-cyan-400 text-slate-950 font-medium hover:bg-cyan-300 transition"
          >
            I understand — Enter Studio
          </button>
        </div>
      </div>
    </div>
  )
}
