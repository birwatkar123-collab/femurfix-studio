import React, { useState } from 'react'
import DisclaimerGate from './components/DisclaimerGate.jsx'
import UploadStep from './components/UploadStep.jsx'
import SimView from './components/SimView.jsx'

export default function App() {
  const [accepted, setAccepted] = useState(false)
  const [stage, setStage] = useState('upload')
  const [caseData, setCaseData] = useState(null)

  const onCaseCreated = async (created) => {
    try {
      const res = await fetch(`/api/cases/${created.case_id}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      })
      let data = null
      try {
        data = await res.json()
      } catch {
        // Non-JSON response
      }
      if (!res.ok) throw new Error(data?.detail || `Analysis failed (${res.status} ${res.statusText || 'Server error'}).`)
      setCaseData({
        case_id: created.case_id,
        kind: created.kind,
        analysis: data.analysis,
        preview_glb_url: data.preview_glb_url,
        recommendations: data.recommendations
      })
      setStage('simulate')
    } catch (err) {
      alert(err.message)
    }
  }

  const onReset = () => {
    setStage('upload')
    setCaseData(null)
  }

  if (!accepted) return <DisclaimerGate onAccept={() => setAccepted(true)} />
  if (stage === 'upload') return <UploadStep onCaseCreated={onCaseCreated} />
  if (stage === 'simulate' && caseData) return <SimView caseData={caseData} onReset={onReset} />
  return <UploadStep onCaseCreated={onCaseCreated} />
}
