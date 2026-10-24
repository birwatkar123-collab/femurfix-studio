import React, { useState } from 'react'

export default function UploadStep({ onCaseCreated }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select an image or volume.')
      return
    }
    setLoading(true)
    setError('')
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/api/cases', {
        method: 'POST',
        body: form
      })
      let data = null
      try {
        data = await res.json()
      } catch {
        // Non-JSON response (e.g. proxy error or gateway timeout)
      }
      if (!res.ok) {
        throw new Error(data?.detail || `Upload failed (${res.status} ${res.statusText || 'Server unreachable'}). Please ensure the backend is running.`)
      }
      onCaseCreated(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
      <div className="max-w-xl w-full space-y-6">
        <header className="space-y-2 text-center">
          <p className="text-xs uppercase tracking-[0.4em] text-cyan-400">FemurFix Studio</p>
          <h1 className="text-3xl font-semibold">Upload Imaging</h1>
          <p className="text-slate-400">X-ray (primary), CT, or MRI. JPEG/PNG/DICOM/NIfTI supported by backend.</p>
        </header>
        <form onSubmit={handleSubmit} className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-slate-300">Imaging file</label>
            <input
              type="file"
              accept="image/*,.dcm,.dicom,.nii,.nii.gz,.npy"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-cyan-400 file:text-slate-950 hover:file:bg-cyan-300"
            />
            {file && <p className="text-xs text-slate-500">{file.name} · {(file.size/1024/1024).toFixed(2)} MB</p>}
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <button
            disabled={loading || !file}
            className="w-full px-4 py-2 rounded-lg bg-cyan-400 text-slate-950 font-medium disabled:opacity-50 hover:bg-cyan-300 transition"
          >
            {loading ? 'Uploading...' : 'Upload & Analyze'}
          </button>
        </form>
      </div>
    </div>
  )
}
