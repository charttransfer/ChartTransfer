import React, { useState, useCallback } from 'react'
import { Routes, Route } from 'react-router-dom'
import WorkspaceView from './views/WorkspaceView'

function sanitizeUsername(name) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '_')
}

function SessionGate({ children }) {
  const [username, setUsername] = useState(() => localStorage.getItem('chartgalaxy_user') || '')
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('chartgalaxy_user')
    return saved ? sanitizeUsername(saved) : null
  })

  const handleStart = useCallback(() => {
    if (!username.trim()) return
    const sid = sanitizeUsername(username.trim())
    localStorage.setItem('chartgalaxy_user', username.trim())
    setSessionId(sid)
  }, [username])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter') handleStart()
  }, [handleStart])

  if (!sessionId) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '100vh', background: '#f0f2f5',
      }}>
        <div style={{
          background: '#fff', borderRadius: 12, padding: '48px 40px',
          boxShadow: '0 2px 16px rgba(0,0,0,0.08)', maxWidth: 400, width: '100%',
          textAlign: 'center',
        }}>
          <h1 style={{ margin: '0 0 8px', fontSize: 24, color: '#1a1a2e' }}>ChartTransfer</h1>
          <p style={{ margin: '0 0 28px', color: '#666', fontSize: 14 }}>Enter your username to start</p>
          <input
            type="text"
            placeholder="your_username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              width: '100%', padding: '10px 14px', fontSize: 15,
              border: '1px solid #d0d5dd', borderRadius: 8,
              outline: 'none', boxSizing: 'border-box', marginBottom: 16,
            }}
            autoFocus
          />
          <button
            onClick={handleStart}
            disabled={!username.trim()}
            style={{
              width: '100%', padding: '10px 0', fontSize: 15, fontWeight: 600,
              background: username.trim() ? '#4f46e5' : '#c7c7cc', color: '#fff',
              border: 'none', borderRadius: 8, cursor: username.trim() ? 'pointer' : 'default',
            }}
          >
            Start
          </button>
        </div>
      </div>
    )
  }

  return children(sessionId)
}

function App() {
  return (
    <SessionGate>
      {(sessionId) => (
        <Routes>
          <Route path="/:userId" element={<WorkspaceView sessionId={sessionId} />} />
          <Route path="/" element={<WorkspaceView sessionId={sessionId} />} />
        </Routes>
      )}
    </SessionGate>
  )
}

export default App
