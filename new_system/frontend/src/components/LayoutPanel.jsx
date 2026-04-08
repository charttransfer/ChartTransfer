import React, { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import { toPng } from 'html-to-image'
import './LayoutPanel.css'

function pollTask(taskId, onUpdate, intervalMs = 1500) {
  const poll = () => {
    axios.get(`/api/task-status/${taskId}`)
      .then(res => {
        const { status, result } = res.data
        onUpdate(status, result)
        if (status !== 'done' && status !== 'error') {
          setTimeout(poll, intervalMs)
        }
      })
      .catch(() => onUpdate('error', null))
  }
  setTimeout(poll, intervalMs)
}

function SvgTextElement({ svgData, width, height }) {
  if (!svgData?.lines) return null
  const { lines, viewBox } = svgData
  const svgW = svgData.width || width
  const svgH = svgData.height || height
  const vb = viewBox || `0 0 ${svgW} ${svgH}`

  return (
    <svg
      width={width}
      height={height}
      viewBox={vb}
      preserveAspectRatio="xMinYMin meet"
      style={{ display: 'block', overflow: 'hidden' }}
    >
      {lines.map((line, li) => (
        <text
          key={li}
          x={line.x}
          y={line.y}
          opacity={line.opacity}
          textAnchor={line.text_anchor}
        >
          {(line.segments || []).map((seg, si) => (
            <tspan
              key={si}
              fontFamily={seg.font_family || 'sans-serif'}
              fontWeight={seg.font_weight || 'normal'}
              fontSize={seg.font_size || 16}
              fill={seg.fill || '#000'}
            >
              {seg.text}
            </tspan>
          ))}
        </text>
      ))}
    </svg>
  )
}

function LayoutPanel({ selectedExample, selectedData, sceneGraphData, genState, onBilevelSceneGraphChange, sessionId }) {
  const [elements, setElements] = useState([])
  const [canvasSize, setCanvasSize] = useState({ w: 1080, h: 1080 })
  const [bgColor, setBgColor] = useState('#ffffff')
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 0.5 })
  const [selectedEl, setSelectedEl] = useState(null)
  const [optimizing, setOptimizing] = useState(false)
  const [resultImage, setResultImage] = useState(null)
  const [optimizerType] = useState('bilevel')
  const [bilevelCandidates, setBilevelCandidates] = useState([])
  const [activeCandidateIdx, setActiveCandidateIdx] = useState(0)

  const viewportRef = useRef(null)
  const canvasRef = useRef(null)
  const dragRef = useRef(null)
  const initialElementsRef = useRef([])

  const centerBboxes = useCallback((bboxMap, els) => {
    const matched = els.filter(el => bboxMap[el.path])
    if (matched.length === 0) return bboxMap
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const el of matched) {
      const b = bboxMap[el.path]
      minX = Math.min(minX, b.x)
      minY = Math.min(minY, b.y)
      maxX = Math.max(maxX, b.x + b.width)
      maxY = Math.max(maxY, b.y + b.height)
    }
    const contentW = maxX - minX
    const contentH = maxY - minY
    const offsetX = (canvasSize.w - contentW) / 2 - minX
    const offsetY = (canvasSize.h - contentH) / 2 - minY
    if (Math.abs(offsetX) < 5 && Math.abs(offsetY) < 5) return bboxMap
    const centered = {}
    for (const [key, b] of Object.entries(bboxMap)) {
      centered[key] = { ...b, x: b.x + offsetX, y: b.y + offsetY }
    }
    return centered
  }, [canvasSize])

  useEffect(() => {
    if (!selectedExample?.id || !selectedData?.data_file) {
      setElements([])
      setResultImage(null)
      return
    }
    axios.get(`/api/layout-elements/${selectedExample.id}/${selectedData.data_file}`, {
      params: { session_id: sessionId },
    }).then(res => {
        const { elements: els, canvasWidth, canvasHeight, backgroundColor } = res.data
        setElements(els || [])
        initialElementsRef.current = JSON.parse(JSON.stringify(els || []))
        setCanvasSize({ w: canvasWidth || 1080, h: canvasHeight || 1080 })
        setBgColor(backgroundColor || '#ffffff')
        setSelectedEl(null)
        setResultImage(null)
      })
      .catch(() => {
        setElements([])
      })
  }, [selectedExample?.id, selectedData?.data_file, sessionId])

  useEffect(() => {
    if (!elements.length || !genState) return
    setElements(prev => prev.map(el => {
      const gs = genState[el.path]
      if (!gs) return el
      let imageUrl = el.imageUrl
      let generated = el.generated
      if (el.type === 'text' && gs.pngUrl) {
        imageUrl = gs.pngUrl
        generated = true
      } else if (el.type === 'image' && gs.imageUrl) {
        imageUrl = gs.imageUrl
        generated = true
      } else if (el.type === 'chart' && gs.chartUrl) {
        imageUrl = gs.chartUrl
        generated = true
      }
      const urlChanged = imageUrl !== el.imageUrl
      return { ...el, imageUrl, generated, svgData: gs.svgData || null,
        fitWidth: urlChanged ? undefined : el.fitWidth,
        fitHeight: urlChanged ? undefined : el.fitHeight,
      }
    }))
  }, [genState])

  const fitToView = useCallback(() => {
    if (!viewportRef.current) return
    const vw = viewportRef.current.clientWidth
    const vh = viewportRef.current.clientHeight
    const padding = 40
    const scaleX = (vw - padding * 2) / canvasSize.w
    const scaleY = (vh - padding * 2) / canvasSize.h
    const s = Math.min(scaleX, scaleY, 1)
    const x = (vw - canvasSize.w * s) / 2
    const y = (vh - canvasSize.h * s) / 2
    setTransform({ x, y, scale: s })
  }, [canvasSize])

  useEffect(() => {
    if (elements.length > 0) fitToView()
  }, [canvasSize])

  useEffect(() => {
    if (!optimizing && elements.length > 0) {
      const timer = setTimeout(() => fitToView(), 200)
      return () => clearTimeout(timer)
    }
  }, [optimizing, fitToView])

  const handleWheel = useCallback((e) => {
    e.preventDefault()
    const rect = viewportRef.current.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const factor = e.deltaY < 0 ? 1.1 : 0.9
    setTransform(prev => {
      const newScale = Math.max(0.05, Math.min(5, prev.scale * factor))
      const ratio = newScale / prev.scale
      return {
        x: mx - (mx - prev.x) * ratio,
        y: my - (my - prev.y) * ratio,
        scale: newScale,
      }
    })
  }, [])

  useEffect(() => {
    const vp = viewportRef.current
    if (!vp) return
    vp.addEventListener('wheel', handleWheel, { passive: false })
    return () => vp.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  const handleMouseDown = useCallback((e) => {
    if (e.target.closest('.lp-element') || e.target.closest('.lp-toolbar')) return
    setSelectedEl(null)
    dragRef.current = { type: 'pan', startX: e.clientX, startY: e.clientY, origTx: transform.x, origTy: transform.y }
    e.preventDefault()
  }, [transform])

  const handleElementMouseDown = useCallback((e, path) => {
    e.stopPropagation()
    if (e.target.closest('.lp-resize-handle')) {
      const el = elements.find(x => x.path === path)
      if (!el) return
      const displayW = el.fitWidth || el.bbox.width
      const displayH = el.fitHeight || el.bbox.height
      dragRef.current = {
        type: 'resize',
        path,
        startX: e.clientX,
        startY: e.clientY,
        origW: displayW,
        origH: displayH,
        aspect: displayW / displayH,
      }
    } else {
      const el = elements.find(x => x.path === path)
      if (!el) return
      dragRef.current = {
        type: 'drag',
        path,
        startX: e.clientX,
        startY: e.clientY,
        origX: el.bbox.x,
        origY: el.bbox.y,
      }
    }
    setSelectedEl(path)
    e.preventDefault()
  }, [elements])

  useEffect(() => {
    const handleMouseMove = (e) => {
      const d = dragRef.current
      if (!d) return
      const dx = e.clientX - d.startX
      const dy = e.clientY - d.startY

      if (d.type === 'pan') {
        setTransform(prev => ({
          ...prev,
          x: d.origTx + dx,
          y: d.origTy + dy,
        }))
      } else if (d.type === 'drag') {
        setElements(prev => prev.map(el =>
          el.path === d.path
            ? { ...el, bbox: { ...el.bbox, x: d.origX + dx / transform.scale, y: d.origY + dy / transform.scale } }
            : el
        ))
      } else if (d.type === 'resize') {
        const newW = Math.max(20, d.origW + dx / transform.scale)
        const newH = newW / d.aspect
        setElements(prev => prev.map(el =>
          el.path === d.path
            ? { ...el, bbox: { ...el.bbox, width: newW, height: newH }, fitWidth: undefined, fitHeight: undefined }
            : el
        ))
      }
    }
    const handleMouseUp = () => { dragRef.current = null }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [transform.scale])

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedEl) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
        setElements(prev => prev.filter(el => el.path !== selectedEl))
        setSelectedEl(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedEl])

  const handleOptimize = useCallback(() => {
    if (!selectedExample?.id || !selectedData?.data_file || optimizing) return
    const excludedPaths = []
    const initialPaths = new Set(initialElementsRef.current.map(e => e.path))
    const currentPaths = new Set(elements.map(e => e.path))
    initialPaths.forEach(p => { if (!currentPaths.has(p)) excludedPaths.push(p) })

    setOptimizing(true)
    setResultImage(null)
    setBilevelCandidates([])
    setActiveCandidateIdx(0)
    axios.post('/api/run-layout', {
      example_id: selectedExample.id,
      data_file: selectedData.data_file,
      elements: elements.map(el => ({ path: el.path, type: el.type, bbox: el.bbox })),
      excludedPaths,
      optimizer: optimizerType,
      session_id: sessionId,
    }).then(res => {
      const { task_id } = res.data
      pollTask(task_id, (status, result) => {
        if (status === 'done' && result?.bboxes) {
          console.log('[handleOptimize] raw result.bboxes:', JSON.stringify(result.bboxes))
          const bboxes = centerBboxes(result.bboxes, elements)
          console.log('[handleOptimize] centered bboxes:', JSON.stringify(bboxes))
          setElements(prev => prev.map(el => {
            const nb = bboxes[el.path]
            console.log(`[handleOptimize] el.path=${el.path}, matched bbox=`, nb ? `x=${nb.x.toFixed(1)},y=${nb.y.toFixed(1)},w=${nb.width.toFixed(1)},h=${nb.height.toFixed(1)}` : 'HIDDEN')
            if (nb) return { ...el, bbox: { x: nb.x, y: nb.y, width: nb.width, height: nb.height }, fitWidth: undefined, fitHeight: undefined, hidden: false }
            return { ...el, hidden: true }
          }))
          if (result.resultImage) setResultImage(result.resultImage)
          if (result.bilevelCandidates?.length > 0) {
            console.log('[handleOptimize] bilevel candidates:', result.bilevelCandidates.length)
            result.bilevelCandidates.forEach((c, i) => {
              console.log(`[handleOptimize] candidate ${i}: variation=${c.variation_id}, bboxes=`, JSON.stringify(c.bboxes))
            })
            const centeredCandidates = result.bilevelCandidates.map(c => ({
              ...c,
              bboxes: centerBboxes(c.bboxes, elements),
            }))
            setBilevelCandidates(centeredCandidates)
            setActiveCandidateIdx(0)
            onBilevelSceneGraphChange?.(centeredCandidates[0]?.sceneGraph || null)
          }
          setOptimizing(false)
        } else if (status === 'error') {
          setOptimizing(false)
        }
      })
    }).catch(() => setOptimizing(false))
  }, [selectedExample, selectedData, elements, optimizing, optimizerType, sessionId])

  const switchCandidate = useCallback((idx) => {
    if (idx < 0 || idx >= bilevelCandidates.length) return
    const cand = bilevelCandidates[idx]
    console.log('[switchCandidate] idx=', idx, 'variation=', cand.variation_id, 'bboxes=', JSON.stringify(cand.bboxes))
    setActiveCandidateIdx(idx)
    setElements(prev => prev
      .map(el => {
        const nb = cand.bboxes[el.path]
        console.log(`[switchCandidate] el.path=${el.path}, matched bbox=`, nb ? `x=${nb.x.toFixed(1)},y=${nb.y.toFixed(1)},w=${nb.width.toFixed(1)},h=${nb.height.toFixed(1)}` : 'HIDDEN')
        if (nb) return { ...el, bbox: { x: nb.x, y: nb.y, width: nb.width, height: nb.height }, fitWidth: undefined, fitHeight: undefined, hidden: false }
        return { ...el, hidden: true }
      }))
    setResultImage(cand.resultImage || null)
    onBilevelSceneGraphChange?.(cand.sceneGraph || null)
    setTimeout(() => fitToView(), 150)
  }, [bilevelCandidates, fitToView, onBilevelSceneGraphChange])

  const handleReset = useCallback(() => {
    setElements(JSON.parse(JSON.stringify(initialElementsRef.current)).map(el => ({ ...el, hidden: false })))
    setSelectedEl(null)
    setResultImage(null)
    setBilevelCandidates([])
    setActiveCandidateIdx(0)
    onBilevelSceneGraphChange?.(null)
  }, [onBilevelSceneGraphChange])

  const handleExport = useCallback(() => {
    if (!canvasRef.current) return
    toPng(canvasRef.current, {
      width: canvasSize.w,
      height: canvasSize.h,
      pixelRatio: 2,
      style: { transform: 'none', position: 'static' },
    }).then(dataUrl => {
      const link = document.createElement('a')
      link.download = `layout_${selectedExample?.id || 'export'}.png`
      link.href = dataUrl
      link.click()
    })
  }, [canvasSize, selectedExample])

  if (!selectedExample) {
    return (
      <div className="layout-panel">
        <div className="panel-header">Layout View</div>
        <div className="panel-content">
          <div className="placeholder-text">Select an example and data to view layout</div>
        </div>
      </div>
    )
  }

  return (
    <div className="layout-panel">
      <div className="panel-header">Layout View</div>
      <div className="lp-toolbar">
        <button
          className="lp-btn lp-btn-primary"
          onClick={handleOptimize}
          disabled={optimizing || elements.length === 0}
        >
          {optimizing ? 'Optimizing\u2026' : 'Optimize Layout'}
        </button>
        <button className="lp-btn" onClick={handleReset}>Reset</button>
        <button className="lp-btn" onClick={fitToView}>Fit View</button>
        <button className="lp-btn" onClick={handleExport} disabled={elements.length === 0}>Export PNG</button>
        <label className="lp-bg-picker">
          <span>BG</span>
          <input
            type="color"
            value={bgColor}
            onChange={(e) => setBgColor(e.target.value)}
          />
        </label>
        {selectedEl && (
          <span className="lp-selection-hint">
            Selected: {elements.find(e => e.path === selectedEl)?.label || selectedEl}
          </span>
        )}
      </div>
      {bilevelCandidates.length > 1 && (
        <div className="lp-candidate-bar">
          <span className="lp-candidate-label">Layout Results:</span>
          {bilevelCandidates.map((cand, idx) => {
            const isInitial = cand.variation_id === 'initial'
            return (
              <button
                key={idx}
                className={`lp-candidate-btn ${idx === activeCandidateIdx ? 'active' : ''} ${isInitial ? 'initial' : ''}`}
                onClick={() => switchCandidate(idx)}
              >
                {isInitial ? 'Original' : `#${idx + 1}`}
                <span className="lp-candidate-loss">loss: {cand.loss}</span>
              </button>
            )
          })}
        </div>
      )}
      <div
        className="lp-viewport"
        ref={viewportRef}
        onMouseDown={handleMouseDown}
      >
        <div
          className="lp-canvas"
          ref={canvasRef}
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            width: canvasSize.w,
            height: canvasSize.h,
          }}
        >
          <div className="lp-bg" style={{ width: canvasSize.w, height: canvasSize.h, backgroundColor: bgColor }} />
          {resultImage && (
            <img
              src={resultImage}
              alt="Layout result"
              draggable={false}
              className="lp-result-image"
              style={{ width: canvasSize.w, height: canvasSize.h }}
            />
          )}
          {!resultImage && [...elements].filter(el => !el.hidden).sort((a, b) => (a.type === 'chart' ? -1 : b.type === 'chart' ? 1 : 0)).map(el => {
            const gs = genState?.[el.path]
            const isSelected = selectedEl === el.path
            const displayW = el.fitWidth || el.bbox.width
            const displayH = el.fitHeight || el.bbox.height
            return (
              <div
                key={el.path}
                className={`lp-element ${isSelected ? 'selected' : ''} ${el.generated ? '' : 'placeholder'}`}
                style={{
                  left: el.bbox.x,
                  top: el.bbox.y,
                  width: displayW,
                  height: displayH,
                }}
                onMouseDown={(e) => handleElementMouseDown(e, el.path)}
              >
                {el.imageUrl ? (
                  <img
                    src={el.imageUrl}
                    alt={el.label}
                    draggable={false}
                    onLoad={(e) => {
                      const img = e.target
                      const natW = img.naturalWidth
                      const natH = img.naturalHeight
                      if (natW > 0 && natH > 0) {
                        const scaleX = el.bbox.width / natW
                        const scaleY = el.bbox.height / natH
                        const s = Math.min(scaleX, scaleY)
                        const fitW = natW * s
                        const fitH = natH * s
                        if (Math.abs(fitW - el.bbox.width) > 1 || Math.abs(fitH - el.bbox.height) > 1) {
                          setElements(prev => prev.map(e2 =>
                            e2.path === el.path ? { ...e2, fitWidth: fitW, fitHeight: fitH } : e2
                          ))
                        }
                      }
                    }}
                  />
                ) : el.type === 'text' && gs?.svgData ? (
                  <SvgTextElement svgData={gs.svgData} width={displayW} height={displayH} />
                ) : (
                  <div className="lp-placeholder-inner">
                    <span>{el.type}</span>
                    <span className="lp-placeholder-label">{el.label}</span>
                  </div>
                )}
                {isSelected && <div className="lp-resize-handle" />}
              </div>
            )
          })}
        </div>
        {optimizing && (
          <div className="lp-optimizing-overlay">
            <div className="lp-optimizing-content">
              <span className="lp-optimizing-spinner" />
              <span>Optimizing layout…</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default LayoutPanel
