import React, { useState, useRef, useEffect, useMemo } from 'react'
import './SceneTree.css'

function deepEqual(a, b) {
  if (a === b) return true
  if (a == null || b == null) return a == b
  if (typeof a === 'number' && typeof b === 'number') {
    return Math.abs(a - b) < 0.01
  }
  if (typeof a !== typeof b) return false
  if (typeof a !== 'object') return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false
    return a.every((v, i) => deepEqual(v, b[i]))
  }
  const keysA = Object.keys(a), keysB = Object.keys(b)
  if (keysA.length !== keysB.length) return false
  return keysA.every(k => deepEqual(a[k], b[k]))
}

function computeDiffMap(original, optimized, path = '0', out = {}) {
  if (!original || !optimized) return out

  const origChildren = original.children || []
  const optChildren = optimized.children || []
  const isLeafNode = origChildren.length === 0 && optChildren.length === 0
  if (isLeafNode) return out

  const diff = {}

  const origC = original.constraints || {}
  const optC = optimized.constraints || {}
  const cstDiffs = {}
  const allCstKeys = new Set([...Object.keys(origC), ...Object.keys(optC)])
  for (const key of allCstKeys) {
    if (!deepEqual(origC[key], optC[key])) {
      cstDiffs[key] = { from: origC[key], to: optC[key] }
    }
  }
  if (Object.keys(cstDiffs).length > 0) {
    diff.constraintsChanged = cstDiffs
  }

  if (origChildren.length !== optChildren.length) {
    diff.childCountChanged = { from: origChildren.length, to: optChildren.length }
  }

  if (Object.keys(diff).length > 0) {
    out[path] = diff
  }

  const minLen = Math.min(origChildren.length, optChildren.length)
  for (let i = 0; i < minLen; i++) {
    computeDiffMap(origChildren[i], optChildren[i], `${path}-${i}`, out)
  }
  for (let i = minLen; i < optChildren.length; i++) {
    out[`${path}-${i}`] = { addedNode: true }
  }

  return out
}

const NODE_COLORS = {
  layer: '#6366f1',
  column: '#8b5cf6',
  row: '#a78bfa',
  text: '#2563eb',
  chart: '#059669',
  image: '#d97706',
  legend: '#dc2626',
}

const LEAF_TYPES = new Set(['text', 'chart', 'image', 'legend'])

function getNodeColor(type) {
  return NODE_COLORS[type?.toLowerCase()] || '#6b7280'
}

function isLeaf(node) {
  return LEAF_TYPES.has(node.type?.toLowerCase()) || !node.children?.length
}

function StepBadge({ step, status }) {
  const cls = status === 'done' ? 'done' : status === 'generating' ? 'active' : 'idle'
  return (
    <span className={`st-step-badge ${cls}`}>
      {status === 'done' ? '✓' : step}
    </span>
  )
}

function getLineText(line) {
  return (line.segments || []).map(s => s.text).join('')
}

function getLineFontSize(line) {
  const segs = line.segments || []
  if (segs.length === 0) return 16
  return Math.max(...segs.map(s => s.font_size || 16))
}

function getSelectionInLine(lineEl) {
  const sel = window.getSelection()
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null
  const range = sel.getRangeAt(0)
  if (!lineEl.contains(range.startContainer) || !lineEl.contains(range.endContainer)) return null
  let charStart = 0, charEnd = 0, found = false
  const walk = (node, countOnly) => {
    if (node.nodeType === 3) {
      const len = node.textContent.length
      if (!countOnly && node === range.startContainer) charStart = charStart + range.startOffset
      else if (!found) charStart += len
      if (!countOnly && node === range.startContainer) found = true
      if (node === range.endContainer) { charEnd = charStart + range.endOffset - (found && node === range.startContainer ? range.startOffset : 0); return true }
      if (found) charEnd += len
      return false
    }
    for (const c of node.childNodes) { if (walk(c, countOnly)) return true }
    return false
  }
  charStart = 0; charEnd = 0; found = false
  let offset = 0
  const walker = document.createTreeWalker(lineEl, NodeFilter.SHOW_TEXT)
  let textNode = walker.nextNode()
  let startOff = 0, endOff = 0
  while (textNode) {
    const len = textNode.textContent.length
    if (textNode === range.startContainer) startOff = offset + range.startOffset
    if (textNode === range.endContainer) { endOff = offset + range.endOffset; break }
    offset += len
    textNode = walker.nextNode()
  }
  if (startOff >= endOff) return null
  return { charStart: startOff, charEnd: endOff }
}

function parseLineSegments(lineEl, baseFontSize) {
  const segments = []
  for (const child of lineEl.childNodes) {
    if (child.nodeType === 3) {
      if (child.textContent) segments.push({ text: child.textContent, _plain: true })
    } else if (child.nodeType === 1) {
      const style = child.style
      segments.push({
        text: child.textContent || '',
        font_family: style.fontFamily || undefined,
        font_size: style.fontSize ? parseFloat(style.fontSize) : undefined,
        font_weight: style.fontWeight || undefined,
        fill: style.color || undefined,
      })
    }
  }
  return segments
}

function SvgEditorCanvas({ svgData, path, node, onGenerate, onSelectionChange }) {
  const { width, height, viewBox, lines } = svgData
  const containerRef = useRef(null)
  const lineRefs = useRef([])
  const [scale, setScale] = useState(1)

  const vbParts = (viewBox || `0 0 ${width} ${height}`).split(/\s+/).map(Number)
  const vbW = vbParts[2], vbH = vbParts[3]

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = entries[0].contentRect.width
      if (w > 0) setScale(w / vbW)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [vbW])

  const containerH = vbH * scale

  const handleLineBlur = (index, e) => {
    const el = e.currentTarget
    const oldText = getLineText(lines[index])
    const newText = el.textContent || ''
    if (newText !== oldText) {
      const existingSegs = lines[index].segments || []
      const baseSeg = existingSegs[0] || {}
      onGenerate?.(path, node, 'editSvgLine', {
        index,
        segments: [{
          font_family: baseSeg.font_family || 'Sans-Serif',
          font_size: baseSeg.font_size || 16,
          font_weight: baseSeg.font_weight || 'normal',
          fill: baseSeg.fill || '#000000',
          text: newText,
        }],
      })
    }
  }

  const handleSelectionChange = () => {
    for (let i = 0; i < lineRefs.current.length; i++) {
      const el = lineRefs.current[i]
      if (!el) continue
      const info = getSelectionInLine(el)
      if (info) {
        onSelectionChange?.({ lineIndex: i, ...info })
        return
      }
    }
    onSelectionChange?.(null)
  }

  useEffect(() => {
    document.addEventListener('selectionchange', handleSelectionChange)
    return () => document.removeEventListener('selectionchange', handleSelectionChange)
  }, [lines])

  const getLeftStyle = (l) => {
    if (l.text_anchor === 'middle') return { left: l.x * scale, transform: 'translateX(-50%)' }
    if (l.text_anchor === 'end') return { right: (vbW - l.x) * scale }
    return { left: l.x * scale }
  }

  const lineKey = (line, i) => {
    const segsKey = (line.segments || []).map(s =>
      `${s.text}|${s.fill}|${s.font_weight}|${s.font_family}|${s.font_size}`
    ).join('/')
    return `${i}-${segsKey}`
  }

  return (
    <div
      ref={containerRef}
      className="st-svg-editor-container"
      style={{ minHeight: containerH || 'auto' }}
    >
      <svg
        className="st-svg-editor-bg"
        width="100%"
        viewBox={viewBox || `0 0 ${width} ${height}`}
        xmlns="http://www.w3.org/2000/svg"
        style={{ pointerEvents: 'none' }}
      >
        <rect width={vbW} height={vbH} fill="#f8f9fa" rx="4" />
      </svg>
      {scale > 0 && lines.map((l, i) => {
        const fontSize = getLineFontSize(l)
        return (
          <div
            key={lineKey(l, i)}
            ref={el => lineRefs.current[i] = el}
            contentEditable
            suppressContentEditableWarning
            className="st-svg-editable-line"
            style={{
              position: 'absolute',
              top: (l.y - fontSize * 0.82) * scale,
              ...getLeftStyle(l),
              opacity: l.opacity,
              fontSize: fontSize * scale,
            }}
            onBlur={e => handleLineBlur(i, e)}
          >
            {(l.segments || []).map((seg, j) => (
              <span
                key={j}
                style={{
                  fontFamily: seg.font_family,
                  fontWeight: seg.font_weight,
                  color: seg.fill,
                  fontSize: seg.font_size !== fontSize ? `${seg.font_size * scale}px` : undefined,
                }}
              >{seg.text}</span>
            ))}
          </div>
        )
      })}
    </div>
  )
}

const FONT_OPTIONS = [
  'Arial', 'Arial Black', 'Calibri', 'Cambria', 'Comic Sans MS',
  'Courier New', 'Georgia', 'Helvetica', 'Impact', 'Lucida Sans',
  'Montserrat', 'Open Sans', 'Oswald', 'Palatino', 'Roboto',
  'Roboto Condensed', 'Roboto Slab', 'Sans-Serif', 'Sans-Serif Bold',
  'Sans-Serif Extra Bold', 'Rounded Sans-Serif Bold',
  'Tahoma', 'Times New Roman', 'Trebuchet MS', 'Verdana',
]

const _fontHistory = []
const MAX_FONT_HISTORY = 6

function addFontHistory(font) {
  const idx = _fontHistory.indexOf(font)
  if (idx !== -1) _fontHistory.splice(idx, 1)
  _fontHistory.unshift(font)
  if (_fontHistory.length > MAX_FONT_HISTORY) _fontHistory.length = MAX_FONT_HISTORY
}

function FontCombobox({ value, onChange }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const wrapRef = useRef(null)
  const triggerRef = useRef(null)
  const searchRef = useRef(null)
  const [listPos, setListPos] = useState({ top: 0, left: 0, width: 240 })

  useEffect(() => {
    const handler = e => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const openDropdown = () => {
    setOpen(true)
    setQuery('')
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setListPos({ top: rect.bottom + 2, left: rect.left, width: Math.max(rect.width, 240) })
    }
    setTimeout(() => searchRef.current?.focus(), 0)
  }

  const select = (f) => {
    addFontHistory(f)
    onChange(f)
    setOpen(false)
  }

  const lq = query.toLowerCase()
  const recentFonts = _fontHistory.filter(f => f !== value && (!lq || f.toLowerCase().includes(lq)))
  const allFonts = FONT_OPTIONS.filter(f =>
    f !== value && !_fontHistory.includes(f) && (!lq || f.toLowerCase().includes(lq))
  )

  return (
    <div className="st-font-combo" ref={wrapRef}>
      <button
        ref={triggerRef}
        className="st-font-combo-trigger"
        style={{ fontFamily: value }}
        onClick={openDropdown}
        type="button"
      >
        {value} <span className="st-font-combo-arrow">▾</span>
      </button>
      {open && (
        <div className="st-font-combo-panel" style={{ position: 'fixed', top: listPos.top, left: listPos.left, width: listPos.width }}>
          <div className="st-font-combo-search-wrap">
            <input
              ref={searchRef}
              type="text"
              className="st-font-combo-search"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search font…"
            />
          </div>
          <ul className="st-font-combo-list">
            <li className="st-font-combo-section st-font-combo-section-current">Current</li>
            <li
              className="st-font-combo-item st-font-combo-current"
              style={{ fontFamily: value }}
              onMouseDown={e => { e.preventDefault(); select(value) }}
            >{value}</li>
            {recentFonts.length > 0 && (
              <>
                <li className="st-font-combo-section st-font-combo-section-recent">Recent</li>
                {recentFonts.map(f => (
                  <li key={f} className="st-font-combo-item" style={{ fontFamily: f }}
                    onMouseDown={e => { e.preventDefault(); select(f) }}>{f}</li>
                ))}
              </>
            )}
            <li className="st-font-combo-section st-font-combo-section-all">All Fonts</li>
            {allFonts.length > 0 ? allFonts.map(f => (
              <li key={f} className="st-font-combo-item" style={{ fontFamily: f }}
                onMouseDown={e => { e.preventDefault(); select(f) }}>{f}</li>
            )) : (
              <li className="st-font-combo-item st-font-combo-empty">No match</li>
            )}
          </ul>
        </div>
      )}
    </div>
  )
}

function getStyleAtSelection(svgData, selection) {
  if (!selection) {
    const seg = svgData.lines[0]?.segments?.[0]
    return seg || { font_family: 'Sans-Serif', font_weight: 'normal', font_size: 16, fill: '#000000' }
  }
  const line = svgData.lines[selection.lineIndex]
  if (!line) return { font_family: 'Sans-Serif', font_weight: 'normal', font_size: 16, fill: '#000000' }
  let offset = 0
  for (const seg of (line.segments || [])) {
    const segEnd = offset + seg.text.length
    if (selection.charStart >= offset && selection.charStart < segEnd) return seg
    offset = segEnd
  }
  return line.segments?.[0] || { font_family: 'Sans-Serif', font_weight: 'normal', font_size: 16, fill: '#000000' }
}

function SvgStyleControls({ svgData, path, node, onGenerate, selection }) {
  const selRef = useRef(null)
  const [, forceUpdate] = useState(0)
  if (selection && selection.charStart < selection.charEnd) {
    selRef.current = selection
  }
  const activeSel = selRef.current
  const style = getStyleAtSelection(svgData, activeSel)

  const edit = (field, value) => {
    const sel = selRef.current
    if (sel) {
      onGenerate?.(path, node, 'editSegmentStyle', {
        lineIndex: sel.lineIndex,
        charStart: sel.charStart,
        charEnd: sel.charEnd,
        field,
        value,
      })
    } else {
      onGenerate?.(path, node, 'editStyle', { field, value })
    }
  }

  const clearSel = () => { selRef.current = null; forceUpdate(n => n + 1) }

  return (
    <div className="st-style-editor">
      <label className="st-style-field">
        <span>Font</span>
        <FontCombobox value={style.font_family} onChange={v => edit('font_family', v)} />
      </label>
      <label className="st-style-field">
        <span>Weight</span>
        <select value={style.font_weight} onChange={e => edit('font_weight', e.target.value)}>
          <option value="normal">Normal</option>
          <option value="bold">Bold</option>
          <option value="300">300</option>
          <option value="400">400</option>
          <option value="500">500</option>
          <option value="600">600</option>
          <option value="700">700</option>
          <option value="800">800</option>
          <option value="900">900</option>
        </select>
      </label>
      <label className="st-style-field">
        <span>Size</span>
        <input type="number" min={8} max={200} value={style.font_size}
          onChange={e => edit('font_size', parseFloat(e.target.value) || 16)} />
      </label>
      <label className="st-style-field">
        <span>Color</span>
        <input type="color" value={style.fill} onChange={e => edit('fill', e.target.value)} />
      </label>
      {activeSel && (
        <span className="st-style-selection-badge" onClick={clearSel} title="Click to clear selection">
          Selection ✕
        </span>
      )}
    </div>
  )
}

function SvgEditorModal({ svgData, path, node, onGenerate, onClose }) {
  const [selection, setSelection] = useState(null)

  useEffect(() => {
    const handleKey = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div className="st-svg-modal-overlay" onClick={onClose}>
      <div className="st-svg-modal" onClick={e => e.stopPropagation()}>
        <div className="st-svg-modal-header">
          <span className="st-svg-modal-title">Text Style Editor</span>
          <button className="st-svg-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="st-svg-modal-body">
          <SvgEditorCanvas svgData={svgData} path={path} node={node} onGenerate={onGenerate} onSelectionChange={setSelection} />
          <SvgStyleControls svgData={svgData} path={path} node={node} onGenerate={onGenerate} selection={selection} />
        </div>
      </div>
    </div>
  )
}

function SvgTextEditor({ svgData, path, node, onGenerate }) {
  const [expanded, setExpanded] = useState(false)
  if (!svgData?.lines?.length) return null
  const { width, height, viewBox, lines } = svgData
  const vb = viewBox || `0 0 ${width} ${height}`

  return (
    <div className="st-svg-editor-wrap">
      <div className="st-svg-thumb-row">
        <svg
          className="st-svg-thumb"
          width="100%"
          viewBox={vb}
          xmlns="http://www.w3.org/2000/svg"
        >
          <rect width={width} height={height} fill="#f8f9fa" rx="4" />
          {lines.map((l, i) => (
            <text key={i} x={l.x} y={l.y}
              opacity={l.opacity} textAnchor={l.text_anchor}
            >
              {(l.segments || []).map((seg, j) => (
                <tspan key={j}
                  fontFamily={seg.font_family} fontSize={seg.font_size}
                  fontWeight={seg.font_weight} fill={seg.fill}
                >{seg.text}</tspan>
              ))}
            </text>
          ))}
        </svg>
        <button className="st-svg-expand-btn" onClick={() => setExpanded(true)} title="Edit in larger view">
          Edit ↗
        </button>
      </div>
      {expanded && (
        <SvgEditorModal
          svgData={svgData} path={path} node={node}
          onGenerate={onGenerate} onClose={() => setExpanded(false)}
        />
      )}
    </div>
  )
}

function TextGenerateSection({ node, path, genState, onGenerate }) {
  const state = genState?.[path] || {}
  const step1 = state.step1Status || 'idle'
  const step2 = state.step2Status || 'idle'
  const textContent = state.textContent || null
  const svgData = state.svgData || null
  const isStep1Busy = step1 === 'generating'
  const isStep2Busy = step2 === 'generating'
  const busy = isStep1Busy || isStep2Busy

  return (
    <div className="st-gen-section">
      {/* Step 1: Content Generation */}
      <div className="st-gen-step">
        <div className="st-gen-step-header">
          <StepBadge step="1" status={step1} />
          <span className="st-gen-step-title">Text Content</span>
          <button
            className={`st-gen-btn st-gen-btn-sm ${isStep1Busy ? 'generating' : ''}`}
            disabled={busy}
            onClick={() => onGenerate?.(path, node, 'step1')}
          >
            {isStep1Busy ? (
              <><span className="st-gen-spinner" />Generating…</>
            ) : step1 === 'done' ? (
              'Re-generate'
            ) : (
              'Generate'
            )}
          </button>
        </div>
        {isStep1Busy && (
          <div className="st-gen-result st-gen-result-placeholder">
            <div className="st-gen-placeholder-shimmer" />
          </div>
        )}
        {step1 === 'done' && textContent != null && (
          <textarea
            className="st-gen-textarea"
            value={textContent}
            rows={Math.max(2, textContent.split('\n').length)}
            onChange={e => onGenerate?.(path, node, 'editText', e.target.value)}
          />
        )}
      </div>

      {/* Step 2: Text Rendering */}
      <div className="st-gen-step">
        <div className="st-gen-step-header">
          <StepBadge step="2" status={step2} />
          <span className="st-gen-step-title">Rendered SVG</span>
          <button
            className={`st-gen-btn st-gen-btn-sm ${isStep2Busy ? 'generating' : ''}`}
            disabled={busy || step1 !== 'done'}
            onClick={() => onGenerate?.(path, node, 'step2')}
          >
            {isStep2Busy ? (
              <><span className="st-gen-spinner" />Rendering…</>
            ) : step2 === 'done' ? (
              'Re-render'
            ) : (
              'Render'
            )}
          </button>
        </div>
        {isStep2Busy && (
          <div className="st-gen-result st-gen-result-placeholder">
            <div className="st-gen-placeholder-shimmer" />
          </div>
        )}
        {step2 === 'done' && state.pngUrl && (
          <>
            <img
              className="st-gen-result-img"
              src={state.pngUrl}
              alt="Rendered text"
              style={{ marginTop: 6, maxHeight: 120, objectFit: 'contain' }}
            />
            <div className="st-align-bar">
              <span className="st-align-label">Align:</span>
              {['left', 'center', 'right'].map(a => (
                <button
                  key={a}
                  className={`st-align-btn ${(state.alignment || node.alignment || 'left') === a ? 'active' : ''}`}
                  onClick={() => onGenerate?.(path, node, 'changeAlignment', a)}
                  disabled={state.alignBusy}
                >
                  {a === 'left' ? 'L' : a === 'center' ? 'C' : 'R'}
                </button>
              ))}
              {state.alignBusy && <span className="st-gen-spinner" />}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function isDecorativeBar(node) {
  const b = node.bbox || {}
  const w = b.width || 1, h = b.height || 1
  return (h / w > 4 || w / h > 4)
}

function ImageGenModal({ node, path, genState, onGenerate, onClose }) {
  const state = genState?.[path] || {}
  const step1 = state.step1Status || 'idle'
  const step2 = state.step2Status || 'idle'
  const imagePrompt = state.imagePrompt || null
  const imageUrl = state.imageUrl || null
  const selectedModel = state.selectedModel || 'gpt-image-1.5'
  const isStep1Busy = step1 === 'generating'
  const isStep2Busy = step2 === 'generating'
  const busy = isStep1Busy || isStep2Busy
  const [enlarged, setEnlarged] = useState(false)

  useEffect(() => {
    const handleKey = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div className="st-imggen-modal-overlay" onClick={onClose}>
      <div className="st-imggen-modal" onClick={e => e.stopPropagation()}>
        <div className="st-imggen-modal-header">
          <span className="st-imggen-modal-title">Image Generation</span>
          <button className="st-imggen-modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Node info summary */}
        <div className="st-imggen-info">
          {node.role && <div className="st-imggen-info-row"><b>Role</b> {node.role}</div>}
          {node.purpose && <div className="st-imggen-info-row"><b>Purpose</b> {node.purpose}</div>}
          {node.content_requirements && (
            <div className="st-imggen-info-row"><b>Requirements</b> {node.content_requirements}</div>
          )}
        </div>

        <div className="st-imggen-body">
          {/* Step 1: Design Prompt */}
          <div className="st-gen-step">
            <div className="st-gen-step-header">
              <StepBadge step="1" status={step1} />
              <span className="st-gen-step-title">Image Prompt</span>
              <button
                className={`st-gen-btn st-gen-btn-sm ${isStep1Busy ? 'generating' : ''}`}
                disabled={busy}
                onClick={() => onGenerate?.(path, node, 'step1')}
              >
                {isStep1Busy ? (
                  <><span className="st-gen-spinner" />Designing…</>
                ) : step1 === 'done' ? 'Re-design' : 'Design Prompt'}
              </button>
            </div>
            {isStep1Busy && (
              <div className="st-gen-result st-gen-result-placeholder">
                <div className="st-gen-placeholder-shimmer" />
              </div>
            )}
            {step1 === 'done' && imagePrompt != null && (
              <textarea
                className="st-gen-textarea st-gen-textarea-prompt"
                value={imagePrompt}
                rows={Math.max(4, Math.ceil(imagePrompt.length / 80) + imagePrompt.split('\n').length)}
                onChange={e => onGenerate?.(path, node, 'editPrompt', e.target.value)}
              />
            )}
          </div>

          {/* Step 2: Generate Image */}
          <div className="st-gen-step">
            <div className="st-gen-step-header">
              <StepBadge step="2" status={step2} />
              <span className="st-gen-step-title">Generated Image</span>
              <button
                className={`st-gen-btn st-gen-btn-sm ${isStep2Busy ? 'generating' : ''}`}
                disabled={busy || step1 !== 'done'}
                onClick={() => onGenerate?.(path, node, 'step2')}
              >
                {isStep2Busy ? (
                  <><span className="st-gen-spinner" />Generating…</>
                ) : step2 === 'done' ? 'Re-generate' : 'Generate'}
              </button>
            </div>
            {isStep2Busy && (
              <div className="st-gen-result st-gen-result-placeholder">
                <div className="st-gen-placeholder-shimmer" />
              </div>
            )}
            {step2 === 'done' && imageUrl && (
              <div className="st-gen-image-result">
                <img
                  className="st-gen-result-img"
                  src={imageUrl}
                  alt="Generated"
                  onClick={() => setEnlarged(true)}
                  title="Click to enlarge"
                />
              </div>
            )}
          </div>
        </div>

        {enlarged && imageUrl && (
          <div className="st-gen-lightbox" onClick={() => setEnlarged(false)}>
            <div className="st-gen-lightbox-inner" onClick={e => e.stopPropagation()}>
              <img src={imageUrl} alt="Generated (enlarged)" />
              <button className="st-gen-lightbox-close" onClick={() => setEnlarged(false)}>✕</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ImageGenerateSection({ node, path, genState, onGenerate }) {
  const state = genState?.[path] || {}
  const step1 = state.step1Status || 'idle'
  const step2 = state.step2Status || 'idle'
  const imageUrl = state.imageUrl || null
  const busy = step1 === 'generating' || step2 === 'generating'
  const [modalOpen, setModalOpen] = useState(false)
  const isBar = isDecorativeBar(node)
  const barBusy = state.barGenerating

  if (isBar) {
    return (
      <div className="st-gen-section">
        <div className="st-gen-actions" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className="st-gen-btn"
            disabled={barBusy}
            onClick={() => onGenerate?.(path, node, 'generateBar')}
          >
            {barBusy ? 'Generating…' : step2 === 'done' ? 'Re-generate Color Block' : 'Generate Color Block'}
          </button>
          {barBusy && <span className="st-gen-spinner" />}
          {step2 === 'done' && <span className="st-gen-status-hint">Done</span>}
        </div>
        {step2 === 'done' && imageUrl && (
          <img
            className="st-gen-result-img"
            src={imageUrl}
            alt="Decorative bar"
            style={{ marginTop: 6, maxHeight: 120, objectFit: 'contain' }}
          />
        )}
      </div>
    )
  }

  const statusText = busy
    ? (step1 === 'generating' ? 'Designing prompt…' : 'Generating image…')
    : step2 === 'done' ? 'Done' : step1 === 'done' ? 'Prompt ready' : null

  return (
    <div className="st-gen-section">
      <div className="st-gen-actions" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          className="st-gen-btn"
          onClick={() => setModalOpen(true)}
        >
          Generate Image ↗
        </button>
        {busy && <span className="st-gen-spinner" />}
        {statusText && <span className="st-gen-status-hint">{statusText}</span>}
      </div>
      {step2 === 'done' && imageUrl && (
        <img
          className="st-gen-result-img"
          src={imageUrl}
          alt="Generated"
          style={{ marginTop: 6, maxHeight: 120, objectFit: 'contain', cursor: 'pointer' }}
          onClick={() => setModalOpen(true)}
        />
      )}
      {modalOpen && (
        <ImageGenModal
          node={node} path={path} genState={genState}
          onGenerate={onGenerate} onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  )
}

function ChartGenModal({ node, path, genState, onGenerate, onClose }) {
  const state = genState?.[path] || {}
  const step1 = state.step1Status || 'idle'
  const chartTemplates = state.chartTemplates || []
  const selectedChartType = state.selectedChartType || null
  const variationPreviews = state.variationPreviews || {}
  const selectedTemplate = state.selectedTemplate || null
  const chartUrl = state.chartUrl || null
  const isStep1Busy = step1 === 'generating'
  const anyRendering = Object.values(variationPreviews).some(v => v.status === 'generating')
  const [enlarged, setEnlarged] = useState(null)

  const chartType = node.chart_type || null

  const groupedByType = useMemo(() => {
    const map = {}
    chartTemplates.forEach(t => {
      const ct = t.chart_type || 'unknown'
      if (!map[ct]) map[ct] = []
      map[ct].push(t)
    })
    return map
  }, [chartTemplates])

  const chartTypes = useMemo(() => Object.keys(groupedByType), [groupedByType])
  const currentVariations = selectedChartType ? (groupedByType[selectedChartType] || []) : []

  const handleSelectChartType = (ct) => {
    onGenerate?.(path, node, 'renderVariations', ct)
  }

  useEffect(() => {
    if (step1 === 'idle') {
      onGenerate?.(path, node, 'step1')
    }
  }, [])

  useEffect(() => {
    const handleKey = e => { if (e.key === 'Escape') { if (enlarged) setEnlarged(null); else onClose() } }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose, enlarged])

  return (
    <div className="st-imggen-modal-overlay st-chartgen-modal-overlay" onClick={onClose}>
      <div className="st-chartgen-modal" onClick={e => e.stopPropagation()}>
        <div className="st-imggen-modal-header">
          <span className="st-imggen-modal-title">Chart Generation</span>
          <button className="st-imggen-modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="st-imggen-info">
          {chartType && <div className="st-imggen-info-row"><b>Chart Type</b> {chartType}</div>}
          {node.content && <div className="st-imggen-info-row"><b>Description</b> {node.content}</div>}
        </div>

        <div className="st-imggen-body">
          {/* Step 1: Find Templates */}
          <div className="st-gen-step">
            <div className="st-gen-step-header">
              <StepBadge step="1" status={step1} />
              <span className="st-gen-step-title">Find Compatible Templates</span>
              <button
                className={`st-gen-btn st-gen-btn-sm ${isStep1Busy ? 'generating' : ''}`}
                disabled={isStep1Busy}
                onClick={() => onGenerate?.(path, node, 'step1')}
              >
                {isStep1Busy ? (
                  <><span className="st-gen-spinner" />Searching…</>
                ) : step1 === 'done' ? 'Re-search' : 'Find Templates'}
              </button>
            </div>
            {isStep1Busy && (
              <div className="st-gen-result st-gen-result-placeholder">
                <div className="st-gen-placeholder-shimmer" />
              </div>
            )}
            {step1 === 'done' && chartTemplates.length === 0 && (
              <div className="st-chart-no-templates">No compatible templates found.</div>
            )}
          </div>

          {/* Chart Type Grid */}
          {step1 === 'done' && chartTypes.length > 0 && (
            <div className="st-gen-step">
              <div className="st-gen-step-header">
                <StepBadge step="2" status={selectedChartType ? 'done' : 'idle'} />
                <span className="st-gen-step-title">Select Chart Type ({chartTypes.length} types, {chartTemplates.length} templates)</span>
              </div>
              <div className="st-chart-type-grid">
                {chartTypes.map(ct => (
                  <div
                    key={ct}
                    className={`st-chart-type-card ${selectedChartType === ct ? 'active' : ''}`}
                    onClick={() => handleSelectChartType(ct)}
                  >
                    <img
                      className="st-chart-type-img"
                      src={`/api/chart-type-image/${encodeURIComponent(ct)}`}
                      alt={ct}
                      onError={e => { e.target.style.display = 'none' }}
                    />
                    <div className="st-chart-type-label">{ct}</div>
                    <div className="st-chart-type-badge">{groupedByType[ct].length}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Variation Preview Grid */}
          {selectedChartType && currentVariations.length > 0 && (
            <div className="st-gen-step">
              <div className="st-gen-step-header">
                <StepBadge step="3" status={selectedTemplate ? 'done' : (anyRendering ? 'generating' : 'idle')} />
                <span className="st-gen-step-title">
                  Variations for "{selectedChartType}" ({currentVariations.length})
                </span>
                {anyRendering && <span className="st-gen-spinner" style={{ marginLeft: 6 }} />}
              </div>
              <div className="st-chart-var-grid">
                {currentVariations.map(tpl => {
                  const vp = variationPreviews[tpl.name]
                  const vpStatus = vp?.status || 'generating'
                  const vpUrl = vp?.chartUrl || null
                  const isSelected = selectedTemplate === tpl.name
                  return (
                    <div
                      key={tpl.name}
                      className={`st-chart-var-card ${isSelected ? 'active' : ''} ${vpStatus}`}
                      onClick={() => {
                        if (vpUrl) onGenerate?.(path, node, 'selectVariation', tpl.name)
                      }}
                    >
                      {vpStatus === 'generating' && (
                        <div className="st-chart-var-placeholder">
                          <span className="st-gen-spinner" />
                        </div>
                      )}
                      {vpStatus === 'done' && vpUrl && (
                        <img
                          className="st-chart-var-img"
                          src={vpUrl}
                          alt={tpl.name}
                        />
                      )}
                      {vpStatus === 'error' && (
                        <div className="st-chart-var-placeholder st-chart-var-error">Error</div>
                      )}
                      <div className="st-chart-var-label">{tpl.name}</div>
                      {isSelected && <div className="st-chart-var-check">✓</div>}
                      {vpStatus === 'done' && vpUrl && (
                        <button
                          className="st-chart-var-enlarge"
                          onClick={e => { e.stopPropagation(); setEnlarged(vpUrl) }}
                          title="Enlarge"
                        >⤢</button>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {enlarged && (
          <div className="st-gen-lightbox" onClick={() => setEnlarged(null)}>
            <div className="st-gen-lightbox-inner" onClick={e => e.stopPropagation()}>
              <img src={enlarged} alt="Enlarged chart" />
              <button className="st-gen-lightbox-close" onClick={() => setEnlarged(null)}>✕</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ChartGenerateSection({ node, path, genState, onGenerate }) {
  const state = genState?.[path] || {}
  const step1 = state.step1Status || 'idle'
  const chartUrl = state.chartUrl || null
  const variationPreviews = state.variationPreviews || {}
  const anyRendering = Object.values(variationPreviews).some(v => v.status === 'generating')
  const busy = step1 === 'generating' || anyRendering
  const [modalOpen, setModalOpen] = useState(false)

  const statusText = busy
    ? (step1 === 'generating' ? 'Searching templates…' : 'Rendering variations…')
    : chartUrl ? 'Selected' : step1 === 'done' ? 'Templates ready' : null

  return (
    <div className="st-gen-section">
      <div className="st-gen-actions" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          className="st-gen-btn"
          onClick={() => setModalOpen(true)}
        >
          Generate Chart ↗
        </button>
        {busy && <span className="st-gen-spinner" />}
        {statusText && <span className="st-gen-status-hint">{statusText}</span>}
      </div>
      {chartUrl && (
        <img
          className="st-gen-result-img"
          src={chartUrl}
          alt="Generated chart"
          style={{ marginTop: 6, maxHeight: 120, objectFit: 'contain', cursor: 'pointer' }}
          onClick={() => setModalOpen(true)}
        />
      )}
      {modalOpen && (
        <ChartGenModal
          node={node} path={path} genState={genState}
          onGenerate={onGenerate} onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  )
}

function TextNodeDetail({ node, path, genState, onGenerate }) {
  return (
    <div className="st-detail">
      {node.role && (
        <div className="st-detail-row">
          <span className="st-detail-label">Role</span>
          <span className="st-detail-value">{node.role}</span>
        </div>
      )}
      <div className="st-detail-row">
        <span className="st-detail-label">Font</span>
        <span className="st-detail-value">
          {node.font_family || '—'}
          {` · ${node.font_weight || 'normal'}`}
          {node.font_style && node.font_style !== 'normal' ? ` · ${node.font_style}` : ''}
          {node.font_size_px ? ` · ${node.font_size_px}px` : ''}
        </span>
      </div>
      {node.color && (
        <div className="st-detail-row">
          <span className="st-detail-label">Color</span>
          <span className="st-detail-value st-detail-color">
            <span className="st-detail-swatch" style={{ background: node.color }} />
            {node.color}
          </span>
        </div>
      )}
      {node.text_transform && node.text_transform !== 'none' && (
        <div className="st-detail-row">
          <span className="st-detail-label">Transform</span>
          <span className="st-detail-value">{node.text_transform}</span>
        </div>
      )}
      {node.letter_spacing != null && node.letter_spacing !== 0 && (
        <div className="st-detail-row">
          <span className="st-detail-label">Spacing</span>
          <span className="st-detail-value">{node.letter_spacing}px</span>
        </div>
      )}
      {node.line_count != null && (
        <div className="st-detail-row">
          <span className="st-detail-label">Lines</span>
          <span className="st-detail-value">{node.line_count}</span>
        </div>
      )}
      {node.content_requirements && (
        <div className="st-detail-row st-detail-row-block">
          <span className="st-detail-label">Requirements</span>
          <span className="st-detail-value st-detail-long">{node.content_requirements}</span>
        </div>
      )}
      <TextGenerateSection node={node} path={path} genState={genState} onGenerate={onGenerate} />
    </div>
  )
}

function ImageNodeDetail({ node, path, genState, onGenerate }) {
  return (
    <div className="st-detail">
      {node.role && (
        <div className="st-detail-row">
          <span className="st-detail-label">Role</span>
          <span className="st-detail-value">{node.role}</span>
        </div>
      )}
      {node.purpose && (
        <div className="st-detail-row">
          <span className="st-detail-label">Purpose</span>
          <span className="st-detail-value st-detail-ellipsis">{node.purpose}</span>
        </div>
      )}
      <ImageGenerateSection node={node} path={path} genState={genState} onGenerate={onGenerate} />
    </div>
  )
}


function ChartNodeDetail({ node, path, genState, onGenerate }) {
  const chartType = node.chart_type || null
  return (
    <div className="st-detail">
      {chartType && (
        <div className="st-detail-row">
          <span className="st-detail-label">Chart Type</span>
          <span className="st-detail-value">{chartType}</span>
        </div>
      )}
      {node.content && (
        <div className="st-detail-row">
          <span className="st-detail-label">Description</span>
          <span className="st-detail-value st-detail-ellipsis">{node.content}</span>
        </div>
      )}
      <ChartGenerateSection node={node} path={path} genState={genState} onGenerate={onGenerate} />
    </div>
  )
}

function childLabel(index, node) {
  const child = node.children?.[index]
  if (!child) return `[${index}]`
  const t = child.type?.toUpperCase() || '?'
  return `${t}[${index}]`
}

function OldConstraintValue({ label, children }) {
  return (
    <div className="st-cst-row st-cst-row-old">
      <span className="st-cst-key">{label}</span>
      {children}
    </div>
  )
}

function ConstraintsCard({ node, depth, diffMap, path }) {
  const [open, setOpen] = useState(false)
  const c = node.constraints
  const cstDiffs = diffMap?.[path]?.constraintsChanged || {}
  const hasAnyDiff = Object.keys(cstDiffs).length > 0

  if (!c && !hasAnyDiff) return null

  const { alignment, gap, padding, relative_size, orientation, overlap } = c || {}

  const hasPadding = padding && (
    (padding.horizontal?.left || 0) + (padding.horizontal?.right || 0) +
    (padding.vertical?.top || 0) + (padding.vertical?.bottom || 0) > 0
  )

  const constraintCount =
    (alignment ? 1 : 0) + (gap ? 1 : 0) + (hasPadding ? 1 : 0) +
    (relative_size?.length || 0) + (orientation?.length || 0) + (overlap?.length || 0)

  const removedKeys = Object.keys(cstDiffs).filter(k => {
    const fromVal = cstDiffs[k]?.from
    const toVal = cstDiffs[k]?.to
    return fromVal && !toVal
  })

  return (
    <div className={`st-constraints-card ${hasAnyDiff ? 'st-cst-changed' : ''}`} style={{ marginLeft: `${12 + (depth + 1) * 16}px` }}>
      <div className="st-constraints-header" onClick={() => setOpen(!open)}>
        <span className="st-constraints-toggle">{open ? '▾' : '▸'}</span>
        <span className="st-constraints-badge">CONSTRAINTS</span>
        <span className="st-constraints-count">{constraintCount}</span>
        {hasAnyDiff && <span className="st-diff-badge" style={{ marginLeft: 6 }}>MODIFIED</span>}
      </div>
      {open && (
        <div className="st-constraints-body">
          {alignment && (
            <>
              {cstDiffs.alignment?.from && (
                <OldConstraintValue label="Alignment">
                  <span className="st-cst-val">{cstDiffs.alignment.from.value} ({cstDiffs.alignment.from.direction})</span>
                </OldConstraintValue>
              )}
              <div className={`st-cst-row ${cstDiffs.alignment ? 'st-cst-row-changed' : ''}`}>
                <span className="st-cst-key">Alignment</span>
                <span className="st-cst-val">{alignment.value} ({alignment.direction})</span>
              </div>
            </>
          )}

          {gap && (
            <>
              {cstDiffs.gap?.from && (
                <OldConstraintValue label="Gap">
                  <span className="st-cst-val">{Math.round(cstDiffs.gap.from.value)}px ({cstDiffs.gap.from.direction})</span>
                </OldConstraintValue>
              )}
              <div className={`st-cst-row ${cstDiffs.gap ? 'st-cst-row-changed' : ''}`}>
                <span className="st-cst-key">Gap</span>
                <span className="st-cst-val">{Math.round(gap.value)}px ({gap.direction})</span>
              </div>
            </>
          )}

          {hasPadding && (
            <div className={`st-cst-row ${cstDiffs.padding ? 'st-cst-row-changed' : ''}`}>
              <span className="st-cst-key">Padding</span>
              <span className="st-cst-val">
                T:{padding.vertical?.top || 0}  R:{padding.horizontal?.right || 0}  B:{padding.vertical?.bottom || 0}  L:{padding.horizontal?.left || 0}
              </span>
            </div>
          )}

          {relative_size && relative_size.length > 0 && (
            <>
              {cstDiffs.relative_size?.from && Array.isArray(cstDiffs.relative_size.from) && (
                <div className="st-cst-group st-cst-group-old">
                  <div className="st-cst-key">Size Relations</div>
                  {cstDiffs.relative_size.from.map((r, i) => {
                    const dim = r.type === 'relative_height' ? 'height' : 'width'
                    const op = r.operator === 'greater_than' ? '>' : r.operator === 'less_than' ? '<' : '='
                    return (
                      <div key={i} className="st-cst-relation">
                        <span className="st-cst-child">{childLabel(r.source_index, node)}</span>
                        <span className="st-cst-dim">{dim}</span>
                        <span className="st-cst-op">{op}</span>
                        <span className="st-cst-ratio">{r.ratio.toFixed(2)}x</span>
                        <span className="st-cst-child">{childLabel(r.target_index, node)}</span>
                      </div>
                    )
                  })}
                </div>
              )}
              <div className={`st-cst-group ${cstDiffs.relative_size ? 'st-cst-row-changed' : ''}`}>
                <div className="st-cst-key">Size Relations</div>
                {relative_size.map((r, i) => {
                  const dim = r.type === 'relative_height' ? 'height' : 'width'
                  const op = r.operator === 'greater_than' ? '>' : r.operator === 'less_than' ? '<' : '='
                  return (
                    <div key={i} className="st-cst-relation">
                      <span className="st-cst-child">{childLabel(r.source_index, node)}</span>
                      <span className="st-cst-dim">{dim}</span>
                      <span className="st-cst-op">{op}</span>
                      <span className="st-cst-ratio">{r.ratio.toFixed(2)}x</span>
                      <span className="st-cst-child">{childLabel(r.target_index, node)}</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {orientation && orientation.length > 0 && (
            <>
              {cstDiffs.orientation?.from && Array.isArray(cstDiffs.orientation.from) && (
                <div className="st-cst-group st-cst-group-old">
                  <div className="st-cst-key">Orientation</div>
                  {cstDiffs.orientation.from.map((o, i) => (
                    <div key={i} className="st-cst-relation">
                      <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                      <span className="st-cst-dim">at</span>
                      <span className="st-cst-val">{o.position}</span>
                      <span className="st-cst-dim">of</span>
                      <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className={`st-cst-group ${cstDiffs.orientation ? 'st-cst-row-changed' : ''}`}>
                <div className="st-cst-key">Orientation</div>
                {orientation.map((o, i) => (
                  <div key={i} className="st-cst-relation">
                    <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                    <span className="st-cst-dim">at</span>
                    <span className="st-cst-val">{o.position}</span>
                    <span className="st-cst-dim">of</span>
                    <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {overlap && overlap.length > 0 && (
            <>
              {cstDiffs.overlap?.from && Array.isArray(cstDiffs.overlap.from) && (
                <div className="st-cst-group st-cst-group-old">
                  <div className="st-cst-key">Overlap</div>
                  {cstDiffs.overlap.from.map((o, i) => {
                    const label = o.type === 'non_overlap' ? 'non-overlap'
                      : o.type === 'fully_overlap' ? 'fully overlapping' : 'partially overlapping'
                    return (
                      <div key={i} className="st-cst-relation">
                        <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                        <span className="st-cst-val">{label}</span>
                        <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                      </div>
                    )
                  })}
                </div>
              )}
              <div className={`st-cst-group ${cstDiffs.overlap ? 'st-cst-row-changed' : ''}`}>
                <div className="st-cst-key">Overlap</div>
                {overlap.map((o, i) => {
                  const label = o.type === 'non_overlap' ? 'non-overlap'
                    : o.type === 'fully_overlap' ? 'fully overlapping' : 'partially overlapping'
                  return (
                    <div key={i} className="st-cst-relation">
                      <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                      <span className="st-cst-val">{label}</span>
                      <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {removedKeys.map(key => {
            const fromVal = cstDiffs[key].from
            if (key === 'relative_size' && Array.isArray(fromVal)) {
              return (
                <div key={key} className="st-cst-group st-cst-group-old st-cst-group-removed">
                  <div className="st-cst-key">Size Relations <span className="st-cst-removed-tag">REMOVED</span></div>
                  {fromVal.map((r, i) => {
                    const dim = r.type === 'relative_height' ? 'height' : 'width'
                    const op = r.operator === 'greater_than' ? '>' : r.operator === 'less_than' ? '<' : '='
                    return (
                      <div key={i} className="st-cst-relation">
                        <span className="st-cst-child">{childLabel(r.source_index, node)}</span>
                        <span className="st-cst-dim">{dim}</span>
                        <span className="st-cst-op">{op}</span>
                        <span className="st-cst-ratio">{r.ratio.toFixed(2)}x</span>
                        <span className="st-cst-child">{childLabel(r.target_index, node)}</span>
                      </div>
                    )
                  })}
                </div>
              )
            }
            if (key === 'orientation' && Array.isArray(fromVal)) {
              return (
                <div key={key} className="st-cst-group st-cst-group-old st-cst-group-removed">
                  <div className="st-cst-key">Orientation <span className="st-cst-removed-tag">REMOVED</span></div>
                  {fromVal.map((o, i) => (
                    <div key={i} className="st-cst-relation">
                      <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                      <span className="st-cst-dim">at</span>
                      <span className="st-cst-val">{o.position}</span>
                      <span className="st-cst-dim">of</span>
                      <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                    </div>
                  ))}
                </div>
              )
            }
            if (key === 'overlap' && Array.isArray(fromVal)) {
              return (
                <div key={key} className="st-cst-group st-cst-group-old st-cst-group-removed">
                  <div className="st-cst-key">Overlap <span className="st-cst-removed-tag">REMOVED</span></div>
                  {fromVal.map((o, i) => {
                    const label = o.type === 'non_overlap' ? 'non-overlap'
                      : o.type === 'fully_overlap' ? 'fully overlapping' : 'partially overlapping'
                    return (
                      <div key={i} className="st-cst-relation">
                        <span className="st-cst-child">{childLabel(o.source_index, node)}</span>
                        <span className="st-cst-val">{label}</span>
                        <span className="st-cst-child">{childLabel(o.target_index, node)}</span>
                      </div>
                    )
                  })}
                </div>
              )
            }
            if (key === 'alignment' && fromVal) {
              return (
                <OldConstraintValue key={key} label={<>Alignment <span className="st-cst-removed-tag">REMOVED</span></>}>
                  <span className="st-cst-val">{fromVal.value} ({fromVal.direction})</span>
                </OldConstraintValue>
              )
            }
            if (key === 'gap' && fromVal) {
              return (
                <OldConstraintValue key={key} label={<>Gap <span className="st-cst-removed-tag">REMOVED</span></>}>
                  <span className="st-cst-val">{Math.round(fromVal.value)}px ({fromVal.direction})</span>
                </OldConstraintValue>
              )
            }
            return null
          })}
        </div>
      )}
    </div>
  )
}

function DiffSummary({ diff }) {
  if (!diff) return null
  const items = []
  if (diff.alignmentChanged) {
    items.push(`Alignment: ${diff.alignmentChanged.from || 'none'} \u2192 ${diff.alignmentChanged.to || 'none'}`)
  }
  if (diff.childCountChanged) {
    items.push(`Children: ${diff.childCountChanged.from} \u2192 ${diff.childCountChanged.to}`)
  }
  if (diff.constraintsChanged) {
    const fmtAlign = (v) => v ? `${v.value || '?'} (${v.direction || '?'})` : 'removed'
    const fmtGap = (v) => v ? `${Math.round(v.value)}px (${v.direction || '?'})` : 'removed'
    const fmtPad = (v) => {
      if (!v) return 'removed'
      const t = v.vertical?.top || 0, r = v.horizontal?.right || 0
      const b = v.vertical?.bottom || 0, l = v.horizontal?.left || 0
      return `T:${t} R:${r} B:${b} L:${l}`
    }
    for (const [key, val] of Object.entries(diff.constraintsChanged)) {
      if (key === 'alignment') {
        items.push(`Constraint alignment: ${fmtAlign(val.from)} \u2192 ${fmtAlign(val.to)}`)
      } else if (key === 'gap') {
        items.push(`Constraint gap: ${fmtGap(val.from)} \u2192 ${fmtGap(val.to)}`)
      } else if (key === 'padding') {
        items.push(`Padding: ${fmtPad(val.from)} \u2192 ${fmtPad(val.to)}`)
      } else if (key === 'relative_size') {
        const fromLen = Array.isArray(val.from) ? val.from.length : 0
        const toLen = Array.isArray(val.to) ? val.to.length : 0
        items.push(`Size relations: ${fromLen} \u2192 ${toLen}`)
      } else if (key === 'orientation') {
        const fromLen = Array.isArray(val.from) ? val.from.length : 0
        const toLen = Array.isArray(val.to) ? val.to.length : 0
        items.push(`Orientation: ${fromLen} \u2192 ${toLen}`)
      } else if (key === 'overlap') {
        const fromLen = Array.isArray(val.from) ? val.from.length : 0
        const toLen = Array.isArray(val.to) ? val.to.length : 0
        items.push(`Overlap: ${fromLen} \u2192 ${toLen}`)
      } else {
        items.push(`${key} changed`)
      }
    }
  }
  if (diff.addedNode) {
    items.push('New node (added by optimization)')
  }
  if (items.length === 0) return null
  return (
    <div className="st-diff-summary">
      {items.map((item, i) => (
        <div key={i} className="st-diff-item">{item}</div>
      ))}
    </div>
  )
}

function SceneTreeNode({
  node, depth, path,
  expandedNodes, hoveredNodePath, selectedNodePath,
  onToggleExpand, onNodeHover, onNodeSelect,
  genState, onGenerate,
  diffMap,
}) {
  const nodeType = node.type?.toLowerCase() || 'unknown'
  const color = getNodeColor(nodeType)
  const leaf = isLeaf(node)
  const hasChildren = node.children && node.children.length > 0
  const expanded = expandedNodes.has(path)
  const isHovered = hoveredNodePath === path
  const isSelected = selectedNodePath === path

  const roleLabel = node.role || node.role_base
  const diff = diffMap?.[path]
  const hasDiff = !!diff

  const handleClick = () => {
    if (hasChildren) {
      onToggleExpand(path)
    }
    onNodeSelect(isSelected ? null : path)
  }

  return (
    <div className="st-node-group">
      <div
        className={`st-node ${isHovered ? 'hovered' : ''} ${isSelected ? 'selected' : ''} ${hasDiff ? 'st-node-changed' : ''}`}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onMouseEnter={() => onNodeHover(path)}
        onMouseLeave={() => onNodeHover(null)}
        onClick={handleClick}
      >
        {hasChildren ? (
          <span className="st-toggle">
            {expanded ? '▾' : '▸'}
          </span>
        ) : (
          <span className="st-toggle-spacer" />
        )}
        <span className="st-type-badge" style={{ background: color }}>
          {nodeType.toUpperCase()}
        </span>
        {roleLabel && (
          <span className="st-role">{roleLabel}</span>
        )}
        {!roleLabel && node.alignment && (
          <span className="st-align">{node.alignment}</span>
        )}
        {hasChildren && (
          <span className="st-child-count">[{node.children.length}]</span>
        )}
        {leaf && node.content && (
          <span className="st-content" title={node.content}>
            {node.content.length > 30 ? node.content.slice(0, 30) + '…' : node.content}
          </span>
        )}
        {hasDiff && <span className="st-diff-badge">CHANGED</span>}
      </div>
      {hasDiff && (isSelected || expanded) && (
        <div style={{ paddingLeft: `${28 + depth * 16}px` }}>
          <DiffSummary diff={diff} />
        </div>
      )}
      {isSelected && leaf && (
        <div className="st-detail-wrapper" style={{ paddingLeft: `${28 + depth * 16}px` }}>
          <div className="st-detail-card-container">
            <button className="st-detail-close-btn" onClick={(e) => { e.stopPropagation(); onNodeSelect(null) }}>✕</button>
            {nodeType === 'text' && <TextNodeDetail node={node} path={path} genState={genState} onGenerate={onGenerate} />}
            {nodeType === 'image' && <ImageNodeDetail node={node} path={path} genState={genState} onGenerate={onGenerate} />}
            {nodeType === 'chart' && <ChartNodeDetail node={node} path={path} genState={genState} onGenerate={onGenerate} />}
          </div>
        </div>
      )}
      {expanded && hasChildren && (
        <>
          <div className="st-children">
            {node.children.map((child, i) => (
              <SceneTreeNode
                key={`${path}-${i}`}
                node={child}
                depth={depth + 1}
                path={`${path}-${i}`}
                expandedNodes={expandedNodes}
                hoveredNodePath={hoveredNodePath}
                selectedNodePath={selectedNodePath}
                onToggleExpand={onToggleExpand}
                onNodeHover={onNodeHover}
                onNodeSelect={onNodeSelect}
                genState={genState}
                onGenerate={onGenerate}
                diffMap={diffMap}
              />
            ))}
          </div>
          {node.constraints && <ConstraintsCard node={node} depth={depth} diffMap={diffMap} path={path} />}
        </>
      )}
    </div>
  )
}

function countNodes(node) {
  let total = 1, leaves = 0
  if (isLeaf(node)) leaves = 1
  if (node.children) {
    for (const child of node.children) {
      const sub = countNodes(child)
      total += sub.total
      leaves += sub.leaves
    }
  }
  return { total, leaves }
}

function SceneTree({
  sceneGraph,
  hoveredNodePath, onNodeHover,
  selectedNodePath, onNodeSelect,
  genState, onGenerate,
  originalSceneGraph,
}) {
  const [expandedNodes, setExpandedNodes] = useState(new Set(['0']))

  const diffMap = useMemo(() => {
    if (!originalSceneGraph) return null
    return computeDiffMap(originalSceneGraph, sceneGraph)
  }, [originalSceneGraph, sceneGraph])

  const toggleExpand = (path) => {
    setExpandedNodes(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const expandAll = () => {
    const all = new Set()
    const collect = (node, path) => {
      all.add(path)
      if (node.children) {
        node.children.forEach((child, i) => collect(child, `${path}-${i}`))
      }
    }
    collect(sceneGraph, '0')
    setExpandedNodes(all)
  }

  const collapseAll = () => {
    setExpandedNodes(new Set(['0']))
  }

  const stats = countNodes(sceneGraph)
  const diffCount = diffMap ? Object.keys(diffMap).length : 0

  return (
    <div className="st-container">
      <div className="st-toolbar">
        <span className="st-stats">
          {stats.total} nodes · {stats.leaves} leaves
          {diffCount > 0 && <span className="st-diff-count"> · {diffCount} changed</span>}
        </span>
        <button className="st-btn" onClick={expandAll}>Expand All</button>
        <button className="st-btn" onClick={collapseAll}>Collapse</button>
      </div>
      <div className="st-tree">
        <SceneTreeNode
          node={sceneGraph}
          depth={0}
          path="0"
          expandedNodes={expandedNodes}
          hoveredNodePath={hoveredNodePath}
          selectedNodePath={selectedNodePath}
          onToggleExpand={toggleExpand}
          onNodeHover={onNodeHover}
          onNodeSelect={onNodeSelect}
          genState={genState}
          onGenerate={onGenerate}
          diffMap={diffMap}
        />
      </div>
    </div>
  )
}

export default SceneTree
export { getNodeColor, isLeaf, TextNodeDetail, ImageNodeDetail, ChartNodeDetail, ChartGenModal, ConstraintsCard, computeDiffMap }
