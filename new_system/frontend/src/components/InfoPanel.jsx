import React, { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import SceneTreeSwitcher from './SceneTreeSwitcher'
import './InfoPanel.css'

/* ─── Font list (shared with SceneTree) ────────────────────────────── */
const FONT_OPTIONS = [
  'Arial', 'Arial Black', 'Calibri', 'Cambria', 'Comic Sans MS',
  'Courier New', 'Georgia', 'Helvetica', 'Impact', 'Lucida Sans',
  'Montserrat', 'Open Sans', 'Oswald', 'Palatino', 'Roboto',
  'Roboto Condensed', 'Roboto Slab', 'Sans-Serif', 'Sans-Serif Bold',
  'Sans-Serif Extra Bold', 'Rounded Sans-Serif Bold',
  'Tahoma', 'Times New Roman', 'Trebuchet MS', 'Verdana',
]

/* ─── Editable Color Swatch ───────────────────────────────────────── */
function ColorSwatch({ color, label, onChange, onRemove }) {
  const inputRef = useRef(null)
  return (
    <div className="swatch-item" title={`${label}: ${color}`}>
      <div
        className="swatch-color swatch-color-editable"
        style={{ background: color }}
        onClick={() => inputRef.current?.click()}
      />
      <input
        ref={inputRef}
        type="color"
        className="swatch-color-input"
        value={color}
        onChange={e => onChange(e.target.value)}
      />
      <span className="swatch-hex">{color}</span>
      {onRemove && (
        <button className="swatch-remove-btn" onClick={onRemove} title="Remove color">&times;</button>
      )}
    </div>
  )
}

function ColorGroup({ title, colors, groupKey, onColorChange }) {
  if (!groupKey) return null
  const handleChange = (index, newColor) => {
    const updated = [...(colors || [])]
    updated[index] = newColor
    onColorChange(groupKey, updated)
  }
  const handleRemove = (index) => {
    const updated = (colors || []).filter((_, i) => i !== index)
    onColorChange(groupKey, updated)
  }
  const handleAdd = () => {
    const updated = [...(colors || []), '#888888']
    onColorChange(groupKey, updated)
  }
  if (!colors || colors.length === 0) {
    return (
      <div className="color-group">
        <div className="color-group-label">{title}</div>
        <div className="swatch-row">
          <button className="swatch-add-btn" onClick={handleAdd} title="Add color">+</button>
        </div>
      </div>
    )
  }
  return (
    <div className="color-group">
      <div className="color-group-label">{title}</div>
      <div className="swatch-row">
        {colors.map((c, i) => (
          <ColorSwatch
            key={i}
            color={c}
            label={title}
            onChange={v => handleChange(i, v)}
            onRemove={colors.length > 1 ? () => handleRemove(i) : null}
          />
        ))}
        <button className="swatch-add-btn" onClick={handleAdd} title="Add color">+</button>
      </div>
    </div>
  )
}

/* ─── Inline Font Combobox (lightweight) ──────────────────────────── */
function InlineFontCombobox({ value, onChange }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const wrapRef = useRef(null)
  const triggerRef = useRef(null)
  const searchRef = useRef(null)
  const [listPos, setListPos] = useState({ top: 0, left: 0, width: 200 })

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
      setListPos({ top: rect.bottom + 2, left: rect.left, width: Math.max(rect.width, 200) })
    }
    setTimeout(() => searchRef.current?.focus(), 0)
  }

  const select = (f) => { onChange(f); setOpen(false) }

  const lq = query.toLowerCase()
  const filtered = FONT_OPTIONS.filter(f => !lq || f.toLowerCase().includes(lq))

  return (
    <div className="ip-font-combo" ref={wrapRef}>
      <button
        ref={triggerRef}
        className="ip-font-combo-trigger"
        style={{ fontFamily: value }}
        onClick={openDropdown}
        type="button"
      >
        {value || 'Sans-Serif'} <span className="ip-font-combo-arrow">&#9662;</span>
      </button>
      {open && (
        <div className="ip-font-combo-panel" style={{ position: 'fixed', top: listPos.top, left: listPos.left, width: listPos.width, zIndex: 9999 }}>
          <input
            ref={searchRef}
            type="text"
            className="ip-font-combo-search"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search font…"
          />
          <ul className="ip-font-combo-list">
            {filtered.length > 0 ? filtered.map(f => (
              <li
                key={f}
                className={'ip-font-combo-item' + (f === value ? ' ip-font-combo-active' : '')}
                style={{ fontFamily: f }}
                onMouseDown={e => { e.preventDefault(); select(f) }}
              >{f}</li>
            )) : (
              <li className="ip-font-combo-item ip-font-combo-empty">No match</li>
            )}
          </ul>
        </div>
      )}
    </div>
  )
}

const SAMPLE_TEXT = {
  TITLE: 'Main Title',
  TITLE_PRIMARY: 'Primary Title',
  SUBTITLE: 'Subtitle Text',
  DESCRIPTION: 'Description paragraph text here.',
  BODY: 'Body text content for reading.',
  LABEL: 'Data Label',
  CAPTION: 'Source / caption text',
}

const BASE_FONT_SIZE = 64
const MAX_PREVIEW_SIZE = 32
const MIN_PREVIEW_SIZE = 11

function getPreviewFontSize(info) {
  const rawPx = info.font_size_px || (info.size_ratio != null ? Math.round(info.size_ratio * BASE_FONT_SIZE) : null)
  if (!rawPx) return 14
  return Math.max(MIN_PREVIEW_SIZE, Math.min(MAX_PREVIEW_SIZE, rawPx))
}

function getActualFontSize(info) {
  if (info.font_size_px) return info.font_size_px
  if (info.size_ratio != null) return Math.round(info.size_ratio * BASE_FONT_SIZE)
  return 14
}

function mapFontFamily(family) {
  if (!family) return 'sans-serif'
  const lower = family.toLowerCase()
  const quoted = family.includes(' ') ? `"${family}"` : family
  if (lower.includes('serif') && !lower.includes('sans')) return `${quoted}, Georgia, "Times New Roman", serif`
  if (lower.includes('mono')) return `${quoted}, "SF Mono", Menlo, Consolas, monospace`
  return `${quoted}, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
}

/* ─── Editable Typography Row ─────────────────────────────────────── */
function TypographyRow({ role, info, bgColor, onInfoChange }) {
  const fontColor = info.color || '#000'
  const previewSize = getPreviewFontSize(info)
  const actualSize = getActualFontSize(info)
  const fontWeight = info.weight === 'bold' || parseInt(info.weight) >= 600 ? 700 : 400
  const fontStyle = info.font_style === 'italic' ? 'italic' : 'normal'
  const sampleText = SAMPLE_TEXT[role] || role
  const needsDarkBg = bgColor && isLightColor(fontColor) && !isLightColor(bgColor)
  const colorInputRef = useRef(null)

  const update = (field, value) => {
    onInfoChange(role, { ...info, [field]: value })
  }

  return (
    <div className="typo-card">
      <div className="typo-card-header">
        <span className="typo-role">{role}</span>
        <div className="typo-controls">
          <InlineFontCombobox
            value={info.font_family || 'Sans-Serif'}
            onChange={v => update('font_family', v)}
          />
          <select
            className="typo-weight-select"
            value={info.weight || 'normal'}
            onChange={e => update('weight', e.target.value)}
          >
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
          <input
            type="number"
            className="typo-size-input"
            min={8}
            max={200}
            value={actualSize}
            onChange={e => update('font_size_px', parseFloat(e.target.value) || 14)}
            title="Font size (px)"
          />
          <div
            className="typo-color-swatch typo-color-swatch-editable"
            style={{ background: fontColor }}
            title={fontColor}
            onClick={() => colorInputRef.current?.click()}
          />
          <input
            ref={colorInputRef}
            type="color"
            className="typo-color-input-hidden"
            value={fontColor}
            onChange={e => update('color', e.target.value)}
          />
        </div>
      </div>
      <div
        className="typo-preview"
        style={{
          fontSize: `${previewSize}px`,
          fontWeight,
          fontStyle,
          fontFamily: mapFontFamily(info.font_family),
          color: fontColor,
          background: needsDarkBg ? bgColor : undefined,
        }}
      >
        {sampleText}
      </div>
    </div>
  )
}

function isLightColor(hex) {
  if (!hex || !hex.startsWith('#')) return false
  const c = hex.replace('#', '')
  const r = parseInt(c.substring(0, 2), 16)
  const g = parseInt(c.substring(2, 4), 16)
  const b = parseInt(c.substring(4, 6), 16)
  return (r * 299 + g * 587 + b * 114) / 1000 > 160
}

function getTypoSortSize(info) {
  if (info.font_size_px) return info.font_size_px
  if (info.size_ratio != null) return Math.round(info.size_ratio * BASE_FONT_SIZE)
  return 0
}

/* ─── Debounced save helper ───────────────────────────────────────── */
function useDebouncedSave(exampleId, delay = 800) {
  const timerRef = useRef(null)
  const save = useCallback((palette, typography, colorScheme) => {
    if (!exampleId) return
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      axios.put(`/api/style/${exampleId}`, {
        color_palette: palette,
        typography,
        color_scheme: colorScheme,
      }).catch(err => console.warn('Failed to save style:', err))
    }, delay)
  }, [exampleId, delay])

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [])

  return save
}

function InfoPanel({
  selectedExample,
  sceneGraphData, onSceneGraphLoaded,
  hoveredNodePath, onNodeHover,
  selectedNodePath, onNodeSelect,
  genState, onGenerate,
  bilevelSceneGraph,
  onBatchGenerate, batchState, onBatchDismiss,
}) {
  const [styleData, setStyleData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sgLoading, setSgLoading] = useState(false)
  const [splitRatio, setSplitRatio] = useState(0.5)
  const [outerSplitRatio, setOuterSplitRatio] = useState(0.3)
  const splitContainerRef = useRef(null)
  const outerContainerRef = useRef(null)
  const draggingSplit = useRef(false)
  const draggingOuterSplit = useRef(false)

  const saveStyle = useDebouncedSave(selectedExample?.id)

  const handleOuterSplitMouseDown = useCallback((e) => {
    e.preventDefault()
    draggingOuterSplit.current = true
    const onMove = (ev) => {
      if (!draggingOuterSplit.current || !outerContainerRef.current) return
      const rect = outerContainerRef.current.getBoundingClientRect()
      const y = ev.clientY - rect.top
      const ratio = Math.max(0.1, Math.min(0.6, y / rect.height))
      setOuterSplitRatio(ratio)
    }
    const onUp = () => {
      draggingOuterSplit.current = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  const handleSplitMouseDown = useCallback((e) => {
    e.preventDefault()
    draggingSplit.current = true
    const onMove = (ev) => {
      if (!draggingSplit.current || !splitContainerRef.current) return
      const rect = splitContainerRef.current.getBoundingClientRect()
      const y = ev.clientY - rect.top
      const ratio = Math.max(0.15, Math.min(0.85, y / rect.height))
      setSplitRatio(ratio)
    }
    const onUp = () => {
      draggingSplit.current = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [])

  useEffect(() => {
    if (!selectedExample) {
      setStyleData(null)
      onSceneGraphLoaded(null)
      return
    }
    setLoading(true)
    setSgLoading(true)
    axios.get(`/api/style/${selectedExample.id}`)
      .then(res => setStyleData(res.data))
      .catch(() => setStyleData(null))
      .finally(() => setLoading(false))
    axios.get(`/api/scene-graph/${selectedExample.id}`)
      .then(res => onSceneGraphLoaded(res.data))
      .catch(() => onSceneGraphLoaded(null))
      .finally(() => setSgLoading(false))
  }, [selectedExample])

  /* ─── Palette editing ─────────────────────────────────────────── */
  const handleColorChange = useCallback((groupKey, newColors) => {
    setStyleData(prev => {
      if (!prev) return prev
      let updatedPalette
      if (groupKey === 'data_mark_color') {
        updatedPalette = { ...prev.color_palette, data_mark_color: newColors[0] || null }
      } else {
        updatedPalette = { ...prev.color_palette, [groupKey]: newColors }
      }
      const next = { ...prev, color_palette: updatedPalette }
      saveStyle(updatedPalette, prev.typography, prev.color_scheme)
      return next
    })
  }, [saveStyle])

  /* ─── Typography editing ──────────────────────────────────────── */
  const handleTypoChange = useCallback((role, newInfo) => {
    setStyleData(prev => {
      if (!prev) return prev
      const updatedTypo = { ...prev.typography, [role]: newInfo }
      const next = { ...prev, typography: updatedTypo }
      saveStyle(prev.color_palette, updatedTypo, prev.color_scheme)
      return next
    })
  }, [saveStyle])

  const palette = styleData?.color_palette
  const typography = styleData?.typography

  const sortedTypoEntries = typography
    ? Object.entries(typography).sort(([, a], [, b]) => getTypoSortSize(b) - getTypoSortSize(a))
    : []

  return (
    <div className="info-panel">
      <div className="panel-header">Info View</div>
      <div className="panel-content" ref={outerContainerRef}>
        <div className="panel-section panel-section-small" style={{ flex: `0 0 ${outerSplitRatio * 100}%` }}>
          <div className="section-title">Global Style</div>

          {!selectedExample && (
            <div className="placeholder-text">Select an example to view style info</div>
          )}

          {selectedExample && loading && (
            <div className="placeholder-text">Loading style...</div>
          )}

          {selectedExample && !loading && !styleData && (
            <div className="placeholder-text">No style data available</div>
          )}

          {styleData && !loading && (
            <div className="style-content">
              {/* Color Palette */}
              <div className="style-block">
                <div className="style-block-title">
                  Color Palette
                  {styleData.color_scheme && (
                    <span className="scheme-badge">{styleData.color_scheme}</span>
                  )}
                </div>
                <div className="palette-groups">
                  <ColorGroup title="Background" colors={palette.background_colors} groupKey="background_colors" onColorChange={handleColorChange} />
                  {palette.data_mark_color && (
                    <ColorGroup title="Data Mark" colors={[palette.data_mark_color]} groupKey="data_mark_color" onColorChange={handleColorChange} />
                  )}
                  <ColorGroup title="Categorical" colors={palette.categorical_encoding_colors} groupKey="categorical_encoding_colors" onColorChange={handleColorChange} />
                  <ColorGroup title="Numerical" colors={palette.numerical_encoding_colors} groupKey="numerical_encoding_colors" onColorChange={handleColorChange} />
                  <ColorGroup title="Theme" colors={palette.foreground_theme_colors} groupKey="foreground_theme_colors" onColorChange={handleColorChange} />
                </div>
              </div>

              {/* Typography */}
              {sortedTypoEntries.length > 0 && (
                <div className="style-block">
                  <div className="style-block-title">Typography</div>
                  <div className="typo-list">
                    {sortedTypoEntries.map(([role, info]) => (
                      <TypographyRow
                        key={role}
                        role={role}
                        info={info}
                        bgColor={palette?.background_colors?.[0]}
                        onInfoChange={handleTypoChange}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="ip-split-handle ip-outer-split-handle" onMouseDown={handleOuterSplitMouseDown}>
          <div className="ip-split-handle-bar" />
        </div>

        {!bilevelSceneGraph ? (
          <div className="panel-section panel-section-large">
            <div className="section-title">Example Scene Graph</div>
            {!selectedExample && (
              <div className="placeholder-text">Select an example to view scene graph</div>
            )}
            {selectedExample && sgLoading && (
              <div className="placeholder-text">Loading scene graph...</div>
            )}
            {selectedExample && !sgLoading && !sceneGraphData?.scene_graph && (
              <div className="placeholder-text">No scene graph data available</div>
            )}
            {sceneGraphData?.scene_graph && !sgLoading && (
              <SceneTreeSwitcher
                sceneGraph={sceneGraphData.scene_graph}
                hoveredNodePath={hoveredNodePath}
                onNodeHover={onNodeHover}
                selectedNodePath={selectedNodePath}
                onNodeSelect={onNodeSelect}
                genState={genState}
                onGenerate={onGenerate}
                onBatchGenerate={onBatchGenerate}
                batchState={batchState}
                onBatchDismiss={onBatchDismiss}
              />
            )}
          </div>
        ) : (
          <div className="ip-split-container" ref={splitContainerRef}>
            <div className="ip-split-pane" style={{ flex: `0 0 ${splitRatio * 100}%` }}>
              <div className="section-title">Example Scene Graph</div>
              {sceneGraphData?.scene_graph && !sgLoading && (
                <SceneTreeSwitcher
                  sceneGraph={sceneGraphData.scene_graph}
                  hoveredNodePath={hoveredNodePath}
                  onNodeHover={onNodeHover}
                  selectedNodePath={selectedNodePath}
                  onNodeSelect={onNodeSelect}
                  genState={genState}
                  onGenerate={onGenerate}
                  onBatchGenerate={onBatchGenerate}
                  batchState={batchState}
                  onBatchDismiss={onBatchDismiss}
                />
              )}
            </div>
            <div className="ip-split-handle" onMouseDown={handleSplitMouseDown}>
              <div className="ip-split-handle-bar" />
            </div>
            <div className="ip-split-pane ip-bilevel-section" style={{ flex: 1 }}>
              <div className="section-title">
                Target Scene Graph
                <span className="ip-bilevel-badge">Bilevel Optimized</span>
                <span className="ip-diff-legend">
                  <span className="ip-diff-legend-icon" />
                  <span className="ip-diff-legend-text">= differs from original</span>
                </span>
              </div>
              <SceneTreeSwitcher
                sceneGraph={bilevelSceneGraph}
                hoveredNodePath={hoveredNodePath}
                onNodeHover={onNodeHover}
                selectedNodePath={selectedNodePath}
                onNodeSelect={onNodeSelect}
                genState={genState}
                onGenerate={onGenerate}
                originalSceneGraph={sceneGraphData?.scene_graph}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default InfoPanel
