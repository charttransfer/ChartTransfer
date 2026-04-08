import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ImageWithBoxes from './ImageWithBoxes'
import SceneGraphModal from './SceneGraphModal'
import './DataExamplePanel.css'

function DataExamplePanel({
  userId, selectedData, onSelectData, selectedExample, onSelectExample,
  sceneGraphData, hoveredNodePath, selectedNodePath,
  onNodeHover, onNodeSelect,
  genState, onGenerate,
}) {
  const [groups, setGroups] = useState([])
  const [dataContent, setDataContent] = useState(null)
  const [loadingData, setLoadingData] = useState(false)
  const [loadingGroups, setLoadingGroups] = useState(true)
  const [dataDropdownOpen, setDataDropdownOpen] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const dataDropdownRef = useRef(null)

  const currentGroup = groups.find(g => g.data_file === selectedData?.data_file)

  useEffect(() => {
    setLoadingGroups(true)
    const params = userId != null ? { user_id: userId } : {}
    axios.get('/api/groups', { params })
      .then(res => {
        setGroups(res.data)
        if (res.data.length === 1 && !selectedData) {
          onSelectData({ data_file: res.data[0].data_file, label: res.data[0].label })
        }
      })
      .finally(() => setLoadingGroups(false))
  }, [userId])

  useEffect(() => {
    if (!selectedData) {
      setDataContent(null)
      return
    }
    setLoadingData(true)
    axios.get(`/api/data/${selectedData.data_file}`)
      .then(res => setDataContent(res.data))
      .finally(() => setLoadingData(false))
  }, [selectedData])

  useEffect(() => {
    function handleClickOutside(e) {
      if (dataDropdownRef.current && !dataDropdownRef.current.contains(e.target)) {
        setDataDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleDataSelect = (group) => {
    onSelectData({ data_file: group.data_file, label: group.label })
    onSelectExample(null)
    setDataDropdownOpen(false)
  }

  const handleExampleSelect = (example) => {
    onSelectExample(example)
    setLightboxOpen(false)
  }

  return (
    <div className="dep">
      <div className="panel-header">Data / Example</div>

      {/* ─── Top section: Tabular Data ─── */}
      <div className="dep-section dep-section-top">
        <div className="section-title">Tabular Data</div>
        <div className="dropdown-wrapper" ref={dataDropdownRef}>
          <button
            className="dropdown-btn"
            onClick={() => setDataDropdownOpen(!dataDropdownOpen)}
          >
            <span className="dropdown-btn-label">
              {selectedData ? selectedData.label : 'Select a dataset...'}
            </span>
            <svg className="dropdown-chevron" viewBox="0 0 20 20" width="14" height="14"
              style={{ transform: dataDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
              <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" fill="currentColor"/>
            </svg>
          </button>
          {dataDropdownOpen && (
            <div className="dropdown-list">
              {loadingGroups && <div className="dropdown-empty">Loading...</div>}
              {groups.map(g => (
                <div
                  key={g.data_file}
                  className={`dropdown-option ${selectedData?.data_file === g.data_file ? 'selected' : ''}`}
                  onClick={() => handleDataSelect(g)}
                >
                  <span className="option-name">{g.label}</span>
                  <span className="option-sub">{g.data_file}</span>
                </div>
              ))}
              {!loadingGroups && groups.length === 0 && (
                <div className="dropdown-empty">No data groups configured</div>
              )}
            </div>
          )}
        </div>

        {loadingData && <div className="loading-bar"><div className="loading-bar-inner" /></div>}

        {dataContent && !loadingData && (
          <div className="data-table-container">
            {dataContent.metadata && (
              <div className="data-meta">
                {dataContent.metadata.title && (
                  <div className="data-meta-title">{dataContent.metadata.title}</div>
                )}
                {dataContent.metadata.description && (
                  <div className="data-meta-desc">{dataContent.metadata.description}</div>
                )}
              </div>
            )}
            <div className="data-table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    {dataContent.columns.map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataContent.rows.slice(0, 50).map((row, i) => (
                    <tr key={i}>
                      {dataContent.columns.map(col => (
                        <td key={col}>{row[col] ?? ''}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {dataContent.rows.length > 50 && (
                <div className="data-table-overflow">
                  Showing 50 of {dataContent.rows.length} rows
                </div>
              )}
            </div>
          </div>
        )}

        {!selectedData && !loadingData && (
          <div className="empty-hint">Select a dataset to preview</div>
        )}
      </div>

      {/* ─── Bottom section: Example ─── */}
      <div className="dep-section dep-section-bottom">
        <div className="section-title">Example Infographic</div>

        {!selectedData && (
          <div className="empty-hint">Select a dataset first to see examples</div>
        )}

        {selectedData && currentGroup && (
          <>
            <div className="example-grid">
              {currentGroup.examples.map((ex, idx) => (
                <div
                  key={ex.id}
                  className={`example-card ${selectedExample?.id === ex.id ? 'selected' : ''}`}
                  onClick={() => handleExampleSelect(ex)}
                >
                  <div className="example-card-img-wrapper">
                    {ex.has_image ? (
                      <img
                        className="example-card-img"
                        src={`/api/example-image/${ex.id}`}
                        alt={`Example ${idx + 1}`}
                      />
                    ) : (
                      <div className="example-card-no-img">No image</div>
                    )}
                  </div>
                  <div className="example-card-label">Example {idx + 1}</div>
                  {ex.chart_type && (
                    <div className="example-card-type">{ex.chart_type}</div>
                  )}
                </div>
              ))}
            </div>

            {selectedExample && selectedExample.has_image && (
              <div className="example-preview">
                <ImageWithBoxes
                  imageSrc={`/api/example-image/${selectedExample.id}`}
                  sceneGraph={sceneGraphData?.scene_graph}
                  imageWidth={sceneGraphData?.image_width}
                  imageHeight={sceneGraphData?.image_height}
                  hoveredNodePath={hoveredNodePath}
                  selectedNodePath={selectedNodePath}
                  onNodeHover={onNodeHover}
                  onNodeSelect={onNodeSelect}
                />
                <div className="example-info-bar">
                  <button className="inspect-btn" onClick={() => setLightboxOpen(true)}>
                    <svg viewBox="0 0 20 20" width="14" height="14" fill="currentColor">
                      <path d="M4 4a2 2 0 00-2 2v1h2V6h1V4H4zM15 4h1a2 2 0 012 2v1h-2V6h-1V4zM2 13v1a2 2 0 002 2h1v-2H4v-1H2zM18 13v1a2 2 0 01-2 2h-1v-2h1v-1h2zM8 8h4v4H8V8z"/>
                    </svg>
                    Inspect
                  </button>
                  {selectedExample.chart_type && (
                    <span className="example-tag">{selectedExample.chart_type}</span>
                  )}
                  {selectedExample.style_keywords && selectedExample.style_keywords.slice(0, 3).map(kw => (
                    <span key={kw} className="example-tag tag-style">{kw}</span>
                  ))}
                </div>
              </div>
            )}

            {lightboxOpen && selectedExample && (
              <SceneGraphModal
                imageSrc={`/api/example-image/${selectedExample.id}`}
                sceneGraphData={sceneGraphData}
                onClose={() => setLightboxOpen(false)}
                genState={genState}
                onGenerate={onGenerate}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default DataExamplePanel
