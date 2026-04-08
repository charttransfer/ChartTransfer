import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import axios from 'axios'
import DataExamplePanel from '../components/DataExamplePanel'
import InfoPanel from '../components/InfoPanel'
import LayoutPanel from '../components/LayoutPanel'
import './WorkspaceView.css'

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
      .catch(() => {
        onUpdate('error', null)
      })
  }
  setTimeout(poll, intervalMs)
}

function WorkspaceView({ sessionId }) {
  const { userId } = useParams()
  const [selectedData, setSelectedData] = useState(null)
  const [selectedExample, setSelectedExample] = useState(null)
  const [sceneGraphData, setSceneGraphData] = useState(null)
  const [hoveredNodePath, setHoveredNodePath] = useState(null)
  const [selectedNodePath, setSelectedNodePath] = useState(null)
  const [genState, setGenState] = useState({})
  const [bilevelSceneGraph, setBilevelSceneGraph] = useState(null)
  const selectedDataRef = useRef(null)
  selectedDataRef.current = selectedData
  const genStateRef = useRef(genState)
  genStateRef.current = genState
  const [batchState, setBatchState] = useState(null)
  const batchAutoChainRef = useRef(new Set())

  const updateGenState = useCallback((path, updates) => {
    setGenState(prev => ({
      ...prev,
      [path]: { ...(prev[path] || {}), ...updates },
    }))
  }, [])

  const handleGenerate = useCallback((path, node, action, payload) => {
    if (action === 'editText') {
      setGenState(prev => ({
        ...prev,
        [path]: { ...(prev[path] || {}), textContent: payload, step2Status: 'idle', svgData: null },
      }))
      return
    }

    if (action === 'editSvgLine') {
      setGenState(prev => {
        const cur = prev[path] || {}
        const svgData = cur.svgData
        if (!svgData?.lines) return prev
        const newLines = svgData.lines.map((line, i) =>
          i === payload.index ? { ...line, segments: payload.segments } : line
        )
        return {
          ...prev,
          [path]: { ...cur, svgData: { ...svgData, lines: newLines } },
        }
      })
      return
    }

    if (action === 'editStyle') {
      setGenState(prev => {
        const cur = prev[path] || {}
        const svgData = cur.svgData
        if (!svgData?.lines) return prev
        const newLines = svgData.lines.map(line => ({
          ...line,
          segments: (line.segments || []).map(seg => ({
            ...seg,
            [payload.field]: payload.value,
          })),
        }))
        return {
          ...prev,
          [path]: { ...cur, svgData: { ...svgData, lines: newLines } },
        }
      })
      return
    }

    if (action === 'editSegmentStyle') {
      setGenState(prev => {
        const cur = prev[path] || {}
        const svgData = cur.svgData
        if (!svgData?.lines) return prev
        const { lineIndex, charStart, charEnd, field, value } = payload
        const newLines = svgData.lines.map((line, i) => {
          if (i !== lineIndex) return line
          const segs = line.segments || []
          const result = []
          let offset = 0
          for (const seg of segs) {
            const segEnd = offset + seg.text.length
            if (segEnd <= charStart || offset >= charEnd) {
              result.push(seg)
            } else {
              if (offset < charStart)
                result.push({ ...seg, text: seg.text.slice(0, charStart - offset) })
              const s = Math.max(0, charStart - offset)
              const e = Math.min(seg.text.length, charEnd - offset)
              result.push({ ...seg, text: seg.text.slice(s, e), [field]: value })
              if (segEnd > charEnd)
                result.push({ ...seg, text: seg.text.slice(charEnd - offset) })
            }
            offset = segEnd
          }
          const merged = []
          for (const seg of result) {
            const prev = merged[merged.length - 1]
            if (prev && prev.font_family === seg.font_family && prev.font_size === seg.font_size
                && prev.font_weight === seg.font_weight && prev.fill === seg.fill) {
              prev.text += seg.text
            } else {
              merged.push({ ...seg })
            }
          }
          return { ...line, segments: merged }
        })
        return {
          ...prev,
          [path]: { ...cur, svgData: { ...svgData, lines: newLines } },
        }
      })
      return
    }

    if (action === 'changeAlignment') {
      const cur = genStateRef.current[path] || {}
      const pngUrl = cur.pngUrl
      if (!pngUrl) return
      updateGenState(path, { alignBusy: true })
      const exampleId = selectedExample?.id
      const dataFile = selectedDataRef.current?.data_file
      axios.post('/api/realign-text', {
        png_url: pngUrl,
        alignment: payload,
        example_id: exampleId,
        data_file: dataFile,
        node_path: path,
        session_id: sessionId,
      }).then(res => {
        updateGenState(path, {
          alignBusy: false,
          pngUrl: res.data.pngUrl,
          alignment: res.data.alignment,
        })
      }).catch(() => {
        updateGenState(path, { alignBusy: false })
      })
      return
    }

    if (action === 'editPrompt') {
      setGenState(prev => ({
        ...prev,
        [path]: { ...(prev[path] || {}), imagePrompt: payload, step2Status: 'idle', imageUrl: null },
      }))
      return
    }

    if (action === 'selectModel') {
      updateGenState(path, { selectedModel: payload })
      return
    }

    if (action === 'selectTemplate') {
      updateGenState(path, { selectedTemplate: payload, chartUrl: null })
      return
    }

    if (action === 'selectChartType') {
      updateGenState(path, { selectedChartType: payload, variationPreviews: {}, selectedTemplate: null, chartUrl: null })
      return
    }

    if (action === 'selectVariation') {
      setGenState(prev => {
        const cur = prev[path] || {}
        const vp = cur.variationPreviews || {}
        const preview = vp[payload]
        return {
          ...prev,
          [path]: { ...cur, selectedTemplate: payload, chartUrl: preview?.chartUrl || null },
        }
      })
      return
    }

    const nodeType = node.type?.toLowerCase()
    const exampleId = selectedExample?.id
    const dataFile = selectedDataRef.current?.data_file

    if (!exampleId || !dataFile) {
      console.warn('[Generate] No example or data selected')
      return
    }

    if (nodeType === 'text') {
      if (action === 'step1') {
        updateGenState(path, { step1Status: 'generating', textContent: null, step2Status: 'idle', svgData: null })
        axios.post('/api/generate-text-content', {
          example_id: exampleId,
          data_file: dataFile,
          node_path: path,
          node_info: node,
          session_id: sessionId,
        }).then(res => {
          const { task_id } = res.data
          pollTask(task_id, (status, result) => {
            if (status === 'done' && result) {
              updateGenState(path, { step1Status: 'done', textContent: result })
            } else if (status === 'error') {
              updateGenState(path, { step1Status: 'idle' })
            }
          })
        }).catch(() => {
          updateGenState(path, { step1Status: 'idle' })
        })
      } else if (action === 'step2') {
        setGenState(prev => {
          const cur = prev[path] || {}
          if (!cur.textContent) return prev
          const next = { ...prev, [path]: { ...cur, step2Status: 'generating', svgData: null } }

          let styleOverrides = null
          if (cur.svgData?.lines?.length > 0) {
            const seg = cur.svgData.lines[0].segments?.[0]
            if (seg) {
              styleOverrides = {
                font_family: seg.font_family,
                font_weight: seg.font_weight,
                font_size_px: seg.font_size,
                color: seg.fill,
              }
            }
          }

          axios.post('/api/render-text', {
            example_id: exampleId,
            data_file: dataFile,
            node_path: path,
            text_content: cur.textContent,
            node_info: node,
            style_overrides: styleOverrides,
            session_id: sessionId,
          }).then(res => {
            const { task_id } = res.data
            pollTask(task_id, (status, result) => {
              if (status === 'done' && result?.svg_data) {
                updateGenState(path, {
                  step2Status: 'done',
                  svgData: result.svg_data,
                  pngUrl: result.png_path ? `/api/element-image/${result.png_path}` : undefined,
                })
              } else if (status === 'error') {
                updateGenState(path, { step2Status: 'idle' })
              }
            })
          }).catch(() => {
            updateGenState(path, { step2Status: 'idle' })
          })
          return next
        })
      }
    } else if (nodeType === 'image') {
      if (action === 'generateBar') {
        updateGenState(path, { barGenerating: true })
        axios.post('/api/generate-decorative-bar', {
          example_id: exampleId,
          data_file: dataFile,
          node_path: path,
          node_info: node,
          session_id: sessionId,
        }).then(res => {
          const { imageUrl } = res.data
          updateGenState(path, {
            barGenerating: false,
            step1Status: 'done',
            step2Status: 'done',
            imageUrl,
          })
        }).catch(() => {
          updateGenState(path, { barGenerating: false })
        })
        return
      }
      if (action === 'step1') {
        updateGenState(path, { step1Status: 'generating', imagePrompt: null, step2Status: 'idle', imageUrl: null })
        axios.post('/api/generate-image-prompt', {
          example_id: exampleId,
          data_file: dataFile,
          node_path: path,
          node_info: node,
          session_id: sessionId,
        }).then(res => {
          const { task_id } = res.data
          pollTask(task_id, (status, result) => {
            if (status === 'done' && result?.prompt) {
              updateGenState(path, { step1Status: 'done', imagePrompt: result.prompt })
            } else if (status === 'error') {
              updateGenState(path, { step1Status: 'idle' })
            }
          })
        }).catch(() => {
          updateGenState(path, { step1Status: 'idle' })
        })
      } else if (action === 'step2') {
        setGenState(prev => {
          const cur = prev[path] || {}
          if (!cur.imagePrompt) return prev
          const model = cur.selectedModel || 'gpt-image-1'
          const next = { ...prev, [path]: { ...cur, step2Status: 'generating', imageUrl: null } }

          axios.post('/api/generate-image', {
            prompt: cur.imagePrompt,
            model,
            node_path: path,
            example_id: exampleId,
            data_file: dataFile,
            bbox: node.bbox || {},
            session_id: sessionId,
          }).then(res => {
            const { task_id } = res.data
            pollTask(task_id, (status, result) => {
              if (status === 'done' && result?.image_path) {
                updateGenState(path, {
                  step2Status: 'done',
                  imageUrl: `/api/element-image/${result.image_path}`,
                })
              } else if (status === 'error') {
                updateGenState(path, { step2Status: 'idle' })
              }
            })
          }).catch(() => {
            updateGenState(path, { step2Status: 'idle' })
          })
          return next
        })
      }
    } else if (nodeType === 'chart') {
      if (action === 'step1') {
        updateGenState(path, {
          step1Status: 'generating', chartTemplates: null,
          selectedChartType: null, variationPreviews: {},
          selectedTemplate: null, chartUrl: null,
        })
        axios.post('/api/find-chart-templates', {
          example_id: exampleId,
          data_file: dataFile,
          node_info: node,
          session_id: sessionId,
        }).then(res => {
          const { templates: tpls } = res.data
          updateGenState(path, {
            step1Status: 'done',
            chartTemplates: tpls || [],
          })
        }).catch(() => {
          updateGenState(path, { step1Status: 'idle' })
        })
      } else if (action === 'renderVariations') {
        const chartTypeToRender = payload
        setGenState(prev => {
          const cur = prev[path] || {}
          const templates = (cur.chartTemplates || []).filter(t => t.chart_type === chartTypeToRender)
          if (!templates.length) return prev

          const initPreviews = {}
          templates.forEach(t => { initPreviews[t.name] = { status: 'generating', chartUrl: null } })
          const next = {
            ...prev,
            [path]: { ...cur, selectedChartType: chartTypeToRender, variationPreviews: initPreviews, selectedTemplate: null, chartUrl: null },
          }

          templates.forEach(tpl => {
            axios.post('/api/generate-chart', {
              example_id: exampleId,
              data_file: dataFile,
              template_name: tpl.name,
              template_path: tpl.template,
              template_fields: tpl.fields,
              node_path: path,
              node_info: node,
              session_id: sessionId,
            }).then(res => {
              const { task_id } = res.data
              pollTask(task_id, (status, result) => {
                if (status === 'done' && result?.image_path) {
                  setGenState(p => {
                    const c = p[path] || {}
                    const vp = { ...(c.variationPreviews || {}) }
                    vp[tpl.name] = { status: 'done', chartUrl: `/api/element-image/${result.image_path}` }
                    return { ...p, [path]: { ...c, variationPreviews: vp } }
                  })
                } else if (status === 'error') {
                  setGenState(p => {
                    const c = p[path] || {}
                    const vp = { ...(c.variationPreviews || {}) }
                    vp[tpl.name] = { status: 'error', chartUrl: null }
                    return { ...p, [path]: { ...c, variationPreviews: vp } }
                  })
                }
              })
            }).catch(() => {
              setGenState(p => {
                const c = p[path] || {}
                const vp = { ...(c.variationPreviews || {}) }
                vp[tpl.name] = { status: 'error', chartUrl: null }
                return { ...p, [path]: { ...c, variationPreviews: vp } }
              })
            })
          })

          return next
        })
      }
    } else {
      console.log('[Generate] unhandled node type', nodeType, path, action)
    }
  }, [selectedExample, updateGenState, sessionId])

  useEffect(() => {
    if (!selectedExample?.id || !selectedData?.data_file) return
    axios.get(`/api/gen-cache/${selectedExample.id}/${selectedData.data_file}`, {
      params: { session_id: sessionId },
    })
      .then(res => {
        const manifest = res.data
        if (manifest && Object.keys(manifest).length > 0) {
          setGenState(manifest)
        }
      })
      .catch(() => {})
  }, [selectedExample?.id, selectedData?.data_file, sessionId])

  const _isBarNode = (node) => {
    const b = node.bbox || {}
    const w = b.width || 1, h = b.height || 1
    return h / w > 4 || w / h > 4
  }

  const handleBatchGenerate = useCallback(() => {
    const sg = sceneGraphData?.scene_graph
    if (!sg || !selectedExample?.id || !selectedDataRef.current?.data_file) return

    const leafNodes = []
    const collect = (node, path) => {
      const type = node.type?.toLowerCase()
      if (['text', 'image', 'chart'].includes(type)) {
        leafNodes.push({ path, node, type })
      }
      if (node.children) {
        node.children.forEach((child, i) => collect(child, `${path}-${i}`))
      }
    }
    collect(sg, '0')
    if (!leafNodes.length) return

    batchAutoChainRef.current = new Set()
    const chartPaths = leafNodes.filter(n => n.type === 'chart').map(n => n.path)
    setBatchState({ active: true, done: false, nodes: leafNodes, chartPaths })

    const gs = genStateRef.current
    for (const { path, node, type } of leafNodes) {
      const state = gs[path] || {}
      const isBar = type === 'image' && _isBarNode(node)

      if (type === 'chart') {
        if (state.step1Status !== 'done' && state.step1Status !== 'generating') {
          handleGenerate(path, node, 'step1')
        }
      } else if (type === 'text') {
        if (state.step2Status === 'done') {
          /* already done */
        } else if (state.step1Status === 'done') {
          batchAutoChainRef.current.add(path)
          handleGenerate(path, node, 'step2')
        } else if (state.step1Status !== 'generating') {
          handleGenerate(path, node, 'step1')
        }
      } else if (type === 'image') {
        if (state.step2Status === 'done') {
          /* already done */
        } else if (isBar) {
          if (state.step2Status !== 'generating') {
            handleGenerate(path, node, 'generateBar')
          }
        } else if (state.step1Status === 'done') {
          batchAutoChainRef.current.add(path)
          handleGenerate(path, node, 'step2')
        } else if (state.step1Status !== 'generating') {
          handleGenerate(path, node, 'step1')
        }
      }
    }
  }, [sceneGraphData, selectedExample, handleGenerate])

  useEffect(() => {
    if (!batchState?.active) return

    let allDone = true
    for (const { path, node, type } of batchState.nodes) {
      const state = genState[path] || {}
      const isBar = type === 'image' && _isBarNode(node)

      if (type === 'chart') {
        if (state.step1Status !== 'done') allDone = false
      } else if (type === 'text') {
        if (state.step1Status === 'done' && state.step2Status !== 'done' && state.step2Status !== 'generating' && !batchAutoChainRef.current.has(path)) {
          batchAutoChainRef.current.add(path)
          handleGenerate(path, node, 'step2')
        }
        if (state.step2Status !== 'done') allDone = false
      } else if (type === 'image' && !isBar) {
        if (state.step1Status === 'done' && state.step2Status !== 'done' && state.step2Status !== 'generating' && !batchAutoChainRef.current.has(path)) {
          batchAutoChainRef.current.add(path)
          handleGenerate(path, node, 'step2')
        }
        if (state.step2Status !== 'done') allDone = false
      } else if (type === 'image' && isBar) {
        if (state.step2Status !== 'done') allDone = false
      }
    }

    if (allDone) {
      setBatchState(prev => prev ? { ...prev, active: false, done: true } : null)
      batchAutoChainRef.current = new Set()
    }
  }, [batchState, genState, handleGenerate])

  return (
    <div className="workspace">
      <header className="workspace-header">
        <h1 className="workspace-title">ChartTransfer</h1>
        <div className="workspace-subtitle">Infographic Authoring System</div>
      </header>
      <div className="workspace-body">
        <div className="panel panel-left">
          <DataExamplePanel
            userId={userId}
            selectedData={selectedData}
            onSelectData={setSelectedData}
            selectedExample={selectedExample}
            onSelectExample={(ex) => {
              setSelectedExample(ex)
              setSceneGraphData(null)
              setHoveredNodePath(null)
              setSelectedNodePath(null)
              setGenState({})
              setBilevelSceneGraph(null)
              setBatchState(null)
              batchAutoChainRef.current = new Set()
            }}
            sceneGraphData={sceneGraphData}
            hoveredNodePath={hoveredNodePath}
            selectedNodePath={selectedNodePath}
            onNodeHover={setHoveredNodePath}
            onNodeSelect={setSelectedNodePath}
            genState={genState}
            onGenerate={handleGenerate}
          />
        </div>
        <div className="panel panel-center">
          <InfoPanel
            selectedExample={selectedExample}
            sceneGraphData={sceneGraphData}
            onSceneGraphLoaded={setSceneGraphData}
            hoveredNodePath={hoveredNodePath}
            onNodeHover={setHoveredNodePath}
            selectedNodePath={selectedNodePath}
            onNodeSelect={setSelectedNodePath}
            genState={genState}
            onGenerate={handleGenerate}
            bilevelSceneGraph={bilevelSceneGraph}
            onBatchGenerate={handleBatchGenerate}
            batchState={batchState}
            onBatchDismiss={() => setBatchState(null)}
          />
        </div>
        <div className="panel panel-right">
          <LayoutPanel
            selectedExample={selectedExample}
            selectedData={selectedData}
            sceneGraphData={sceneGraphData}
            genState={genState}
            onBilevelSceneGraphChange={setBilevelSceneGraph}
            sessionId={sessionId}
          />
        </div>
      </div>
    </div>
  )
}

export default WorkspaceView
