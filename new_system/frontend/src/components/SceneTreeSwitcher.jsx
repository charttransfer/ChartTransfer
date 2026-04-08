import React, { useState, useMemo } from 'react'
import SceneTree from './SceneTree'
import SceneTreeDiagram from './SceneTreeDiagram'
import './SceneTreeSwitcher.css'

function SceneTreeSwitcher({
  sceneGraph,
  hoveredNodePath, onNodeHover,
  selectedNodePath, onNodeSelect,
  genState, onGenerate,
  originalSceneGraph,
  onBatchGenerate, batchState, onBatchDismiss,
}) {
  const [viewMode, setViewMode] = useState('diagram')

  const batchProgress = useMemo(() => {
    if (!batchState?.nodes) return null
    const { nodes } = batchState
    let total = nodes.length, done = 0
    for (const { path, node, type } of nodes) {
      const state = genState?.[path] || {}
      const b = node.bbox || {}
      const w = b.width || 1, h = b.height || 1
      const isBar = type === 'image' && (h / w > 4 || w / h > 4)
      if (type === 'chart') {
        if (state.step1Status === 'done') done++
      } else if (state.step2Status === 'done') {
        done++
      }
    }
    return { done, total }
  }, [batchState, genState])

  return (
    <div className="sts-container">
      <div className="sts-toolbar">
        <button
          className={`sts-mode-btn ${viewMode === 'list' ? 'active' : ''}`}
          onClick={() => setViewMode('list')}
        >
          List
        </button>
        <button
          className={`sts-mode-btn ${viewMode === 'diagram' ? 'active' : ''}`}
          onClick={() => setViewMode('diagram')}
        >
          Diagram
        </button>
        {onBatchGenerate && (
          <div className="sts-batch-area">
            {batchState?.active ? (
              <>
                <span className="sts-batch-progress">
                  <span className="sts-batch-spinner" />
                  {batchProgress?.done}/{batchProgress?.total}
                </span>
                <button className="sts-batch-stop" onClick={onBatchDismiss}>Stop</button>
              </>
            ) : (
              <button className="sts-batch-btn" onClick={onBatchGenerate}>
                Batch Generate
              </button>
            )}
          </div>
        )}
      </div>
      {(batchState?.active || batchState?.done) && batchState.chartPaths?.length > 0 && (
        <div className="sts-batch-notification">
          <span>Automatically generating text/image elements. Please manually select chart type and variation for Chart nodes.</span>
          <button className="sts-batch-notification-close" onClick={onBatchDismiss}>✕</button>
        </div>
      )}
      <div className="sts-body">
        {viewMode === 'list' ? (
          <SceneTree
            sceneGraph={sceneGraph}
            hoveredNodePath={hoveredNodePath}
            onNodeHover={onNodeHover}
            selectedNodePath={selectedNodePath}
            onNodeSelect={onNodeSelect}
            genState={genState}
            onGenerate={onGenerate}
            originalSceneGraph={originalSceneGraph}
          />
        ) : (
          <SceneTreeDiagram
            sceneGraph={sceneGraph}
            hoveredNodePath={hoveredNodePath}
            onNodeHover={onNodeHover}
            selectedNodePath={selectedNodePath}
            onNodeSelect={onNodeSelect}
            genState={genState}
            onGenerate={onGenerate}
            originalSceneGraph={originalSceneGraph}
          />
        )}
      </div>
    </div>
  )
}

export default SceneTreeSwitcher
