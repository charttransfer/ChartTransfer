import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react'
import { hierarchy, tree } from 'd3-hierarchy'
import { getNodeColor, isLeaf, TextNodeDetail, ImageNodeDetail, ChartNodeDetail, ConstraintsCard, computeDiffMap } from './SceneTree'
import './SceneTreeDiagram.css'

const NODE_W = 90
const NODE_H = 32
const LEAF_W = 160
const H_GAP = 20
const V_GAP = 60

function nodeWidth(d) {
  return isLeaf(d.data) ? LEAF_W : NODE_W
}

function buildHierarchy(sceneGraph) {
  const root = hierarchy(sceneGraph, d => d.children)
  root.eachBefore(node => {
    if (node.parent) {
      const siblingIdx = node.parent.children.indexOf(node)
      node.data._path = `${node.parent.data._path}-${siblingIdx}`
    } else {
      node.data._path = '0'
    }
  })
  return root
}

function linkPath(source, target) {
  const sy = source.y + NODE_H
  const ty = target.y
  const mid = (sy + ty) / 2
  return `M${source.x},${sy} C${source.x},${mid} ${target.x},${mid} ${target.x},${ty}`
}

function SceneTreeDiagram({
  sceneGraph,
  hoveredNodePath, onNodeHover,
  selectedNodePath, onNodeSelect,
  genState, onGenerate,
  originalSceneGraph,
}) {
  const containerRef = useRef(null)
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 })
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 })
  const dragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 })

  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setContainerSize({ w: width, h: height })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const diffMap = useMemo(() => {
    if (!originalSceneGraph) return null
    return computeDiffMap(originalSceneGraph, sceneGraph)
  }, [originalSceneGraph, sceneGraph])

  const { root, treeWidth, treeHeight, offsetX } = useMemo(() => {
    if (!sceneGraph) return { root: null, treeWidth: 0, treeHeight: 0, offsetX: 0 }
    const root = buildHierarchy(sceneGraph)

    const layout = tree()
      .nodeSize([1, NODE_H + V_GAP])
      .separation((a, b) => {
        const wa = nodeWidth(a)
        const wb = nodeWidth(b)
        return (wa / 2 + wb / 2 + H_GAP)
      })
    layout(root)

    let minX = Infinity, maxX = -Infinity
    root.each(node => {
      const w = nodeWidth(node)
      minX = Math.min(minX, node.x - w / 2)
      maxX = Math.max(maxX, node.x + w / 2)
    })

    const depth = root.height
    const treeH = (depth + 1) * (NODE_H + V_GAP)
    const treeW = maxX - minX + 40

    return { root, treeWidth: treeW, treeHeight: treeH, offsetX: -minX + 20 }
  }, [sceneGraph])

  useEffect(() => {
    if (!root || containerSize.w === 0) return
    const fitScale = Math.min(
      containerSize.w / (treeWidth + 80),
      containerSize.h / (treeHeight + 80),
      1.2
    )
    setTransform({
      x: (containerSize.w - treeWidth * fitScale) / 2,
      y: 30,
      scale: fitScale,
    })
  }, [root, containerSize, treeWidth, treeHeight])

  const handleWheel = useCallback((e) => {
    const tag = e.target.tagName
    if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return
    if (e.target.closest('.st-svg-modal-overlay, .st-svg-modal, .st-imggen-modal-overlay, .st-imggen-modal, .st-chartgen-modal-overlay, .st-chartgen-modal')) return
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setTransform(prev => {
      const newScale = Math.max(0.2, Math.min(3, prev.scale * delta))
      const rect = containerRef.current.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      return {
        scale: newScale,
        x: mx - (mx - prev.x) * (newScale / prev.scale),
        y: my - (my - prev.y) * (newScale / prev.scale),
      }
    })
  }, [])

  const handleMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    if (e.target.closest('.st-svg-modal-overlay, .st-svg-modal, .st-imggen-modal-overlay, .st-imggen-modal, .st-chartgen-modal-overlay, .st-chartgen-modal, .std-detail-card')) return
    dragging.current = true
    dragStart.current = { x: e.clientX, y: e.clientY, tx: transform.x, ty: transform.y }
  }, [transform])

  const handleMouseMove = useCallback((e) => {
    if (!dragging.current) return
    setTransform(prev => ({
      ...prev,
      x: dragStart.current.tx + (e.clientX - dragStart.current.x),
      y: dragStart.current.ty + (e.clientY - dragStart.current.y),
    }))
  }, [])

  const handleMouseUp = useCallback(() => {
    dragging.current = false
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('wheel', handleWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  const selectedNode = useMemo(() => {
    if (!root || !selectedNodePath) return null
    let found = null
    root.each(n => { if (n.data._path === selectedNodePath) found = n })
    return found
  }, [root, selectedNodePath])

  const detailPos = useMemo(() => {
    if (!selectedNode || containerSize.w === 0) return null
    const w = nodeWidth(selectedNode)
    const cx = (selectedNode.x + (offsetX || 0)) * transform.scale + transform.x
    const top = (selectedNode.y + NODE_H + 6) * transform.scale + transform.y
    const cardW = 400
    let left = cx - cardW / 2
    if (left < 4) left = 4
    if (left + cardW > containerSize.w - 4) left = containerSize.w - cardW - 4
    return { left, top }
  }, [selectedNode, transform, offsetX, containerSize])

  function renderDetailCard(node) {
    if (!node) return null
    const d = node.data
    const type = d.type?.toLowerCase() || ''
    const leaf = isLeaf(d)
    return (
      <div className="std-detail-card" style={{ left: detailPos.left, top: detailPos.top }}>
        <div className="std-detail-card-header">
          <span className="std-detail-card-type" style={{ background: getNodeColor(type) }}>
            {type.toUpperCase()}
          </span>
          {d.role && <span className="std-detail-card-role">{d.role}</span>}
          <button className="std-detail-card-close" onClick={() => onNodeSelect(null)}>✕</button>
        </div>
        {leaf && type === 'text' && <TextNodeDetail node={d} path={d._path} genState={genState} onGenerate={onGenerate} />}
        {leaf && type === 'image' && <ImageNodeDetail node={d} path={d._path} genState={genState} onGenerate={onGenerate} />}
        {leaf && type === 'chart' && <ChartNodeDetail node={d} path={d._path} genState={genState} onGenerate={onGenerate} />}
        {!leaf && d.constraints && <ConstraintsCard node={d} depth={-1} diffMap={diffMap} path={d._path} />}
      </div>
    )
  }

  if (!root) {
    return <div className="std-empty">No scene graph data</div>
  }

  const nodes = root.descendants()
  const links = root.links()

  return (
    <div
      className="std-container"
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <svg className="std-svg" width={containerSize.w} height={containerSize.h}>
        <g transform={`translate(${transform.x},${transform.y}) scale(${transform.scale})`}>
          <g transform={`translate(${offsetX || 0},0)`}>
          {links.map((link, i) => (
            <path
              key={i}
              className="std-link"
              d={linkPath(link.source, link.target)}
            />
          ))}
          {nodes.map(node => {
            const path = node.data._path
            const nodeType = node.data.type?.toLowerCase() || 'unknown'
            const color = getNodeColor(nodeType)
            const leaf = isLeaf(node.data)
            const w = leaf ? LEAF_W : NODE_W
            const isHovered = hoveredNodePath === path
            const isSelected = selectedNodePath === path
            const active = isHovered || isSelected
            const hasDiff = diffMap && diffMap[path]

            return (
              <g
                key={path}
                transform={`translate(${node.x - w / 2},${node.y})`}
                className={`std-node ${active ? 'active' : ''}`}
                onMouseEnter={() => onNodeHover(path)}
                onMouseLeave={() => onNodeHover(null)}
                onClick={(e) => { e.stopPropagation(); onNodeSelect(isSelected ? null : path) }}
              >
                {hasDiff && (
                  <rect
                    x={-3}
                    y={-3}
                    width={w + 6}
                    height={NODE_H + 6}
                    rx={7}
                    fill="none"
                    stroke="#ff9800"
                    strokeWidth={2.5}
                    strokeDasharray="4 2"
                  />
                )}
                <rect
                  width={w}
                  height={NODE_H}
                  rx={5}
                  fill={active ? color : color + 'dd'}
                  stroke={isSelected ? '#1967d2' : active ? '#fff' : 'none'}
                  strokeWidth={isSelected ? 2.5 : active ? 1.5 : 0}
                />
                <text
                  x={leaf ? 8 : w / 2}
                  y={NODE_H / 2 + 1}
                  dominantBaseline="central"
                  textAnchor={leaf ? 'start' : 'middle'}
                  className="std-node-label"
                >
                  {nodeType.toUpperCase()}
                </text>
                {leaf && node.data.content && (
                  <text
                    x={8 + nodeType.length * 8 + 6}
                    y={NODE_H / 2 + 1}
                    dominantBaseline="central"
                    className="std-node-content"
                  >
                    {node.data.content.length > 12
                      ? node.data.content.slice(0, 12) + '…'
                      : node.data.content}
                  </text>
                )}
                {!leaf && node.data.children && (
                  <text
                    x={w / 2}
                    y={NODE_H + 14}
                    textAnchor="middle"
                    className="std-node-count"
                  >
                    [{node.data.children.length}]
                  </text>
                )}
              </g>
            )
          })}
          </g>
        </g>
      </svg>
      {selectedNode && detailPos && renderDetailCard(selectedNode)}
    </div>
  )
}

export default SceneTreeDiagram
