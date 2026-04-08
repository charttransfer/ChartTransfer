import React, { useState, useRef, useEffect, useCallback } from 'react'
import { getNodeColor, isLeaf } from './SceneTree'
import './ImageWithBoxes.css'

function collectNodes(node, path = '0') {
  const result = []
  const nodeType = node.type?.toLowerCase() || ''
  const leaf = isLeaf(node)
  if (node.bbox) {
    result.push({ node, path, leaf, type: nodeType })
  }
  if (node.children) {
    node.children.forEach((child, i) => {
      result.push(...collectNodes(child, `${path}-${i}`))
    })
  }
  return result
}

function ImageWithBoxes({
  imageSrc, sceneGraph, imageWidth, imageHeight,
  hoveredNodePath, selectedNodePath,
  onNodeHover, onNodeSelect,
  onClick,
  showGroupBoxes = true,
}) {
  const containerRef = useRef(null)
  const imgRef = useRef(null)
  const [displaySize, setDisplaySize] = useState(null)
  const [natSize, setNatSize] = useState(null)

  useEffect(() => {
    setDisplaySize(null)
    setNatSize(null)
  }, [imageSrc])

  const computeDisplay = useCallback(() => {
    if (!containerRef.current) return
    const srcW = imageWidth || natSize?.w
    const srcH = imageHeight || natSize?.h
    if (!srcW || !srcH) return
    const rect = containerRef.current.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return
    const scale = Math.min(rect.width / srcW, rect.height / srcH)
    setDisplaySize({ w: Math.floor(srcW * scale), h: Math.floor(srcH * scale) })
  }, [imageWidth, imageHeight, natSize])

  const handleImgLoad = useCallback(() => {
    if (imgRef.current) {
      setNatSize({ w: imgRef.current.naturalWidth, h: imgRef.current.naturalHeight })
    }
  }, [])

  useEffect(() => {
    computeDisplay()
  }, [computeDisplay])

  useEffect(() => {
    const ro = new ResizeObserver(() => computeDisplay())
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [computeDisplay])

  const srcW = imageWidth || natSize?.w || 1
  const srcH = imageHeight || natSize?.h || 1
  const scaleX = displaySize ? displaySize.w / srcW : 1
  const scaleY = displaySize ? displaySize.h / srcH : 1

  const allNodes = sceneGraph ? collectNodes(sceneGraph) : []
  const groupNodes = allNodes.filter(n => !n.leaf)
  const leafNodes = allNodes.filter(n => n.leaf)

  const hasSize = displaySize && displaySize.w > 0 && displaySize.h > 0

  return (
    <div className="iwb-container" ref={containerRef}>
      {!hasSize && (
        <img
          className="iwb-img-hidden"
          src={imageSrc}
          ref={imgRef}
          onLoad={handleImgLoad}
          alt=""
        />
      )}
      {hasSize && (
        <div className="iwb-inner" style={{ width: displaySize.w, height: displaySize.h }}>
          <img
            className="iwb-img"
            src={imageSrc}
            ref={imgRef}
            onLoad={handleImgLoad}
            onClick={onClick}
            style={{ width: displaySize.w, height: displaySize.h }}
            draggable={false}
            alt="example"
          />
          {sceneGraph && (
            <svg className="iwb-svg" width={displaySize.w} height={displaySize.h}>
              {showGroupBoxes && groupNodes.map(({ node, path, type }) => {
                const b = node.bbox
                if (!b) return null
                const x = b.x * scaleX, y = b.y * scaleY
                const w = b.width * scaleX, h = b.height * scaleY
                const active = hoveredNodePath === path || selectedNodePath === path
                return (
                  <rect
                    key={`g-${path}`}
                    x={x} y={y} width={w} height={h}
                    fill={active ? getNodeColor(type) + '10' : 'none'}
                    stroke={getNodeColor(type)}
                    strokeWidth={active ? 1.5 : 0.8}
                    strokeDasharray="6,4"
                    opacity={active ? 1 : 0.35}
                    style={{ cursor: 'pointer', pointerEvents: 'all' }}
                    onMouseEnter={() => onNodeHover?.(path)}
                    onMouseLeave={() => onNodeHover?.(null)}
                    onClick={(e) => { e.stopPropagation(); onNodeSelect?.(selectedNodePath === path ? null : path) }}
                  />
                )
              })}
              {leafNodes.map(({ node, path, type }) => {
                const b = node.bbox
                if (!b) return null
                const x = b.x * scaleX, y = b.y * scaleY
                const w = b.width * scaleX, h = b.height * scaleY
                const active = hoveredNodePath === path || selectedNodePath === path
                const color = getNodeColor(type)
                return (
                  <g key={`l-${path}`}>
                    <rect
                      x={x} y={y} width={w} height={h}
                      fill={active ? color + '18' : 'none'}
                      stroke={color}
                      strokeWidth={active ? 2 : 1}
                      opacity={active ? 1 : 0.5}
                      style={{ cursor: 'pointer', pointerEvents: 'all' }}
                      onMouseEnter={() => onNodeHover?.(path)}
                      onMouseLeave={() => onNodeHover?.(null)}
                      onClick={(e) => { e.stopPropagation(); onNodeSelect?.(selectedNodePath === path ? null : path) }}
                    />
                    {active && (
                      <>
                        <rect
                          x={x} y={y - 14}
                          width={type.length * 7 + 8} height={14}
                          rx={2}
                          fill={color}
                        />
                        <text
                          x={x + 4} y={y - 3}
                          fill="#fff"
                          fontSize="9"
                          fontWeight="700"
                          letterSpacing="0.3"
                        >
                          {type.toUpperCase()}
                        </text>
                      </>
                    )}
                  </g>
                )
              })}
            </svg>
          )}
        </div>
      )}
    </div>
  )
}

export default ImageWithBoxes
