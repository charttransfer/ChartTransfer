import React, { useState } from 'react'
import ImageWithBoxes from './ImageWithBoxes'
import SceneTreeSwitcher from './SceneTreeSwitcher'
import './SceneGraphModal.css'

function SceneGraphModal({ imageSrc, sceneGraphData, onClose, genState, onGenerate }) {
  const [hoveredPath, setHoveredPath] = useState(null)
  const [selectedPath, setSelectedPath] = useState(null)

  return (
    <div className="sgm-overlay" onClick={onClose}>
      <div className="sgm-container" onClick={(e) => e.stopPropagation()}>
        <div className="sgm-header">
          <span className="sgm-title">Scene Graph Inspector</span>
          <button className="sgm-close" onClick={onClose}>&times;</button>
        </div>
        <div className="sgm-body">
          <div className="sgm-image-pane">
            {sceneGraphData?.scene_graph ? (
              <ImageWithBoxes
                imageSrc={imageSrc}
                sceneGraph={sceneGraphData.scene_graph}
                imageWidth={sceneGraphData.image_width}
                imageHeight={sceneGraphData.image_height}
                hoveredNodePath={hoveredPath}
                selectedNodePath={selectedPath}
                onNodeHover={setHoveredPath}
                onNodeSelect={(p) => setSelectedPath(selectedPath === p ? null : p)}
              />
            ) : (
              <img className="sgm-plain-img" src={imageSrc} alt="example" />
            )}
          </div>
          <div className="sgm-tree-pane">
            {sceneGraphData?.scene_graph ? (
              <SceneTreeSwitcher
                sceneGraph={sceneGraphData.scene_graph}
                hoveredNodePath={hoveredPath}
                onNodeHover={setHoveredPath}
                selectedNodePath={selectedPath}
                onNodeSelect={(p) => setSelectedPath(selectedPath === p ? null : p)}
                genState={genState}
                onGenerate={onGenerate}
              />
            ) : (
              <div className="sgm-no-tree">No scene graph data</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SceneGraphModal
