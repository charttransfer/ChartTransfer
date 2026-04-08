# ChartTransfer: Example-Based Infographic Chart Construction

> ChartTransfer is an example-based method for constructing infographic charts. It transfers design knowledge from professionally crafted examples to target data through a scene graph representation and bilevel layout optimization, enabling non-expert users to create high-quality infographic charts.

<p align="center">
  <img src="assets/teaser.png" width="100%"/>
</p>

## Overview

Creating effective infographic charts requires substantial expertise in layout design and visual styling. ChartTransfer lowers this barrier by allowing users to construct new infographic charts by referencing existing examples.

**Key idea:** We introduce a **scene graph** representation that captures the hierarchical structure and spatial constraints of an example infographic. Building on this, ChartTransfer follows a **decomposition-recomposition** pipeline:

1. **Decomposition** — Parse the example into a scene graph that encodes element attributes, hierarchical structure, and spatial constraints.
2. **Recomposition** — Generate new visual elements for the target data and compose them via **bilevel layout optimization** that balances visual similarity to the example with readability of the new infographic.

## Features

- **Scene Graph Representation** — Explicitly models hierarchical grouping (tree edges) and spatial constraints (relative size, gap, orientation, overlap) among visual elements.
- **SDF-Based Spatial Optimization** — Uses Signed Distance Functions to transform discrete element masks into continuous distance fields, enabling gradient-based joint optimization of element positions and sizes.
- **Structural Optimization** — Applies discrete scene graph edits (add, delete, move) guided by an Upper Confidence Bound (UCB) strategy when data distribution changes require layout adaptation.
- **Interactive Authoring Tool** — A web-based tool that allows users to refine elements, styles, and layouts, and browse multiple layout candidates.

## Project Structure

```
ChartTransfer/
├── new_system/                  # Authoring tool
│   ├── backend/                 # Flask API server (app.py)
│   ├── frontend/                # React + Vite UI
│   ├── cache/                   # Pre-computed cache for examples
│   └── config.json              # Example and data group configuration
├── chart_modules/               # Chart and element generation
├── layout_system/               # Layout optimization
├── config.py                    # API key configuration
├── requirements.txt             # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Chrome/Chromium (for SVG-to-PNG rendering)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/charttransfer/ChartTransfer.git
cd ChartTransfer
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Install frontend dependencies**

```bash
cd new_system/frontend
npm install
cd ../..
```

**4. Configure API keys**

Edit `config.py` and fill in your API keys:

```python
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "your-llm-api-key"
```

### Running

**Start the backend server:**

```bash
cd new_system/backend
python app.py
```

**Start the frontend dev server (in a separate terminal):**

```bash
cd new_system/frontend
npm run dev
```

Then open the URL shown in the terminal (default: `http://localhost:5173`).

### Cache Mode

The system includes pre-computed cache for the bundled example, allowing you to explore the authoring tool without calling external APIs for chart and layout generation. Cache behavior is controlled by the `USE_CACHE` environment variable:

```bash
# Enable cache (default)
USE_CACHE=1 python app.py

# Disable cache (run full generation pipeline)
USE_CACHE=0 python app.py
```

## Method

### Design Goals

Derived from expert interviews, ChartTransfer addresses four design goals:

| | Goal | Description |
|---|---|---|
| **G1** | Hierarchical Structure | Preserve the grouping hierarchy of the example |
| **G2** | Spatial Constraints | Maintain relative size, gap, orientation, and overlap within groups |
| **G3** | Element Grouping | Enhance readability via CRAP principles (Contrast, Repetition, Alignment, Proximity) |
| **G4** | Spatial Distribution | Improve ink balance and compactness |

### Bilevel Layout Optimization

The layout optimization is formulated as a bilevel problem:

- **Lower level (Spatial Optimization):** Continuously adjusts element positions and sizes via SDF-based gradient descent to minimize readability loss under the scene graph's structural constraints.
- **Upper level (Structural Optimization):** Applies discrete edits to the scene graph (add, delete, move nodes) to restore readability when data distribution changes cause layout conflicts, while preserving similarity to the example.

## Evaluation

### Quantitative Results

Comparison of spatial optimization methods on 500 test cases:

| Method | Contrast ↓ | Repetition ↓ | Alignment ↓ | Proximity ↓ | Compactness ↑ | Ink Balance ↓ | Time (s) ↓ |
|---|---|---|---|---|---|---|---|
| Initialization | 0.180 | 0.201 | 0.241 | 0.591 | 0.391 | 0.062 | - |
| Ours-Grid | 0.153 | 0.187 | 0.206 | 0.521 | 0.502 | 0.043 | 12.82 |
| **Ours-SDF** | **0.081** | **0.176** | **0.130** | **0.402** | **0.605** | **0.029** | **1.63** |

### User Study

Compared with Nano-Banana-Pro (state-of-the-art generative model) across 30 test cases rated by 15 participants:

- **Data Fidelity:** 4.82 vs. 4.16 (p < 0.001)
- **Creativity:** 3.62 vs. 3.43 (p = 0.044)
- **Layout Similarity:** 4.33 vs. 3.87 (p < 0.001)
- **Aesthetics:** 3.77 vs. 3.87 (n.s.)

The authoring tool achieved a **SUS score of 85.67** (Excellent) in a usability study with 10 novice users.

## License

This project is released under the [Apache 2.0 License](LICENSE).
