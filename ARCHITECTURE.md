# System Architecture

## Overview

This system extracts small standard parts (screws, rivets, anchors, washers, nuts) from vector PDF construction plans and generates an accurate BOM with hierarchical multipliers, targeting ≥95% accuracy.

## Three-Layer Architecture

### 1. Perception Layer (`src/perception/`)

**Purpose**: Detect candidate fasteners on each PDF page.

**Components**:
- `pdf_processor.py`: Extracts pages, vector primitives, text blocks, and raster images from PDF
- `detector.py`: Vector-assisted detection using:
  - Circle detection (fastener heads)
  - Thread pattern detection (parallel lines)
  - Text proximity analysis (extract specs from nearby text)
  - Classification logic (screw, rivet, anchor, washer, nut)

**Key Features**:
- Works with vector primitives (circles, lines, paths)
- Extracts text near detections for spec parsing (e.g., "M6 x 80", "Spax")
- Generates confidence scores based on pattern matching
- Filters detections by size and pattern characteristics

**Output**: List of `Detection` objects with:
- Bounding box and center coordinates
- Category and confidence
- Nearby text snippets
- Extracted spec fields (thread, length, brand, etc.)

### 2. Structure Reasoning Layer (`src/structure/`)

**Purpose**: Build page hierarchy and assign multipliers using LLM reasoning.

**Components**:
- `hierarchy_builder.py`: Uses LLM to:
  - Build hierarchy tree (overview → detail → sub-detail)
  - Infer multipliers from callouts and text references
  - Assign detections to hierarchy nodes
  - Reconcile duplicate detections

**LLM Input**:
- Page titles and metadata
- Viewport frames and labels
- Detection summaries with positions
- Text blocks for context

**LLM Output**:
- Hierarchy nodes with parent-child relationships
- Multipliers per node (e.g., "Detail A applies to axes 1-4" → multiplier 4)
- Detection-to-node assignments with reasoning

**Fallback**: If LLM fails, creates simple per-page hierarchy structure.

### 3. Aggregation & QC Layer (`src/aggregation/`)

**Purpose**: Normalize specs, merge instances, and generate final BOM.

**Components**:
- `bom_builder.py`: 
  - Groups detections by normalized specification
  - Calculates total counts (per_instance × multiplier)
  - Applies sanity checks (high counts, missing specs)
  - Flags items needing review

**Normalization**:
- Thread specs: "M6", "M6x80" → "M6"
- Length specs: "x80", "80mm" → "80mm"
- Brand codes: extracts from text (Spax, Hilti, Fischer, etc.)

**Quality Control**:
- Flags items with confidence < threshold
- Flags unusual counts (>1000 fasteners)
- Flags missing critical specs (thread for screws/rivets)

**Output**: List of `BOMItem` objects with:
- Normalized specifications
- Per-instance count and multiplier
- Total count calculation
- Review flags and reasoning

## Export Modules (`src/export/`)

### Excel Exporter (`excel_exporter.py`)

Creates three-sheet Excel workbook:
1. **BOM Sheet**: Main bill of materials with all specs, counts, and multipliers
2. **Hierarchy Sheet**: Page/view hierarchy tree
3. **Evidence Sheet**: Detection evidence tracking

Features:
- Color-coded headers
- Review flags highlighted
- Summary statistics
- Auto-adjusted column widths

## UI (`src/ui/app.py`)

Streamlit-based review interface with:

1. **BOM Table**: 
   - Filterable by category and review status
   - Sortable by count, confidence, category
   - Shows multiplier impact

2. **Hierarchy Tree**:
   - Visual tree structure
   - Multiplier badges
   - Page assignments

3. **Multiplier Visualization**:
   - Bar chart showing total counts by multiplier value

4. **Detection Overview**:
   - Category distribution (pie chart)
   - Confidence histogram

## Main Pipeline (`src/pipeline.py`)

Orchestrates the entire workflow:

1. Extract pages and vector data from PDF
2. Detect fasteners on each page
3. Build hierarchy with LLM
4. Reconcile duplicates
5. Build normalized BOM
6. Apply sanity checks
7. Export Excel and JSON

## Data Flow

```
PDF → PDFProcessor → PageData[]
  ↓
VectorAssistedDetector → Detection[]
  ↓
HierarchyBuilder (LLM) → Hierarchy + Assignments
  ↓
BOMBuilder → BOMItem[]
  ↓
ExcelExporter + JSON → Excel + JSON files
```

## Configuration (`config.yaml`)

- Detection parameters (circle sizes, thresholds)
- LLM provider and model settings
- Part classes and spec fields
- Accuracy targets and review thresholds

## Key Design Decisions

1. **Vector-Assisted Detection**: Uses vector primitives to reduce false positives from raster-only detection
2. **LLM for Structure**: LLM understands context (callouts, repetition) better than rule-based systems
3. **Hierarchical Multipliers**: Properly handles "Detail A applies to 4 locations" scenarios
4. **Review Flags**: Conservative approach - flags uncertain items rather than guessing
5. **Fallback Mechanisms**: System degrades gracefully if LLM fails or no detections found

## Accuracy Strategy

- **Target**: ≥95% item-level acceptance
- **Combined Scoring**: Perception confidence + vector cues + text proximity
- **Review Threshold**: Items with confidence < 0.7 flagged for review
- **Sanity Checks**: Automatic flagging of unusual patterns

## Future Enhancements

Potential improvements mentioned in requirements:
- Fine-tuned YOLO/RT-DETR models for raster detection
- Perceptual hashing for duplicate view detection
- Vendor library integration for part naming
- DWG support when available
- More sophisticated hexagon/nut detection

