# Screw Localization System

A system to extract small standard parts (screws, rivets, anchors, washers, nuts) from vector PDF plans and generate an accurate BOM with hierarchical multipliers.

## Architecture

Three-layer system:
1. **Perception**: Detect candidate fasteners on each page using vector-assisted detection
2. **Structure Reasoning (LLM)**: Build page hierarchy, infer multipliers, reconcile duplicates
3. **Aggregation & QC**: Merge instances, normalize specs, generate Excel BOM

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for LLM reasoning

## Usage

```bash
python main.py --pdf path/to/plan.pdf --output bom.xlsx
```

Or use the UI:
```bash
streamlit run src/ui/app.py
```

## Input/Output

- **Input**: Vector-based PDF plans (exported from DWG)
- **Output**: 
  - Excel BOM with normalized items and counts
  - JSON with coordinates, page references, confidence, evidence
  - UI for hierarchy review and multiplier visualization

