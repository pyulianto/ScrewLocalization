# Setup Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   
   For OpenAI:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
   
   Or for Anthropic:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **Process a PDF:**
   ```bash
   python main.py --pdf path/to/your/plan.pdf
   ```

4. **View results in UI:**
   ```bash
   streamlit run src/ui/app.py
   ```
   Then upload the generated JSON file from the `output/` directory.

## Configuration

Edit `config.yaml` to customize:
- Detection parameters (circle sizes, confidence thresholds)
- LLM provider and model
- Part classes and spec fields
- Accuracy targets

## Architecture

The system has three main layers:

1. **Perception** (`src/perception/`): Detects fasteners using vector-assisted detection
2. **Structure Reasoning** (`src/structure/`): Uses LLM to build hierarchy and assign multipliers
3. **Aggregation** (`src/aggregation/`): Normalizes specs and builds final BOM

## Outputs

- **Excel BOM** (`*_bom.xlsx`): Three sheets:
  - BOM: Main bill of materials with multipliers
  - Hierarchy: Page/view hierarchy tree
  - Evidence: Detection evidence tracking

- **JSON Results** (`*_results.json`): Complete data for UI review including:
  - BOM items with all metadata
  - Detection coordinates and confidence
  - Hierarchy structure

## Testing

Run the example script to test the pipeline:
```bash
python example_usage.py
```

## Troubleshooting

- **No detections found**: This is expected if the PDF doesn't contain recognizable fastener patterns. The system will still create a hierarchy structure.
- **LLM errors**: Check your API key and ensure you have credits. The system will use a fallback hierarchy if LLM fails.
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

