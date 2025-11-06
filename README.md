# Screw Localization System

A system to extract small standard parts (screws, rivets, anchors, washers, nuts) from vector PDF plans and generate an accurate BOM with hierarchical multipliers.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ScrewLocalization.git
   cd ScrewLocalization
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key (Optional but recommended):**
   
   For **OpenAI**:
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Windows (CMD)
   set OPENAI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   For **Anthropic** (alternative):
   ```bash
   # Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   
   # Windows (CMD)
   set ANTHROPIC_API_KEY=your-api-key-here
   
   # Linux/Mac
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   
   **Note:** The system works without an API key but will use a fallback hierarchy (multipliers may not be detected).

### Running the Application

#### Option 1: Web UI (Recommended)

```bash
streamlit run src/ui/app.py
```

Then:
1. Open your browser to the URL shown (usually `http://localhost:8501`)
2. Upload a PDF file in the sidebar
3. Click "🚀 Process PDF"
4. View results and download Excel/JSON files

#### Option 2: Command Line

```bash
python main.py --pdf path/to/your/plan.pdf
```

This will:
- Process the PDF
- Generate Excel BOM in `output/` directory
- Generate JSON results in `output/` directory

#### Option 3: Test Detection Only (No API key needed)

```bash
streamlit run src/ui/test_detection.py
```

This tests only Steps 1 & 2 (PDF processing and detection) without requiring an API key.

## 📋 What You Need

- **Input**: Vector-based PDF files (exported from DWG, not scanned images)
- **Output**: 
  - Excel BOM with normalized items and counts
  - JSON with coordinates, page references, confidence, evidence
  - Interactive UI for review

## 🏗️ Architecture

Three-layer system:
1. **Perception**: Detect candidate fasteners on each page using vector-assisted detection
2. **Structure Reasoning (LLM)**: Build page hierarchy, infer multipliers, reconcile duplicates
3. **Aggregation & QC**: Merge instances, normalize specs, generate Excel BOM

## 📖 Detailed Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup instructions and troubleshooting
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design decisions

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Detection parameters (circle sizes, confidence thresholds)
- LLM provider and model settings
- Part classes and spec fields
- Accuracy targets

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'src'"**
   - Make sure you're in the project root directory
   - Run: `cd ScrewLocalization`

2. **"OPENAI_API_KEY environment variable not set"**
   - This is a warning, not an error
   - The system will work with fallback hierarchy
   - To enable LLM features, set the API key (see Setup above)

3. **Processing is slow**
   - Large PDFs with many vector primitives take time
   - Progress updates show what's happening
   - You can disable thread detection in `config.yaml`:
     ```yaml
     detection:
       enable_thread_detection: false
     ```

4. **No detections found**
   - This is normal for test PDFs
   - The system will still create a hierarchy structure
   - Check that your PDF is vector-based (not scanned)

## 📝 Example Usage

```bash
# Process a PDF
python main.py --pdf construction_plan.pdf

# Output will be in:
# - output/construction_plan_bom.xlsx
# - output/construction_plan_results.json

# View in UI
streamlit run src/ui/app.py
# Then upload the JSON file to review results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license here]

## 👥 Authors

[Add author information]

