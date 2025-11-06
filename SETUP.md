# Setup Guide

Complete step-by-step instructions for setting up and running the Screw Localization System.

## 📋 Prerequisites

Before you begin, ensure you have:
- **Python 3.8 or higher** (check with `python --version`)
- **Git** installed
- **Internet connection** (for installing packages)
- **Optional**: OpenAI or Anthropic API key for enhanced hierarchy detection

## 🛠️ Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ScrewLocalization.git
cd ScrewLocalization
```

**Note:** Replace `YOUR_USERNAME` with the actual GitHub username.

### Step 2: Create a Virtual Environment (Recommended)

This keeps your project dependencies isolated:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages. It may take a few minutes.

**Expected output:** You should see packages being downloaded and installed. If you see errors, see Troubleshooting below.

### Step 4: Configure API Key (Optional but Recommended)

The system works without an API key, but LLM-powered hierarchy detection provides better multiplier detection.

#### For Windows (PowerShell):
```powershell
# OpenAI
$env:OPENAI_API_KEY="sk-your-key-here"

# Or Anthropic
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

#### For Windows (CMD):
```cmd
# OpenAI
set OPENAI_API_KEY=sk-your-key-here

# Or Anthropic
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

#### For Linux/Mac:
```bash
# OpenAI
export OPENAI_API_KEY="sk-your-key-here"

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**To make it permanent:**
- **Windows**: Add to System Environment Variables
- **Linux/Mac**: Add to `~/.bashrc` or `~/.zshrc`:
  ```bash
  echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
  source ~/.bashrc
  ```

### Step 5: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import streamlit; import fitz; print('✓ All dependencies installed!')"
```

If you see the checkmark, you're ready to go!

## 🚀 Running the Application

### Method 1: Web UI (Easiest - Recommended)

1. **Start the Streamlit app:**
   ```bash
   streamlit run src/ui/app.py
   ```

2. **Open your browser:**
   - The terminal will show a URL (usually `http://localhost:8501`)
   - Copy and paste it into your browser
   - Or click the link if your terminal supports it

3. **Use the application:**
   - Upload a PDF file using the sidebar
   - Click "🚀 Process PDF"
   - Wait for processing (progress will be shown)
   - View results in the tabs
   - Download Excel and JSON files

### Method 2: Command Line

```bash
python main.py --pdf path/to/your/plan.pdf --output output
```

**Options:**
- `--pdf`: Path to your PDF file (required)
- `--output`: Output directory (default: `output`)
- `--config`: Path to config file (default: `config.yaml`)

**Example:**
```bash
python main.py --pdf "C:\Users\YourName\Documents\construction_plan.pdf"
```

Output files will be in the `output/` directory:
- `construction_plan_bom.xlsx` - Excel BOM
- `construction_plan_results.json` - JSON data

### Method 3: Test Detection Only

For testing without API key or full processing:

```bash
streamlit run src/ui/test_detection.py
```

This only runs Steps 1 & 2 (PDF processing and detection).

## 📁 Project Structure

```
ScrewLocalization/
├── src/
│   ├── perception/          # PDF processing and detection
│   ├── structure/            # Hierarchy building (LLM)
│   ├── aggregation/          # BOM generation
│   ├── export/               # Excel/JSON export
│   └── ui/                   # Streamlit web interface
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── main.py                   # Command-line entry point
├── README.md                 # Main documentation
├── SETUP.md                  # This file
├── QUICKSTART.md             # Quick start guide
└── ARCHITECTURE.md           # System architecture
```

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
detection:
  min_circle_radius: 0.5      # Minimum circle size (mm)
  max_circle_radius: 5.0      # Maximum circle size (mm)
  min_confidence: 0.5         # Minimum detection confidence
  max_lines_for_thread_detection: 5000  # Limit for performance
  enable_thread_detection: true  # Set false to skip (faster)

llm:
  provider: "openai"          # or "anthropic"
  model: "gpt-4-turbo-preview"
  temperature: 0.1
```

## 📊 Understanding the Output

### Excel BOM (`*_bom.xlsx`)

Three sheets:
1. **BOM**: Main bill of materials
   - Item specifications (thread, length, brand)
   - Per-instance count and multiplier
   - Total count
   - Confidence scores
   - Review flags

2. **Hierarchy**: Page/view structure
   - Parent-child relationships
   - Multipliers per node
   - Page assignments

3. **Evidence**: Detection tracking
   - Source nodes
   - Detection IDs
   - Evidence references

### JSON Results (`*_results.json`)

Complete data structure:
- All BOM items with full metadata
- Detection coordinates and confidence
- Hierarchy structure
- Evidence references

Use this file to review results in the UI.

## 🐛 Troubleshooting

### Issue: "No module named 'src'"

**Solution:**
```bash
# Make sure you're in the project root
cd ScrewLocalization
python main.py --pdf your_file.pdf
```

### Issue: "OPENAI_API_KEY environment variable not set"

**This is a warning, not an error!**
- The system will work with a fallback hierarchy
- Multipliers may not be detected from callouts
- To enable full features, set the API key (see Step 4)

### Issue: Processing is very slow

**Solutions:**
1. **Disable thread detection** (faster but less accurate):
   ```yaml
   # In config.yaml
   detection:
     enable_thread_detection: false
   ```

2. **Reduce line limit**:
   ```yaml
   detection:
     max_lines_for_thread_detection: 2000  # Lower = faster
   ```

3. **Wait for progress updates** - The system shows detailed progress

### Issue: "No detections found"

**This is normal if:**
- PDF doesn't contain recognizable fastener patterns
- PDF is scanned (not vector-based)
- Fasteners are drawn in unusual styles

**The system will still:**
- Process all pages
- Create hierarchy structure
- Generate BOM (may be empty)

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Streamlit won't start

**Solutions:**
1. Check if port 8501 is in use:
   ```bash
   # Windows
   netstat -ano | findstr :8501
   
   # Linux/Mac
   lsof -i :8501
   ```

2. Use a different port:
   ```bash
   streamlit run src/ui/app.py --server.port 8502
   ```

### Issue: PDF processing fails

**Check:**
1. PDF is vector-based (not scanned)
2. PDF is not corrupted
3. You have read permissions
4. File path is correct (use quotes if path has spaces)

## 📝 Example Workflow

1. **Prepare your PDF:**
   - Ensure it's vector-based (exported from DWG/CAD)
   - Not a scanned image

2. **Run the application:**
   ```bash
   streamlit run src/ui/app.py
   ```

3. **Process PDF:**
   - Upload PDF in the UI
   - Click "Process PDF"
   - Wait for completion (watch progress)

4. **Review results:**
   - Check BOM tab for items
   - Review Hierarchy tab for structure
   - Check Multipliers tab for impact
   - Review Detections tab for statistics

5. **Download results:**
   - Click download buttons in sidebar
   - Excel file for spreadsheet use
   - JSON file for programmatic access

## 🔧 Advanced Configuration

### Custom Detection Parameters

Edit `config.yaml`:
```yaml
detection:
  min_circle_radius: 0.3      # Smaller circles
  max_circle_radius: 8.0      # Larger circles
  min_confidence: 0.6         # Higher confidence threshold
```

### Using Different LLM Models

```yaml
llm:
  provider: "openai"
  model: "gpt-4"              # or "gpt-3.5-turbo"
  temperature: 0.1
```

### Performance Tuning

For large PDFs:
```yaml
detection:
  max_lines_for_thread_detection: 3000  # Reduce for speed
  enable_thread_detection: false        # Disable for maximum speed
```

## 📞 Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Review error messages carefully
3. Check that all dependencies are installed
4. Verify your PDF is vector-based
5. Try the test detection UI first

## ✅ Verification Checklist

Before sharing with colleagues, verify:
- [ ] Repository cloned successfully
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Can run `streamlit run src/ui/app.py`
- [ ] UI opens in browser
- [ ] Can process a test PDF
- [ ] Output files are generated
- [ ] API key set (if using LLM features)

## 🎯 Next Steps

After setup:
1. Try processing a sample PDF
2. Review the output Excel file
3. Explore the UI features
4. Read ARCHITECTURE.md to understand the system
5. Customize config.yaml for your needs
