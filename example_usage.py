"""Example usage script for testing the system."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ScrewLocalizationPipeline


def main():
    """Example usage of the pipeline."""
    print("=" * 60)
    print("Screw Localization System - Example Usage")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[WARNING] No LLM API key found!")
        print("   Set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("   The system will still run but hierarchy building may fail.")
        print()
    
    # Example PDF path (user should provide their own)
    pdf_path = input("Enter path to PDF file (or press Enter to skip): ").strip()
    
    if not pdf_path or not Path(pdf_path).exists():
        print("\n[INFO] No valid PDF provided.")
        print("\nTo use the system:")
        print("  1. Place your PDF in the project directory")
        print("  2. Run: python main.py --pdf your_file.pdf")
        print("  3. Or use this script with a valid PDF path")
        return
    
    # Initialize pipeline
    try:
        pipeline = ScrewLocalizationPipeline()
        
        # Process PDF
        results = pipeline.process_pdf(pdf_path, output_dir="output")
        
        print("\n[SUCCESS] Processing complete!")
        print(f"\nResults:")
        print(f"  - Excel BOM: {results['excel_path']}")
        print(f"  - JSON data: {results['json_path']}")
        print(f"\nTo view results in UI:")
        print(f"  streamlit run src/ui/app.py")
        print(f"  Then upload: {results['json_path']}")
        
    except Exception as e:
        print(f"\n[ERROR] Error processing PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

