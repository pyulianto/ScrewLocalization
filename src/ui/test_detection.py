"""Streamlit UI for testing Step 1 (PDF Processing) and Step 2 (Detection) only."""
import streamlit as st
import os
import tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.perception.pdf_processor import PDFProcessor
from src.perception.detector import VectorAssistedDetector
import yaml


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        "detection": {
            "min_circle_radius": 0.5,
            "max_circle_radius": 5.0,
            "thread_pattern_threshold": 0.3,
            "min_confidence": 0.5
        }
    }


def display_page_overview(page_data, detections_on_page):
    """Display page overview with statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Page Number", page_data.page_num + 1)
    with col2:
        st.metric("Page Size", f"{page_data.width:.1f} × {page_data.height:.1f}")
    with col3:
        st.metric("Vector Primitives", len(page_data.vector_primitives))
    with col4:
        st.metric("Text Blocks", len(page_data.text_blocks))
    
    if page_data.scale:
        st.info(f"Scale detected: {page_data.scale}")
    if page_data.title:
        st.write(f"**Title:** {page_data.title}")
    
    st.write(f"**Detections on this page:** {len(detections_on_page)}")


def display_detections_table(detections):
    """Display detections in a table."""
    if not detections:
        st.info("No detections found on this page.")
        return
    
    df_data = []
    for i, det in enumerate(detections):
        df_data.append({
            "ID": i + 1,
            "Category": det.category,
            "Confidence": f"{det.confidence:.2f}",
            "Center X": f"{det.center[0]:.1f}",
            "Center Y": f"{det.center[1]:.1f}",
            "Thread": det.spec_fields.get("thread", "") if det.spec_fields else "",
            "Length": det.spec_fields.get("length", "") if det.spec_fields else "",
            "Brand": det.spec_fields.get("brand_code", "") if det.spec_fields else "",
            "Nearby Text": ", ".join(det.nearby_text[:2]) if det.nearby_text else ""
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_detection_statistics(all_detections):
    """Display overall detection statistics."""
    if not all_detections:
        return
    
    st.subheader("Detection Statistics")
    
    # Count by category
    category_counts = {}
    for det in all_detections:
        cat = det.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detections by Category:**")
        category_df = pd.DataFrame({
            "Category": list(category_counts.keys()),
            "Count": list(category_counts.values())
        })
        st.dataframe(category_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Pie chart
        if category_counts:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(category_counts.keys()),
                    values=list(category_counts.values()),
                    hole=0.3
                )
            ])
            fig.update_layout(title="Category Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    confidences = [det.confidence for det in all_detections]
    if confidences:
        st.write("**Confidence Distribution:**")
        fig = go.Figure(data=[
            go.Histogram(x=confidences, nbinsx=20, marker_color="steelblue")
        ])
        fig.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Number of Detections",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(all_detections))
        with col2:
            st.metric("Avg Confidence", f"{np.mean(confidences):.2f}")
        with col3:
            st.metric("Min Confidence", f"{np.min(confidences):.2f}")


def display_vector_primitives(page_data):
    """Display information about vector primitives."""
    st.subheader("Vector Primitives Analysis")
    
    # Count by type
    primitive_types = {}
    for prim in page_data.vector_primitives:
        prim_type = prim.get("type", "unknown")
        primitive_types[prim_type] = primitive_types.get(prim_type, 0) + 1
    
    if primitive_types:
        st.write("**Primitives by Type:**")
        prim_df = pd.DataFrame({
            "Type": list(primitive_types.keys()),
            "Count": list(primitive_types.values())
        })
        st.dataframe(prim_df, use_container_width=True, hide_index=True)
        
        # Show some circles if found
        circles = [p for p in page_data.vector_primitives if p.get("type") == "circle"]
        if circles:
            st.write(f"**Found {len(circles)} circles (potential fastener heads):**")
            circle_data = []
            for i, circle in enumerate(circles[:10]):  # Show first 10
                radius = circle.get("radius", 0)
                center = circle.get("center", (0, 0))
                circle_data.append({
                    "Index": i + 1,
                    "Radius": f"{radius:.2f}",
                    "Center X": f"{center[0]:.1f}",
                    "Center Y": f"{center[1]:.1f}"
                })
            if circle_data:
                st.dataframe(pd.DataFrame(circle_data), use_container_width=True, hide_index=True)


def main():
    """Main Streamlit app for testing detection."""
    st.set_page_config(
        page_title="Detection Tester - Steps 1 & 2",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Detection Tester - Steps 1 & 2")
    st.markdown("Test PDF processing and fastener detection without LLM or BOM aggregation")
    
    # Initialize session state
    if "page_data_list" not in st.session_state:
        st.session_state.page_data_list = None
    if "all_detections" not in st.session_state:
        st.session_state.all_detections = None
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = 0
    
    # Sidebar
    st.sidebar.header("Upload PDF")
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload PDF file",
        type=["pdf"],
        help="Upload a vector-based PDF to test detection"
    )
    
    if uploaded_pdf:
        st.sidebar.write(f"**File:** {uploaded_pdf.name}")
        st.sidebar.write(f"**Size:** {uploaded_pdf.size / 1024:.1f} KB")
        
        if st.sidebar.button("🚀 Process PDF (Steps 1 & 2)", type="primary", use_container_width=True):
            with st.spinner("Processing PDF... This may take a minute."):
                try:
                    # Save uploaded PDF to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_pdf.read())
                        tmp_path = tmp_file.name
                    
                    # Load config
                    config = load_config()
                    
                    # STEP 1: Process PDF
                    st.write("### Step 1: PDF Processing")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Extracting pages and vector data...")
                    page_data_list = []
                    
                    with PDFProcessor(tmp_path) as processor:
                        page_count = processor.get_page_count()
                        st.info(f"Found {page_count} page(s) in PDF")
                        
                        for page_num in range(page_count):
                            page_data = processor.process_page(page_num)
                            page_data_list.append(page_data)
                            progress_bar.progress((page_num + 1) / page_count)
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✓ Processed {len(page_data_list)} pages")
                    st.success(f"Step 1 Complete: Extracted {len(page_data_list)} pages")
                    
                    # STEP 2: Detect fasteners
                    st.write("### Step 2: Fastener Detection")
                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()
                    
                    status_text2.text("Detecting fasteners on each page...")
                    detector = VectorAssistedDetector(config["detection"])
                    all_detections = []
                    
                    for i, page_data in enumerate(page_data_list):
                        detections = detector.detect(page_data)
                        # Assign detection IDs
                        for j, det in enumerate(detections):
                            det.detection_id = f"det_{len(all_detections)}"
                        all_detections.extend(detections)
                        progress_bar2.progress((i + 1) / len(page_data_list))
                    
                    progress_bar2.progress(1.0)
                    status_text2.text(f"✓ Found {len(all_detections)} detections")
                    st.success(f"Step 2 Complete: Found {len(all_detections)} candidate detections")
                    
                    # Store in session state
                    st.session_state.page_data_list = page_data_list
                    st.session_state.all_detections = all_detections
                    st.session_state.selected_page = 0
                    
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.page_data_list and st.session_state.all_detections:
        page_data_list = st.session_state.page_data_list
        all_detections = st.session_state.all_detections
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Page selector
        page_names = [f"Page {i+1}" + (f" - {page.title[:30]}" if page.title else "") 
                     for i, page in enumerate(page_data_list)]
        
        selected_page_idx = st.selectbox(
            "Select Page to View",
            range(len(page_data_list)),
            format_func=lambda x: page_names[x],
            index=st.session_state.selected_page
        )
        
        st.session_state.selected_page = selected_page_idx
        
        # Get selected page data
        selected_page_data = page_data_list[selected_page_idx]
        detections_on_page = [det for det in all_detections if det.page_num == selected_page_idx]
        
        # Display page overview
        display_page_overview(selected_page_data, detections_on_page)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Detections", "Vector Primitives", "Statistics"])
        
        with tab1:
            st.subheader(f"Detections on Page {selected_page_idx + 1}")
            display_detections_table(detections_on_page)
            
            # Show detection details
            if detections_on_page:
                st.write("### Detection Details")
                for i, det in enumerate(detections_on_page[:5]):  # Show first 5
                    with st.expander(f"Detection {i+1}: {det.category} (Confidence: {det.confidence:.2f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Category:** {det.category}")
                            st.write(f"**Confidence:** {det.confidence:.2f}")
                            st.write(f"**Center:** ({det.center[0]:.1f}, {det.center[1]:.1f})")
                            st.write(f"**BBox:** {det.bbox}")
                        with col2:
                            if det.spec_fields:
                                st.write("**Specifications:**")
                                for key, value in det.spec_fields.items():
                                    st.write(f"- {key}: {value}")
                            if det.nearby_text:
                                st.write("**Nearby Text:**")
                                for text in det.nearby_text[:3]:
                                    st.write(f"- {text[:50]}")
                            if det.vector_cues:
                                st.write("**Vector Cues:**")
                                for key, value in det.vector_cues.items():
                                    st.write(f"- {key}: {value}")
        
        with tab2:
            display_vector_primitives(selected_page_data)
        
        with tab3:
            display_detection_statistics(all_detections)
    
    else:
        # Welcome screen
        st.info("👈 Upload a PDF file to test Steps 1 & 2")
        st.markdown("""
        ### What this tool tests:
        
        **Step 1: PDF Processing**
        - Extracts pages from PDF
        - Extracts vector primitives (circles, lines, paths)
        - Extracts text blocks with positions
        - Renders pages as raster images
        
        **Step 2: Fastener Detection**
        - Detects circles (potential fastener heads)
        - Finds thread patterns (parallel lines)
        - Extracts specifications from nearby text
        - Classifies fasteners (screw, rivet, anchor, washer, nut)
        - Assigns confidence scores
        
        ### No API key needed!
        This tool only tests the perception layer - no LLM required.
        """)


if __name__ == "__main__":
    main()

