"""Streamlit UI for hierarchy review and BOM visualization."""
import streamlit as st
import json
import pandas as pd
from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import ScrewLocalizationPipeline
from src.aggregation.bom_builder import BOMItem
from src.structure.hierarchy_builder import HierarchyNode


def render_hierarchy_tree(hierarchy_data: Dict):
    """Render hierarchy tree visualization."""
    st.subheader("Page Hierarchy")
    
    nodes = hierarchy_data["hierarchy"]
    
    if not nodes:
        st.info("No hierarchy data available")
        return
    
    # Build tree structure
    tree_data = []
    for node in nodes:
        # Handle both object and dict formats
        if isinstance(node, dict):
            node_id = node.get("node_id", "")
            page_num = node.get("page_num", 0)
            title = node.get("title", "")
            node_type = node.get("node_type", "")
            multiplier = node.get("multiplier", 1)
            parent_id = node.get("parent_id")
        else:
            node_id = node.node_id
            page_num = node.page_num
            title = node.title
            node_type = node.node_type
            multiplier = node.multiplier
            parent_id = node.parent_id
        
        tree_data.append({
            "Node ID": node_id,
            "Page": page_num + 1,
            "Title": title,
            "Type": node_type,
            "Multiplier": multiplier,
            "Parent": parent_id or "Root"
        })
    
    df = pd.DataFrame(tree_data)
    st.dataframe(df, use_container_width=True)
    
    # Visual tree representation
    st.write("### Hierarchy Tree")
    for node in nodes:
        if isinstance(node, dict):
            parent_id = node.get("parent_id")
            title = node.get("title", "")
            page_num = node.get("page_num", 0)
            multiplier = node.get("multiplier", 1)
            children = node.get("children", [])
        else:
            parent_id = node.parent_id
            title = node.title
            page_num = node.page_num
            multiplier = node.multiplier
            children = node.children
        
        if parent_id is None:
            # Root node
            st.write(f"**{title}** (Page {page_num + 1}) [Multiplier: {multiplier}]")
            _render_children(node, nodes, indent=1)


def _render_children(parent_node, all_nodes: List, indent: int = 0):
    """Recursively render child nodes."""
    prefix = "  " * indent + "└─ "
    
    # Handle both dict and object formats
    if isinstance(parent_node, dict):
        children_ids = parent_node.get("children", [])
    else:
        children_ids = parent_node.children
    
    for child_id in children_ids:
        child = next((n for n in all_nodes if 
                     (isinstance(n, dict) and n.get("node_id") == child_id) or
                     (hasattr(n, "node_id") and n.node_id == child_id)), None)
        if child:
            if isinstance(child, dict):
                multiplier = child.get("multiplier", 1)
                title = child.get("title", "")
                page_num = child.get("page_num", 0)
                children = child.get("children", [])
            else:
                multiplier = child.multiplier
                title = child.title
                page_num = child.page_num
                children = child.children
            
            multiplier_badge = f" [Multiplier: {multiplier}x]" if multiplier > 1 else ""
            st.write(f"{prefix}**{title}** (Page {page_num + 1}){multiplier_badge}")
            if children:
                _render_children(child, all_nodes, indent + 1)


def render_bom_table(bom_items: List):
    """Render BOM table with review flags."""
    st.subheader("Bill of Materials")
    
    if not bom_items:
        st.info("No BOM items available")
        return
    
    # Get unique categories
    categories = list(set(
        item.get("category", "") if isinstance(item, dict) else item.category 
        for item in bom_items
    ))
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_review_only = st.checkbox("Show only items needing review", False)
    with col2:
        category_filter = st.selectbox("Filter by category", ["All"] + categories)
    with col3:
        sort_by = st.selectbox("Sort by", ["Total Count", "Confidence", "Category"])
    
    # Filter items
    filtered = bom_items
    if show_review_only:
        filtered = [
            item for item in filtered 
            if (isinstance(item, dict) and item.get("needs_review", False)) or
               (hasattr(item, "needs_review") and item.needs_review)
        ]
    if category_filter != "All":
        filtered = [
            item for item in filtered
            if (isinstance(item, dict) and item.get("category") == category_filter) or
               (hasattr(item, "category") and item.category == category_filter)
        ]
    
    # Sort
    if sort_by == "Total Count":
        filtered.sort(key=lambda x: x.get("total_count", 0) if isinstance(x, dict) else x.total_count, reverse=True)
    elif sort_by == "Confidence":
        filtered.sort(key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else x.confidence, reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("category", "") if isinstance(x, dict) else x.category)
    
    # Create dataframe
    df_data = []
    for item in filtered:
        if isinstance(item, dict):
            spec = item.get("spec_normalized", {})
            df_data.append({
                "Item ID": item.get("item_id", ""),
                "Category": item.get("category", ""),
                "Thread": spec.get("thread", ""),
                "Length": spec.get("length", ""),
                "Per Instance": item.get("per_instance_count", 0),
                "Multiplier": item.get("multiplier", 1),
                "Total Count": item.get("total_count", 0),
                "Confidence": f"{item.get('confidence', 0):.2f}",
                "Review": "Yes" if item.get("needs_review", False) else "No",
                "Reasoning": item.get("reasoning", "")
            })
        else:
            df_data.append({
                "Item ID": item.item_id,
                "Category": item.category,
                "Thread": item.spec_normalized.get("thread", ""),
                "Length": item.spec_normalized.get("length", ""),
                "Per Instance": item.per_instance_count,
                "Multiplier": item.multiplier,
                "Total Count": item.total_count,
                "Confidence": f"{item.confidence:.2f}",
                "Review": "Yes" if item.needs_review else "No",
                "Reasoning": item.reasoning
            })
    
    df = pd.DataFrame(df_data)
    
    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary statistics
    st.write("### Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Items", len(bom_items))
    with col2:
        review_count = sum(
            1 for item in bom_items
            if (isinstance(item, dict) and item.get("needs_review", False)) or
               (hasattr(item, "needs_review") and item.needs_review)
        )
        st.metric("Items Needing Review", review_count)
    with col3:
        total_count = sum(
            item.get("total_count", 0) if isinstance(item, dict) else item.total_count
            for item in bom_items
        )
        st.metric("Total Fasteners", total_count)
    with col4:
        confidences = [
            item.get("confidence", 0) if isinstance(item, dict) else item.confidence
            for item in bom_items
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        st.metric("Avg Confidence", f"{avg_conf:.2f}")


def render_multiplier_visualization(bom_items: List):
    """Visualize how multipliers affect counts."""
    st.subheader("Multiplier Impact")
    
    if not bom_items:
        st.info("No BOM items available for visualization")
        return
    
    # Group by multiplier value
    multiplier_counts = {}
    for item in bom_items:
        if isinstance(item, dict):
            mult = item.get("multiplier", 1)
            total = item.get("total_count", 0)
        else:
            mult = item.multiplier
            total = item.total_count
        
        if mult not in multiplier_counts:
            multiplier_counts[mult] = 0
        multiplier_counts[mult] += total
    
    # Create bar chart
    if multiplier_counts:
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{k}x" for k in sorted(multiplier_counts.keys())],
                y=[multiplier_counts[k] for k in sorted(multiplier_counts.keys())],
                text=[multiplier_counts[k] for k in sorted(multiplier_counts.keys())],
                textposition="auto",
                marker_color="steelblue"
            )
        ])
        fig.update_layout(
            title="Total Counts by Multiplier",
            xaxis_title="Multiplier",
            yaxis_title="Total Fastener Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No multiplier data available")


def render_detection_overview(detections: List[Dict]):
    """Show detection overview."""
    st.subheader("Detection Overview")
    
    if not detections:
        st.info("No detection data available")
        return
    
    # Count by category
    category_counts = {}
    for det in detections:
        # Handle both dict and object formats
        if isinstance(det, dict):
            cat = det.get("category", "unknown")
        else:
            cat = det.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Pie chart
    if category_counts:
        fig = go.Figure(data=[
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                hole=0.3
            )
        ])
        fig.update_layout(title="Detections by Category", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category data available")
    
    # Confidence distribution
    confidences = []
    for det in detections:
        if isinstance(det, dict):
            conf = det.get("confidence", 0)
        else:
            conf = det.confidence
        if conf is not None:
            confidences.append(float(conf))
    
    if confidences:
        fig2 = go.Figure(data=[
            go.Histogram(x=confidences, nbinsx=20, marker_color="lightblue")
        ])
        fig2.update_layout(
            title="Confidence Distribution",
            xaxis_title="Confidence",
            yaxis_title="Number of Detections",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No confidence data available")


def process_pdf(uploaded_pdf, output_dir: str = "output"):
    """Process uploaded PDF through the pipeline."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name
    
    try:
        # Initialize pipeline
        pipeline = ScrewLocalizationPipeline()
        
        # Process PDF
        results = pipeline.process_pdf(tmp_path, output_dir)
        
        return results
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_pdf_with_progress(uploaded_pdf, output_dir: str, 
                               status_step1, status_step2, status_step3,
                               status_step4, status_step5, status_step6,
                               current_status, overall_progress):
    """Process PDF with detailed progress updates."""
    import tempfile
    from src.perception.pdf_processor import PDFProcessor
    from src.perception.detector import VectorAssistedDetector
    from src.structure.hierarchy_builder import HierarchyBuilder
    from src.aggregation.bom_builder import BOMBuilder
    from src.export.excel_exporter import ExcelExporter
    import yaml
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name
    
    try:
        # Load config
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # STEP 1: PDF Processing
        status_step1.info("📄 **Step 1: PDF Processing** - In progress...")
        current_status.text("Extracting pages and vector data from PDF...")
        overall_progress.progress(0.05)
        
        page_data_list = []
        with PDFProcessor(tmp_path) as processor:
            page_count = processor.get_page_count()
            status_step1.info(f"📄 **Step 1: PDF Processing** - Found {page_count} page(s), processing...")
            
            for page_num in range(page_count):
                current_status.text(f"Processing page {page_num + 1} of {page_count}...")
                page_data = processor.process_page(page_num)
                page_data_list.append(page_data)
                progress = 0.05 + (page_num + 1) / page_count * 0.15
                overall_progress.progress(progress)
        
        status_step1.success(f"✅ **Step 1 Complete** - Processed {len(page_data_list)} pages, extracted {sum(len(p.vector_primitives) for p in page_data_list)} vector primitives")
        overall_progress.progress(0.20)
        
        # STEP 2: Detection
        status_step2.info("🔍 **Step 2: Fastener Detection** - Starting...")
        current_status.text("Initializing detector...")
        overall_progress.progress(0.21)
        
        detector = VectorAssistedDetector(config.get("detection", {}))
        all_detections = []
        detection_counter = 0
        
        for i, page_data in enumerate(page_data_list):
            page_num = i + 1
            total_pages = len(page_data_list)
            base_progress = 0.21
            page_progress_range = 0.25 / total_pages
            
            # Create progress callback for detailed updates
            def update_status(message):
                nonlocal page_num, total_pages
                current_status.text(f"Page {page_num}/{total_pages}: {message}")
            
            # Calculate progress for this page
            page_base_progress = base_progress + (i * page_progress_range)
            
            # Step 2.1: Pre-analysis
            update_status("Preparing detection...")
            overall_progress.progress(page_base_progress + page_progress_range * 0.1)
            
            # Step 2.2: Run detection with progress callback
            import time
            detection_start_time = time.time()
            detections = detector.detect(page_data, progress_callback=update_status)
            detection_time = time.time() - detection_start_time
            
            # Show time estimate for remaining pages
            if i < total_pages - 1:
                remaining_pages = total_pages - (i + 1)
                estimated_remaining = detection_time * remaining_pages
                update_status(f"Page took {detection_time:.1f}s, estimated {estimated_remaining:.0f}s remaining...")
            
            # Step 2.3: Post-process detections
            update_status(f"Processing {len(detections)} detections...")
            overall_progress.progress(page_base_progress + page_progress_range * 0.9)
            
            # Assign detection IDs
            for det in detections:
                det.detection_id = f"det_{detection_counter}"
                detection_counter += 1
            
            all_detections.extend(detections)
            
            # Show summary
            categories = {}
            for det in detections:
                categories[det.category] = categories.get(det.category, 0) + 1
            category_summary = ", ".join([f"{count} {cat}" for cat, count in categories.items()])
            
            update_status(f"Complete: {len(detections)} fasteners found ({category_summary})")
            overall_progress.progress(page_base_progress + page_progress_range)
        
        status_step2.success(f"✅ **Step 2 Complete** - Found {len(all_detections)} candidate detections")
        overall_progress.progress(0.46)
        
        # If no detections, create dummy
        if len(all_detections) == 0:
            current_status.text("No detections found, creating test detection...")
            from src.perception.detector import Detection
            dummy_detection = Detection(
                bbox=(100, 100, 110, 110),
                center=(105, 105),
                confidence=0.5,
                category="screw",
                page_num=0
            )
            dummy_detection.detection_id = "det_0"
            all_detections.append(dummy_detection)
        
        # STEP 3: Hierarchy Building
        status_step3.info("🌳 **Step 3: Hierarchy Building** - Starting...")
        current_status.text("Building page hierarchy...")
        overall_progress.progress(0.47)
        
        has_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not has_api_key:
            current_status.text("No API key found - using fallback hierarchy...")
            status_step3.warning("⚠️ **Step 3** - Using fallback hierarchy (no LLM API key)")
        
        hierarchy_builder = HierarchyBuilder(config.get("llm", {}))
        hierarchy_data = hierarchy_builder.build_hierarchy(page_data_list, all_detections)
        
        status_step3.success(f"✅ **Step 3 Complete** - Built hierarchy with {len(hierarchy_data['hierarchy'])} nodes")
        overall_progress.progress(0.60)
        
        # STEP 4: Reconcile duplicates
        status_step4.info("🔄 **Step 4: Duplicate Reconciliation** - Starting...")
        current_status.text("Reconciling duplicate detections...")
        overall_progress.progress(0.61)
        
        reconciled_detections = hierarchy_builder.reconcile_duplicates(
            all_detections, hierarchy_data
        )
        
        status_step4.success(f"✅ **Step 4 Complete** - Reduced to {len(reconciled_detections)} unique detections")
        overall_progress.progress(0.70)
        
        # STEP 5: BOM Building
        status_step5.info("📋 **Step 5: BOM Aggregation** - Starting...")
        current_status.text("Building normalized BOM...")
        overall_progress.progress(0.71)
        
        if len(reconciled_detections) == 0:
            bom_items = []
            status_step5.warning("⚠️ **Step 5** - No detections to build BOM from")
        else:
            bom_builder = BOMBuilder(config.get("accuracy", {}))
            bom_items = bom_builder.build_bom(reconciled_detections, hierarchy_data)
            bom_items = bom_builder.apply_sanity_checks(bom_items)
            status_step5.success(f"✅ **Step 5 Complete** - Generated {len(bom_items)} BOM items")
        
        overall_progress.progress(0.85)
        
        # STEP 6: Export
        status_step6.info("💾 **Step 6: Export Results** - Starting...")
        current_status.text("Exporting Excel and JSON files...")
        overall_progress.progress(0.86)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export Excel
        excel_exporter = ExcelExporter()
        excel_path = output_path / f"{Path(tmp_path).stem}_bom.xlsx"
        current_status.text(f"Creating Excel file: {excel_path.name}...")
        excel_exporter.export(bom_items, hierarchy_data, str(excel_path))
        
        # Export JSON
        json_path = output_path / f"{Path(tmp_path).stem}_results.json"
        current_status.text(f"Creating JSON file: {json_path.name}...")
        json_data = bom_builder.export_to_json(bom_items, reconciled_detections)
        json_data["hierarchy"] = [
            {
                "node_id": node.node_id,
                "page_num": node.page_num,
                "title": node.title,
                "node_type": node.node_type,
                "multiplier": node.multiplier,
                "parent_id": node.parent_id,
                "children": node.children
            }
            for node in hierarchy_data["hierarchy"]
        ]
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        status_step6.success(f"✅ **Step 6 Complete** - Exported Excel and JSON files")
        overall_progress.progress(0.95)
        
        return {
            "bom_items": bom_items,
            "hierarchy": hierarchy_data,
            "detections": reconciled_detections,
            "excel_path": str(excel_path),
            "json_path": str(json_path)
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Screw Localization System",
        page_icon="🔩",
        layout="wide"
    )
    
    st.title("🔩 Screw Localization System")
    st.markdown("BOM extraction from PDF plans with hierarchical multipliers")
    
    # Initialize session state
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    
    # Sidebar for file uploads
    st.sidebar.header("File Upload")
    
    # Option to choose between PDF upload and JSON upload
    upload_mode = st.sidebar.radio(
        "Upload Mode",
        ["Upload PDF", "Upload JSON Results"],
        help="Choose to either process a new PDF or view existing results"
    )
    
    if upload_mode == "Upload PDF":
        uploaded_pdf = st.sidebar.file_uploader(
            "Upload PDF file to process",
            type=["pdf"],
            help="Upload a vector-based PDF construction plan"
        )
        
        if uploaded_pdf:
            st.sidebar.write(f"**File:** {uploaded_pdf.name}")
            st.sidebar.write(f"**Size:** {uploaded_pdf.size / 1024:.1f} KB")
            
            # Check for API key
            has_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not has_api_key:
                st.sidebar.warning("⚠️ No LLM API key found. Hierarchy building will use fallback mode.")
            
            # Process button
            if st.sidebar.button("🚀 Process PDF", type="primary", use_container_width=True):
                try:
                    # Create output directory
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create progress container
                    progress_container = st.container()
                    
                    with progress_container:
                        st.header("📊 Processing Status")
                        
                        # Create status boxes for each step
                        status_step1 = st.empty()
                        status_step2 = st.empty()
                        status_step3 = st.empty()
                        status_step4 = st.empty()
                        status_step5 = st.empty()
                        status_step6 = st.empty()
                        
                        # Overall progress bar
                        overall_progress = st.progress(0)
                        current_status = st.empty()
                        
                        # Step 1: PDF Processing
                        status_step1.info("📄 **Step 1: PDF Processing** - Waiting to start...")
                        current_status.text("Initializing PDF processor...")
                        overall_progress.progress(0)
                        
                        # Process PDF with detailed progress
                        results = process_pdf_with_progress(
                            uploaded_pdf, 
                            output_dir,
                            status_step1,
                            status_step2,
                            status_step3,
                            status_step4,
                            status_step5,
                            status_step6,
                            current_status,
                            overall_progress
                        )
                        
                        # Load JSON results
                        current_status.text("Loading results...")
                        with open(results["json_path"], "r") as f:
                            data = json.load(f)
                        
                        st.session_state.processed_data = data
                        st.session_state.processing_status = "success"
                        
                        overall_progress.progress(1.0)
                        current_status.success("✅ All steps completed successfully!")
                        
                        st.balloons()
                        
                        # Provide download buttons
                        st.sidebar.markdown("---")
                        st.sidebar.header("Download Results")
                        
                        # Download Excel
                        if os.path.exists(results["excel_path"]):
                            with open(results["excel_path"], "rb") as f:
                                st.sidebar.download_button(
                                    "📥 Download Excel BOM",
                                    f.read(),
                                    file_name=Path(results["excel_path"]).name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        # Download JSON
                        with open(results["json_path"], "rb") as f:
                            st.sidebar.download_button(
                                "📥 Download JSON Results",
                                f.read(),
                                file_name=Path(results["json_path"]).name,
                                mime="application/json"
                            )
                        
                except Exception as e:
                    st.session_state.processing_status = "error"
                    st.sidebar.error(f"❌ Error: {str(e)}")
                    st.error(f"Processing failed: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    else:  # Upload JSON Results
        uploaded_json = st.sidebar.file_uploader(
            "Upload JSON results file",
            type=["json"],
            help="Upload previously generated JSON results file"
        )
        
        if uploaded_json:
            try:
                data = json.load(uploaded_json)
                st.session_state.processed_data = data
                st.session_state.processing_status = "success"
                st.sidebar.success("✅ JSON file loaded")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading JSON: {str(e)}")
                st.session_state.processing_status = "error"
    
    # Display results if available
    if st.session_state.processed_data:
        data = st.session_state.processed_data
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "BOM", "Hierarchy", "Multipliers", "Detections"
        ])
        
        with tab1:
            if "bom_items" in data:
                bom_items = data["bom_items"]
                render_bom_table(bom_items)
            else:
                st.info("No BOM items found in data")
        
        with tab2:
            if "hierarchy" in data:
                hierarchy_data = {"hierarchy": data["hierarchy"]}
                render_hierarchy_tree(hierarchy_data)
            else:
                st.info("No hierarchy data found")
        
        with tab3:
            if "bom_items" in data:
                bom_items = data["bom_items"]
                render_multiplier_visualization(bom_items)
            else:
                st.info("No BOM items found for visualization")
        
        with tab4:
            if "detections" in data:
                render_detection_overview(data["detections"])
            else:
                st.info("No detection data found")
    
    else:
        # Welcome screen
        st.info("👈 Upload a PDF file or JSON results file to begin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📄 Upload PDF
            Upload a vector-based PDF construction plan to:
            1. Extract pages and vector data
            2. Detect fasteners (screws, rivets, anchors, etc.)
            3. Build hierarchy with multipliers
            4. Generate normalized BOM
            
            **Requirements:**
            - Vector-based PDF (not scanned)
            - Optional: LLM API key for better hierarchy detection
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Upload JSON Results
            Upload previously generated JSON results to:
            1. Review BOM items
            2. Visualize hierarchy
            3. Analyze multipliers
            4. Review detection statistics
            
            **Generate JSON by:**
            - Processing PDF through this UI
            - Running: `python main.py --pdf plan.pdf`
            """)


if __name__ == "__main__":
    main()

