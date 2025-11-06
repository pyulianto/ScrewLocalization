"""PDF processing module to extract pages and vector data."""
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class PageData:
    """Container for processed page data."""
    page_num: int
    width: float
    height: float
    scale: Optional[float] = None
    title: Optional[str] = None
    viewport_frames: List[Dict] = None
    text_blocks: List[Dict] = None
    vector_primitives: List[Dict] = None
    raster_image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.viewport_frames is None:
            self.viewport_frames = []
        if self.text_blocks is None:
            self.text_blocks = []
        if self.vector_primitives is None:
            self.vector_primitives = []


class PDFProcessor:
    """Extract pages, vector data, and text from PDF."""
    
    def __init__(self, pdf_path: str, dpi: int = 300):
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.doc = None
        
    def __enter__(self):
        self.doc = fitz.open(self.pdf_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.doc:
            self.doc.close()
    
    def get_page_count(self) -> int:
        """Get total number of pages."""
        return len(self.doc)
    
    def process_page(self, page_num: int) -> PageData:
        """Process a single page and extract all relevant data."""
        page = self.doc[page_num]
        
        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height
        
        # Extract text blocks with positions
        text_blocks = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                text_content = " ".join([
                    " ".join([span.get("text", "") for span in line.get("spans", [])])
                    for line in block["lines"]
                ]).strip()
                if text_content:
                    # Convert bbox to list (PyMuPDF returns tuples but may contain Rect-like objects)
                    bbox = block.get("bbox", [])
                    if bbox is not None:
                        if hasattr(bbox, 'x0'):  # PyMuPDF Rect object
                            bbox = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
                        else:
                            bbox = [float(x) for x in bbox]
                    
                    text_blocks.append({
                        "text": text_content,
                        "bbox": bbox,
                        "type": block.get("type", 0)
                    })
        
        # Extract vector primitives (paths, circles, lines)
        vector_primitives = []
        for item in page.get_drawings():
            # Extract path elements
            items_list = item.get("items", [])
            for path_item in items_list:
                if path_item[0] == "l":  # line
                    vector_primitives.append({
                        "type": "line",
                        "start": path_item[1],
                        "end": path_item[2],
                        "bbox": item.get("rect", None)
                    })
                elif path_item[0] == "c":  # curve
                    vector_primitives.append({
                        "type": "curve",
                        "points": path_item[1:],
                        "bbox": item.get("rect", None)
                    })
        
        # Extract circles from paths (approximate)
        for path in page.get_drawings():
            rect = path.get("rect")
            if rect:
                # Check if path forms a roughly circular shape
                width = abs(rect[2] - rect[0])
                height = abs(rect[3] - rect[1])
                if abs(width - height) < width * 0.2:  # roughly circular
                    vector_primitives.append({
                        "type": "circle",
                        "center": ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2),
                        "radius": (width + height) / 4,
                        "bbox": rect
                    })
        
        # Render raster image for detection
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA
            img = img[:, :, :3]  # Convert to RGB
        elif pix.n == 1:  # Grayscale
            img = np.stack([img, img, img], axis=2)
        
        # Try to extract scale from text
        scale = self._extract_scale(text_blocks)
        
        # Try to extract title
        title = self._extract_title(text_blocks, page_num)
        
        # Extract viewport frames (rectangles, callouts)
        viewport_frames = self._extract_viewport_frames(page, text_blocks)
        
        return PageData(
            page_num=page_num,
            width=width,
            height=height,
            scale=scale,
            title=title,
            viewport_frames=viewport_frames,
            text_blocks=text_blocks,
            vector_primitives=vector_primitives,
            raster_image=img
        )
    
    def _extract_scale(self, text_blocks: List[Dict]) -> Optional[float]:
        """Extract scale from text (e.g., '1:10', 'SCALE 1:50')."""
        import re
        for block in text_blocks:
            text = block["text"].upper()
            # Match patterns like "1:10", "SCALE 1:50", "1/10"
            match = re.search(r'(?:SCALE\s*)?(\d+)[:/\s](\d+)', text)
            if match:
                num, den = float(match.group(1)), float(match.group(2))
                if den > 0:
                    return num / den
        return None
    
    def _extract_title(self, text_blocks: List[Dict], page_num: int) -> Optional[str]:
        """Extract page title from text blocks."""
        # Look for large text at top of page
        if not text_blocks:
            return f"Page {page_num + 1}"
        
        # Sort by y position (top first)
        sorted_blocks = sorted(text_blocks, key=lambda b: b["bbox"][1])
        if sorted_blocks:
            # Take first significant text block as title
            return sorted_blocks[0]["text"][:100]  # Limit length
        return f"Page {page_num + 1}"
    
    def _extract_viewport_frames(self, page, text_blocks: List[Dict]) -> List[Dict]:
        """Extract viewport frames and callout rectangles."""
        viewports = []
        
        # Look for rectangles that might be viewport frames
        for item in page.get_drawings():
            rect = item.get("rect")
            if rect:
                # Convert rect to list if it's a PyMuPDF Rect object
                if hasattr(rect, 'x0'):  # PyMuPDF Rect object
                    rect_list = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                else:
                    rect_list = [float(x) for x in rect]
                
                # Check if this looks like a frame (has border, reasonable size)
                width = abs(rect_list[2] - rect_list[0])
                height = abs(rect_list[3] - rect_list[1])
                if width > 50 and height > 50:  # Reasonable minimum size
                    # Check for nearby text labels like "Detail A", "SECTION C-C"
                    nearby_labels = self._find_nearby_labels(rect_list, text_blocks)
                    viewports.append({
                        "bbox": rect_list,
                        "width": width,
                        "height": height,
                        "labels": nearby_labels
                    })
        
        return viewports
    
    def _find_nearby_labels(self, rect: Tuple, text_blocks: List[Dict], 
                           threshold: float = 50.0) -> List[str]:
        """Find text blocks near a rectangle."""
        labels = []
        rect_center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        
        for block in text_blocks:
            block_center = (
                (block["bbox"][0] + block["bbox"][2]) / 2,
                (block["bbox"][1] + block["bbox"][3]) / 2
            )
            distance = np.sqrt(
                (rect_center[0] - block_center[0])**2 + 
                (rect_center[1] - block_center[1])**2
            )
            if distance < threshold:
                labels.append(block["text"])
        
        return labels
    
    def process_all_pages(self) -> List[PageData]:
        """Process all pages in the PDF."""
        pages = []
        for page_num in range(len(self.doc)):
            pages.append(self.process_page(page_num))
        return pages

