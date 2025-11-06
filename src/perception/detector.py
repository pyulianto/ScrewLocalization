"""Vector-assisted fastener detection module."""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class Detection:
    """Single fastener detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    confidence: float
    category: str  # screw, rivet, anchor, washer, nut
    subtype: Optional[str] = None
    page_num: int = 0
    nearby_text: List[str] = None
    vector_cues: Dict = None
    spec_fields: Dict = None
    detection_id: Optional[str] = None
    
    def __post_init__(self):
        if self.nearby_text is None:
            self.nearby_text = []
        if self.vector_cues is None:
            self.vector_cues = {}
        if self.spec_fields is None:
            self.spec_fields = {}


class VectorAssistedDetector:
    """Detect fasteners using vector patterns and raster analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_circle_radius = config.get("min_circle_radius", 0.5)
        self.max_circle_radius = config.get("max_circle_radius", 5.0)
        self.thread_threshold = config.get("thread_pattern_threshold", 0.3)
        self.min_confidence = config.get("min_confidence", 0.5)
        self.max_lines_for_thread = config.get("max_lines_for_thread_detection", 5000)
        self.enable_thread_detection = config.get("enable_thread_detection", True)
    
    def detect(self, page_data, progress_callback=None) -> List[Detection]:
        """
        Detect fasteners on a page using vector-assisted approach.
        
        Args:
            page_data: PageData object with vector primitives and text
            progress_callback: Optional callback function(status_message) for progress updates
        """
        detections = []
        
        # Step 1: Find candidate circles (potential fastener heads)
        if progress_callback:
            progress_callback("Finding circles (potential fastener heads)...")
        circles = self._find_circles(page_data.vector_primitives)
        if progress_callback:
            progress_callback(f"Found {len(circles)} candidate circles")
        
        # Step 2: Find thread patterns (parallel lines near circles)
        if self.enable_thread_detection:
            if progress_callback:
                progress_callback("Searching for thread patterns (parallel lines)...")
            thread_patterns = self._find_thread_patterns(page_data.vector_primitives, progress_callback)
            if progress_callback:
                progress_callback(f"Found {len(thread_patterns)} thread patterns")
        else:
            if progress_callback:
                progress_callback("Skipping thread pattern detection (disabled in config)...")
            thread_patterns = []
        
        # Step 3: Match circles to thread patterns and classify
        if progress_callback:
            progress_callback(f"Analyzing {len(circles)} circles for classification...")
        
        for i, circle in enumerate(circles):
            if progress_callback and i % max(1, len(circles) // 10) == 0:
                progress_callback(f"Processing circle {i+1}/{len(circles)}...")
            
            # Check if there's a thread pattern nearby
            nearby_thread = self._find_nearby_thread(circle, thread_patterns)
            
            # Check for nearby text that might indicate specs
            nearby_text = self._find_nearby_text(circle, page_data.text_blocks)
            
            # Classify based on size and patterns
            category, confidence = self._classify_candidate(
                circle, nearby_thread, nearby_text
            )
            
            if confidence >= self.min_confidence:
                # Extract specs from text
                spec_fields = self._extract_specs_from_text(nearby_text)
                
                detection = Detection(
                    bbox=circle["bbox"],
                    center=circle["center"],
                    confidence=confidence,
                    category=category,
                    page_num=page_data.page_num,
                    nearby_text=nearby_text,
                    vector_cues={
                        "has_thread": nearby_thread is not None,
                        "circle_radius": circle["radius"]
                    },
                    spec_fields=spec_fields
                )
                detections.append(detection)
        
        if progress_callback:
            progress_callback(f"Classified {len(detections)} fasteners from circles")
        
        # Step 4: Also look for other fastener types (washers, nuts as separate entities)
        if progress_callback:
            progress_callback("Detecting washers and nuts...")
        washer_detections = self._detect_washers(page_data)
        nut_detections = self._detect_nuts(page_data)
        
        if progress_callback:
            progress_callback(f"Found {len(washer_detections)} washers, {len(nut_detections)} nuts")
        
        return detections + washer_detections + nut_detections
    
    def _find_circles(self, vector_primitives: List[Dict]) -> List[Dict]:
        """Find circular shapes in vector primitives."""
        circles = []
        for prim in vector_primitives:
            if prim["type"] == "circle":
                radius = prim.get("radius", 0)
                if self.min_circle_radius <= radius <= self.max_circle_radius:
                    circles.append(prim)
        
        # Also detect circles from paths that form closed loops
        # This is a simplified approach - in production, use proper circle fitting
        return circles
    
    def _find_thread_patterns(self, vector_primitives: List[Dict], progress_callback=None) -> List[Dict]:
        """Find parallel line patterns that indicate threads."""
        thread_patterns = []
        lines = [p for p in vector_primitives if p["type"] == "line"]
        
        if progress_callback:
            progress_callback(f"Analyzing {len(lines)} lines for parallel patterns...")
        
        # Optimize: Limit to reasonable number of lines to avoid O(n²) explosion
        if len(lines) > self.max_lines_for_thread:
            if progress_callback:
                progress_callback(f"Too many lines ({len(lines):,}), sampling {self.max_lines_for_thread:,} for efficiency...")
            # Sample lines - prioritize shorter lines (more likely to be threads)
            lines_with_length = []
            for line in lines:
                start = line.get("start", (0, 0))
                end = line.get("end", (0, 0))
                length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                lines_with_length.append((length, line))
            
            # Sort by length (shorter = more likely thread pattern)
            lines_with_length.sort(key=lambda x: x[0])
            lines = [line for _, line in lines_with_length[:self.max_lines_for_thread]]
            if progress_callback:
                progress_callback(f"Using {len(lines):,} sampled lines for thread detection...")
        
        total_lines = len(lines)
        total_comparisons = total_lines * (total_lines - 1) // 2
        
        if progress_callback:
            progress_callback(f"Checking {total_comparisons:,} line pairs...")
        
        comparisons_done = 0
        # Update more frequently - every 1% or every 1000 comparisons, whichever is smaller
        update_interval = max(1, min(1000, total_comparisons // 100))
        last_update_percent = -1
        
        # Group parallel lines that are close together
        for i, line1 in enumerate(lines):
            # Progress update per outer loop iteration
            if progress_callback and i % max(1, total_lines // 50) == 0:
                outer_progress = (i / total_lines) * 100
                progress_callback(f"Processing line {i+1}/{total_lines} ({outer_progress:.0f}%)...")
            
            for line2 in lines[i+1:]:
                comparisons_done += 1
                
                # Progress update for inner loop
                if progress_callback and comparisons_done % update_interval == 0:
                    progress = (comparisons_done / total_comparisons) * 100
                    current_percent = int(progress)
                    if current_percent != last_update_percent:
                        progress_callback(f"Checking pairs: {comparisons_done:,}/{total_comparisons:,} ({current_percent}%)...")
                        last_update_percent = current_percent
                
                if self._are_parallel_and_close(line1, line2):
                    thread_patterns.append({
                        "lines": [line1, line2],
                        "center": self._line_pair_center(line1, line2)
                    })
        
        if progress_callback:
            progress_callback(f"Thread pattern search complete: found {len(thread_patterns)} patterns")
        
        return thread_patterns
    
    def _are_parallel_and_close(self, line1: Dict, line2: Dict, 
                                angle_tol: float = 5.0, 
                                dist_tol: float = 1.0) -> bool:
        """Check if two lines are parallel and close together."""
        start1, end1 = line1["start"], line1["end"]
        start2, end2 = line2["start"], line2["end"]
        
        # Calculate angles
        dx1 = end1[0] - start1[0]
        dy1 = end1[1] - start1[1]
        dx2 = end2[0] - start2[0]
        dy2 = end2[1] - start2[1]
        
        angle1 = np.arctan2(dy1, dx1) * 180 / np.pi
        angle2 = np.arctan2(dy2, dx2) * 180 / np.pi
        
        angle_diff = abs(angle1 - angle2) % 180
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        if angle_diff > angle_tol:
            return False
        
        # Check distance between lines
        # Simplified: check distance between midpoints
        mid1 = ((start1[0] + end1[0]) / 2, (start1[1] + end1[1]) / 2)
        mid2 = ((start2[0] + end2[0]) / 2, (start2[1] + end2[1]) / 2)
        dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        
        return dist < dist_tol
    
    def _line_pair_center(self, line1: Dict, line2: Dict) -> Tuple[float, float]:
        """Get center point between two lines."""
        mid1 = (
            (line1["start"][0] + line1["end"][0]) / 2,
            (line1["start"][1] + line1["end"][1]) / 2
        )
        mid2 = (
            (line2["start"][0] + line2["end"][0]) / 2,
            (line2["start"][1] + line2["end"][1]) / 2
        )
        return ((mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2)
    
    def _find_nearby_thread(self, circle: Dict, thread_patterns: List[Dict],
                           threshold: float = 10.0) -> Optional[Dict]:
        """Find thread pattern near a circle."""
        circle_center = circle["center"]
        for pattern in thread_patterns:
            pattern_center = pattern["center"]
            dist = np.sqrt(
                (circle_center[0] - pattern_center[0])**2 +
                (circle_center[1] - pattern_center[1])**2
            )
            if dist < threshold:
                return pattern
        return None
    
    def _find_nearby_text(self, circle: Dict, text_blocks: List[Dict],
                         threshold: float = 30.0) -> List[str]:
        """Find text blocks near a circle."""
        nearby = []
        circle_center = circle["center"]
        for block in text_blocks:
            block_center = (
                (block["bbox"][0] + block["bbox"][2]) / 2,
                (block["bbox"][1] + block["bbox"][3]) / 2
            )
            dist = np.sqrt(
                (circle_center[0] - block_center[0])**2 +
                (circle_center[1] - block_center[1])**2
            )
            if dist < threshold:
                nearby.append(block["text"])
        return nearby
    
    def _classify_candidate(self, circle: Dict, thread_pattern: Optional[Dict],
                           nearby_text: List[str]) -> Tuple[str, float]:
        """Classify a candidate as a specific fastener type."""
        # Default to screw if we have thread pattern
        if thread_pattern:
            # Check text for more specific classification
            text_str = " ".join(nearby_text).upper()
            if "RIVET" in text_str:
                return "rivet", 0.8
            elif "ANCHOR" in text_str:
                return "anchor", 0.85
            else:
                return "screw", 0.75
        
        # Without thread pattern, lower confidence
        # Could be a washer, nut, or rivet head
        text_str = " ".join(nearby_text).upper()
        if "WASHER" in text_str:
            return "washer", 0.7
        elif "NUT" in text_str:
            return "nut", 0.7
        elif "RIVET" in text_str:
            return "rivet", 0.6
        else:
            # Default to screw with lower confidence
            return "screw", 0.5
    
    def _extract_specs_from_text(self, text_list: List[str]) -> Dict:
        """Extract specification fields from nearby text."""
        specs = {}
        full_text = " ".join(text_list)
        
        # Extract thread size (e.g., M6, M8, 1/4")
        thread_pattern = r'\b(M\d+|M\d+x\d+|\d+/\d+"|\d+\.\d+")\b'
        match = re.search(thread_pattern, full_text)
        if match:
            specs["thread"] = match.group(1)
        
        # Extract length (e.g., x80, x100, 80mm)
        length_pattern = r'\b(?:x|×|L)\s*(\d+)\s*(?:mm|m)?\b'
        match = re.search(length_pattern, full_text)
        if match:
            specs["length"] = match.group(1)
        
        # Extract brand (Spax, Hilti, Fischer, etc.)
        brand_pattern = r'\b(Spax|Hilti|Fischer|SFS|Würth|Bosch|TOX)\b'
        match = re.search(brand_pattern, full_text, re.IGNORECASE)
        if match:
            specs["brand_code"] = match.group(1)
        
        # Extract head type from text
        head_types = ["PHILLIPS", "SLOT", "TORX", "HEX", "BUTTON", "FLAT"]
        for head_type in head_types:
            if head_type in full_text.upper():
                specs["head_type"] = head_type.lower()
                break
        
        return specs
    
    def _detect_washers(self, page_data) -> List[Detection]:
        """Detect washers (typically larger circles or rings)."""
        detections = []
        # Look for larger circles that might be washers
        for prim in page_data.vector_primitives:
            if prim["type"] == "circle":
                radius = prim.get("radius", 0)
                # Washers are typically larger than screw heads
                if 3.0 <= radius <= 15.0:
                    nearby_text = self._find_nearby_text(prim, page_data.text_blocks)
                    text_str = " ".join(nearby_text).upper()
                    if "WASHER" in text_str or radius > 8.0:
                        detection = Detection(
                            bbox=prim["bbox"],
                            center=prim["center"],
                            confidence=0.65,
                            category="washer",
                            page_num=page_data.page_num,
                            nearby_text=nearby_text,
                            vector_cues={"circle_radius": radius},
                            spec_fields={}
                        )
                        detections.append(detection)
        return detections
    
    def _detect_nuts(self, page_data) -> List[Detection]:
        """Detect nuts (typically hexagonal shapes)."""
        detections = []
        # Look for hexagonal patterns in vector primitives
        # This is simplified - in production, use proper hexagon detection
        for prim in page_data.vector_primitives:
            if prim["type"] == "circle":
                # Check if nearby text mentions "nut"
                nearby_text = self._find_nearby_text(prim, page_data.text_blocks)
                text_str = " ".join(nearby_text).upper()
                if "NUT" in text_str:
                    detection = Detection(
                        bbox=prim["bbox"],
                        center=prim["center"],
                        confidence=0.7,
                        category="nut",
                        page_num=page_data.page_num,
                        nearby_text=nearby_text,
                        vector_cues={"circle_radius": prim.get("radius", 0)},
                        spec_fields=self._extract_specs_from_text(nearby_text)
                    )
                    detections.append(detection)
        return detections

