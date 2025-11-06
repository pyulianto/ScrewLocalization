"""LLM-based hierarchy builder for page structure and multipliers."""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class HierarchyNode:
    """Node in the page/view hierarchy."""
    node_id: str
    page_num: int
    title: str
    node_type: str  # "overview", "detail", "sub_detail"
    multiplier: int = 1
    parent_id: Optional[str] = None
    children: List[str] = None
    viewport_bbox: Optional[Dict] = None
    scale: Optional[float] = None
    detection_ids: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.detection_ids is None:
            self.detection_ids = []


@dataclass
class DetectionAssignment:
    """Assignment of detection to hierarchy node with confidence."""
    detection_id: str
    node_id: str
    confidence: float
    reasoning: str


class HierarchyBuilder:
    """Build page hierarchy and assign multipliers using LLM reasoning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.llm_provider = config.get("provider", "openai")
        self.llm_model = config.get("model", "gpt-4-turbo-preview")
        self.temperature = config.get("temperature", 0.1)
        self.client = None  # Initialize lazily when needed
    
    def _get_client(self):
        """Get or initialize LLM client (lazy initialization)."""
        if self.client is not None:
            return self.client
        
        # Initialize LLM client only when needed
        if self.llm_provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None  # Return None instead of raising error
            self.client = OpenAI(api_key=api_key)
        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None  # Return None instead of raising error
            self.client = Anthropic(api_key=api_key)
        else:
            return None
        
        return self.client
    
    def build_hierarchy(self, pages_data: List, detections: List[Dict]) -> Dict:
        """
        Build hierarchy from pages and assign detections.
        
        Returns:
            Dict with:
            - hierarchy: List of HierarchyNode
            - assignments: List of DetectionAssignment
            - multiplier_map: Dict mapping node_id to multiplier
        """
        # Prepare input for LLM
        structure_input = self._prepare_structure_input(pages_data, detections)
        
        # Check if we have API key, otherwise use fallback
        client = self._get_client()
        if client is None:
            # Use fallback hierarchy (no API key)
            hierarchy_json = self._fallback_hierarchy(structure_input)
        else:
            # Call LLM to build hierarchy
            hierarchy_json = self._call_llm_for_hierarchy(structure_input)
        
        # Parse and validate hierarchy
        hierarchy_data = json.loads(hierarchy_json)
        
        # Build node objects
        nodes = []
        for node_data in hierarchy_data.get("hierarchy", []):
            node = HierarchyNode(**node_data)
            nodes.append(node)
        
        # Build assignments
        assignments = []
        for assignment_data in hierarchy_data.get("assignments", []):
            assignment = DetectionAssignment(**assignment_data)
            assignments.append(assignment)
        
        # Build multiplier map
        multiplier_map = {node.node_id: node.multiplier for node in nodes}
        
        return {
            "hierarchy": nodes,
            "assignments": assignments,
            "multiplier_map": multiplier_map
        }
    
    def _prepare_structure_input(self, pages_data: List, detections: List[Dict]) -> str:
        """Prepare structured input for LLM."""
        input_data = {
            "pages": [],
            "detections": []
        }
        
        def convert_bbox(bbox):
            """Convert bbox to JSON-serializable format."""
            if bbox is None:
                return None
            # Handle PyMuPDF Rect objects or tuples/lists
            if hasattr(bbox, 'x0'):  # PyMuPDF Rect object
                return [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
            elif isinstance(bbox, (list, tuple)):
                return [float(x) for x in bbox]
            return bbox
        
        # Add page information
        for page in pages_data:
            input_data["pages"].append({
                "page_num": page.page_num,
                "title": page.title,
                "scale": page.scale,
                "viewport_frames": [
                    {
                        "bbox": convert_bbox(vf.get("bbox")),
                        "labels": vf.get("labels", [])
                    }
                    for vf in page.viewport_frames
                ],
                "text_blocks_count": len(page.text_blocks)
            })
        
        # Add detection summaries
        for det in detections:
            det_id = getattr(det, 'detection_id', None) or f"det_{detections.index(det)}"
            input_data["detections"].append({
                "detection_id": det_id,
                "page_num": det.page_num,
                "category": det.category,
                "confidence": float(det.confidence),
                "center": [float(x) for x in det.center] if isinstance(det.center, (tuple, list)) else det.center,
                "nearby_text": det.nearby_text[:3] if det.nearby_text else []  # Limit text
            })
        
        return json.dumps(input_data, indent=2)
    
    def _call_llm_for_hierarchy(self, structure_input: str) -> str:
        """Call LLM to build hierarchy."""
        prompt = f"""You are analyzing a construction plan PDF to build a hierarchy of pages and detail views, and assign multipliers based on repetition.

Your task:
1. Build a hierarchy tree where:
   - Level 0: Overview/plan pages
   - Level 1: Detail views (e.g., "Detail A", "Section C-C")
   - Level 2: Sub-details or callouts
   
2. Assign multipliers to nodes based on:
   - Callout labels (e.g., "Detail A applies to axes 1-4" → multiplier 4)
   - Text references (e.g., "typical for all 8 corners" → multiplier 8)
   - Scale differences (if same detail appears at different scales, count as one)
   
3. Assign each detection to the appropriate hierarchy node.

Input data:
{structure_input}

Output JSON format:
{{
  "hierarchy": [
    {{
      "node_id": "node_0",
      "page_num": 0,
      "title": "Overview Plan",
      "node_type": "overview",
      "multiplier": 1,
      "parent_id": null,
      "children": ["node_1", "node_2"],
      "viewport_bbox": null,
      "scale": null,
      "detection_ids": []
    }},
    {{
      "node_id": "node_1",
      "page_num": 0,
      "title": "Detail A",
      "node_type": "detail",
      "multiplier": 4,
      "parent_id": "node_0",
      "children": [],
      "viewport_bbox": {{"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
      "scale": 0.1,
      "detection_ids": ["det_0", "det_1"]
    }}
  ],
  "assignments": [
    {{
      "detection_id": "det_0",
      "node_id": "node_1",
      "confidence": 0.9,
      "reasoning": "Detection is within Detail A viewport and matches context"
    }}
  ]
}}

Return only valid JSON, no additional text."""

        try:
            client = self._get_client()
            if client is None:
                # No API key, use fallback
                return self._fallback_hierarchy(structure_input)
            
            if self.llm_provider == "openai":
                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a construction plan analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content
            elif self.llm_provider == "anthropic":
                response = client.messages.create(
                    model=self.llm_model,
                    max_tokens=self.config.get("max_tokens", 4000),
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                result = response.content[0].text
            else:
                # Unsupported provider, use fallback
                return self._fallback_hierarchy(structure_input)
            
            # Clean up JSON if it's wrapped in markdown code blocks
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return a fallback structure
            return self._fallback_hierarchy(structure_input)
    
    def _fallback_hierarchy(self, structure_input: str) -> str:
        """Generate a simple fallback hierarchy when LLM fails."""
        import json
        try:
            data = json.loads(structure_input)
            nodes = []
            assignments = []
            
            # Create one node per page
            for i, page in enumerate(data.get("pages", [])):
                node_id = f"node_{i}"
                nodes.append({
                    "node_id": node_id,
                    "page_num": page["page_num"],
                    "title": page.get("title", f"Page {page['page_num'] + 1}"),
                    "node_type": "overview",
                    "multiplier": 1,
                    "parent_id": None,
                    "children": [],
                    "viewport_bbox": None,
                    "scale": page.get("scale"),
                    "detection_ids": []
                })
            
            # Assign all detections to their respective pages
            for det in data.get("detections", []):
                page_num = det["page_num"]
                node_id = f"node_{page_num}"
                assignments.append({
                    "detection_id": det["detection_id"],
                    "node_id": node_id,
                    "confidence": 0.7,
                    "reasoning": "Assigned to page node (fallback)"
                })
            
            return json.dumps({
                "hierarchy": nodes,
                "assignments": assignments
            })
        except:
            # Ultimate fallback
            return json.dumps({
                "hierarchy": [],
                "assignments": []
            })
    
    def reconcile_duplicates(self, detections: List[Dict], 
                            hierarchy: Dict) -> List[Dict]:
        """Reconcile duplicate detections across identical details."""
        # Group detections by similarity
        # This is a simplified version - in production, use proper similarity matching
        reconciled = []
        seen_groups = []
        
        for det in detections:
            # Check if this detection is similar to one we've seen
            matched = False
            for group in seen_groups:
                if self._are_similar(det, group["representative"]):
                    group["detections"].append(det)
                    matched = True
                    break
            
            if not matched:
                seen_groups.append({
                    "representative": det,
                    "detections": [det]
                })
        
        # For each group, keep the highest confidence detection
        for group in seen_groups:
            best = max(group["detections"], key=lambda d: d.confidence)
            reconciled.append(best)
        
        return reconciled
    
    def _are_similar(self, det1: Dict, det2: Dict, 
                    threshold: float = 0.8) -> bool:
        """Check if two detections are similar (same type, similar position)."""
        if det1.category != det2.category:
            return False
        
        # Check if specs match
        specs1 = det1.spec_fields or {}
        specs2 = det2.spec_fields or {}
        if specs1.get("thread") and specs2.get("thread"):
            if specs1["thread"] != specs2["thread"]:
                return False
        
        # Position similarity (if on same page)
        if det1.page_num == det2.page_num:
            dist = ((det1.center[0] - det2.center[0])**2 + 
                   (det1.center[1] - det2.center[1])**2)**0.5
            # If very close, might be duplicate
            if dist < 10.0:
                return True
        
        return False

