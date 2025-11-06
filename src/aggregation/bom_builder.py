"""BOM aggregation, normalization, and QC."""
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import json


@dataclass
class BOMItem:
    """Normalized BOM item with aggregated counts."""
    item_id: str
    category: str
    subtype: str
    spec_normalized: Dict
    per_instance_count: int
    multiplier: int
    total_count: int
    confidence: float
    source_nodes: List[str]
    evidence_detections: List[str]
    reasoning: str
    needs_review: bool = False


class BOMBuilder:
    """Build normalized BOM from detections and hierarchy."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_accuracy = config.get("target_item_level", 0.95)
        self.low_confidence_threshold = config.get("low_confidence_threshold", 0.7)
    
    def build_bom(self, detections: List[Dict], hierarchy_data: Dict) -> List[BOMItem]:
        """
        Build normalized BOM from detections and hierarchy.
        
        Args:
            detections: List of Detection objects
            hierarchy_data: Dict with hierarchy, assignments, multiplier_map
            
        Returns:
            List of BOMItem objects
        """
        # Group detections by normalized spec
        grouped = self._group_by_normalized_spec(detections, hierarchy_data)
        
        # Build BOM items
        bom_items = []
        for item_id, group in grouped.items():
            # Calculate total count with multipliers
            total_count = self._calculate_total_count(group, hierarchy_data)
            
            # Determine confidence
            avg_confidence = sum(d.confidence for d in group["detections"]) / len(group["detections"])
            
            # Check if needs review
            needs_review = (
                avg_confidence < self.low_confidence_threshold or
                len(group["detections"]) == 1  # Single detection might need verification
            )
            
            # Build reasoning string
            reasoning = self._build_reasoning(group, hierarchy_data)
            
            bom_item = BOMItem(
                item_id=item_id,
                category=group["representative"].category,
                subtype=group["representative"].subtype or "standard",
                spec_normalized=group["normalized_spec"],
                per_instance_count=len(group["detections"]),
                multiplier=group["multiplier"],
                total_count=total_count,
                confidence=avg_confidence,
                source_nodes=group["node_ids"],
                evidence_detections=[f"det_{i}" for i in range(len(group["detections"]))],
                reasoning=reasoning,
                needs_review=needs_review
            )
            bom_items.append(bom_item)
        
        # Sort by category, then by spec
        bom_items.sort(key=lambda x: (x.category, str(x.spec_normalized)))
        
        return bom_items
    
    def _group_by_normalized_spec(self, detections: List[Dict], 
                                  hierarchy_data: Dict) -> Dict:
        """Group detections by normalized specification."""
        groups = defaultdict(lambda: {
            "detections": [],
            "normalized_spec": {},
            "node_ids": set(),
            "representative": None
        })
        
        assignments_map = {
            a.detection_id: a.node_id 
            for a in hierarchy_data.get("assignments", [])
        }
        
        multiplier_map = hierarchy_data["multiplier_map"]
        
        for i, det in enumerate(detections):
            # Normalize spec
            normalized = self._normalize_spec(det)
            spec_key = self._spec_to_key(normalized, det.category)
            
            # Get node and multiplier
            det_id = getattr(det, 'detection_id', None) or f"det_{i}"
            node_id = assignments_map.get(det_id, "unknown")
            multiplier = multiplier_map.get(node_id, 1)
            
            groups[spec_key]["detections"].append(det)
            groups[spec_key]["normalized_spec"] = normalized
            groups[spec_key]["node_ids"].add(node_id)
            groups[spec_key]["multiplier"] = multiplier
            
            # Set representative (highest confidence)
            if (groups[spec_key]["representative"] is None or 
                det.confidence > groups[spec_key]["representative"].confidence):
                groups[spec_key]["representative"] = det
        
        # Convert sets to lists
        for group in groups.values():
            group["node_ids"] = list(group["node_ids"])
        
        return groups
    
    def _normalize_spec(self, detection: Dict) -> Dict:
        """Normalize specification fields."""
        spec = detection.spec_fields or {}
        normalized = {}
        
        # Normalize thread
        if "thread" in spec:
            normalized["thread"] = self._normalize_thread(spec["thread"])
        
        # Normalize length
        if "length" in spec:
            normalized["length"] = self._normalize_length(spec["length"])
        
        # Copy other fields
        for field in ["head_type", "material", "brand_code"]:
            if field in spec:
                normalized[field] = spec[field]
        
        return normalized
    
    def _normalize_thread(self, thread_str: str) -> str:
        """Normalize thread specification."""
        import re
        # Standardize formats like "M6", "M6x80", "1/4""
        thread_str = thread_str.upper().strip()
        
        # Metric threads
        if thread_str.startswith("M"):
            # Extract just the size (M6, M8, etc.)
            match = re.match(r'M(\d+)', thread_str)
            if match:
                return f"M{match.group(1)}"
        
        # Imperial threads
        match = re.match(r'(\d+/\d+)"?', thread_str)
        if match:
            return f"{match.group(1)}\""
        
        return thread_str
    
    def _normalize_length(self, length_str: str) -> str:
        """Normalize length specification."""
        import re
        # Extract numeric value
        match = re.search(r'(\d+)', str(length_str))
        if match:
            return f"{match.group(1)}mm"
        return str(length_str)
    
    def _spec_to_key(self, spec: Dict, category: str) -> str:
        """Create a unique key for a specification."""
        parts = [category]
        if spec.get("thread"):
            parts.append(f"thread:{spec['thread']}")
        if spec.get("length"):
            parts.append(f"length:{spec['length']}")
        if spec.get("head_type"):
            parts.append(f"head:{spec['head_type']}")
        return "|".join(parts)
    
    def _calculate_total_count(self, group: Dict, hierarchy_data: Dict) -> int:
        """Calculate total count considering multipliers."""
        per_instance = len(group["detections"])
        multiplier = group.get("multiplier", 1)
        return per_instance * multiplier
    
    def _build_reasoning(self, group: Dict, hierarchy_data: Dict) -> str:
        """Build human-readable reasoning string."""
        parts = []
        
        parts.append(f"Found {len(group['detections'])} instance(s)")
        
        if group.get("multiplier", 1) > 1:
            parts.append(f"applying multiplier {group['multiplier']}")
        
        node_names = [
            hierarchy_data["hierarchy"][i].title 
            for i, node in enumerate(hierarchy_data["hierarchy"])
            if node.node_id in group["node_ids"]
        ]
        if node_names:
            parts.append(f"in {', '.join(node_names[:3])}")
        
        return " | ".join(parts)
    
    def apply_sanity_checks(self, bom_items: List[BOMItem]) -> List[BOMItem]:
        """Apply sanity checks and flag suspicious items."""
        for item in bom_items:
            # Check for unusually high counts
            if item.total_count > 1000:
                item.needs_review = True
                item.reasoning += " | High count flagged for review"
            
            # Check for missing critical specs
            if item.category in ["screw", "rivet"] and not item.spec_normalized.get("thread"):
                item.needs_review = True
                item.reasoning += " | Missing thread specification"
        
        return bom_items
    
    def export_to_json(self, bom_items: List[BOMItem], 
                      detections: List[Dict]) -> Dict:
        """Export BOM and evidence to JSON format."""
        return {
            "bom_items": [asdict(item) for item in bom_items],
            "detections": [
                {
                    "id": f"det_{i}",
                    "bbox": list(det.bbox),
                    "center": list(det.center),
                    "confidence": det.confidence,
                    "category": det.category,
                "page_num": det.page_num,
                "nearby_text": det.nearby_text,
                "spec_fields": det.spec_fields,
                "detection_id": getattr(det, 'detection_id', f"det_{i}")
            }
                for i, det in enumerate(detections)
            ],
            "metadata": {
                "total_items": len(bom_items),
                "items_needing_review": sum(1 for item in bom_items if item.needs_review),
                "average_confidence": sum(item.confidence for item in bom_items) / len(bom_items) if bom_items else 0
            }
        }

