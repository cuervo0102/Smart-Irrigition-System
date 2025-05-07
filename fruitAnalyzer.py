import cv2


class FruitAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, detection_data):
        if detection_data.get("status") != "success":
            return detection_data
            
        disease_ratio = self._calculate_disease_ratio(
            detection_data["disease_mask"], 
            detection_data["fruit_mask"]
        )
        
        ripeness_score = self._calculate_ripeness_score(
            detection_data["ripeness_distribution"]
        )
        
        return {
            "disease": self._get_disease_status(disease_ratio),
            "ripeness": self._get_ripeness_status(ripeness_score),
            "ripeness_distribution": detection_data["ripeness_distribution"],
            "fruit_count": detection_data["fruit_count"],
            "recommendation": self._get_recommendation(
                disease_ratio, 
                ripeness_score, 
                detection_data["fruit_count"]
            ),
            "status": "success"
        }
    
    def _calculate_disease_ratio(self, disease_mask, fruit_mask):
        return cv2.countNonZero(disease_mask) / max(cv2.countNonZero(fruit_mask), 1)
    
    def _calculate_ripeness_score(self, distribution):
        total = sum(distribution.values())
        if total == 0:
            return 0
        return (
            distribution["yellow"] * 0.3 + 
            distribution["orange"] * 0.7 + 
            distribution["red"] * 1.0
        )
    
    def _get_disease_status(self, ratio):
        if ratio > 0.15: return "SEVERE"
        if ratio > 0.07: return "MODERATE"
        if ratio > 0.03: return "MILD"
        return "HEALTHY"

    def _get_ripeness_status(self, ratio):
        if ratio > 0.8: return "FULLY RIPE"
        if ratio > 0.5: return "MOSTLY RIPE"
        if ratio > 0.2: return "HALF-RIPE"
        return "MOSTLY UNRIPE"

    def _get_recommendation(self, disease_ratio, ripeness, fruit_count):
        if disease_ratio > 0.15:
            return "Immediate treatment needed: Apply fungicide and remove severely affected fruits"
        elif disease_ratio > 0.07:
            return "Treatment recommended: Apply organic fungicide and monitor affected fruits"
        elif disease_ratio > 0.03:
            return "Monitor closely: Look for spreading disease and improve air circulation"
        elif fruit_count == 0:
            return "No fruits detected: Check image quality or try another image"
        elif ripeness < 0.2:
            return "Fruits are mostly unripe: Continue normal care and wait for ripening"
        elif ripeness > 0.8:
            return "Fruits are ripe: Ready for harvest"
        return "Plants appear healthy: Maintain current care routine"