import cv2
import numpy as np

class FruitDetector:
    def __init__(self):
       
        self.healthy_green = (np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        
        self.disease_brown = (np.array([5, 50, 50]), np.array([20, 255, 180]))    
        self.disease_black = (np.array([0, 0, 0]), np.array([180, 255, 50]))
        
        
        self.fruit_green = (np.array([35, 100, 100]), np.array([70, 255, 220]))   
        self.fruit_yellow = (np.array([22, 80, 80]), np.array([32, 255, 255]))
        self.fruit_orange = (np.array([10, 150, 150]), np.array([22, 255, 255]))
        self.fruit_red = (np.array([0, 150, 100]), np.array([10, 255, 255]))
    
    def detect(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Check the path of pic"}
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            fruit_mask, fruit_count, fruit_locations = self._detect_fruits(hsv)
            disease_mask = self._create_disease_mask(hsv, fruit_mask)
            ripeness_distribution = self._get_color_distribution(hsv, fruit_mask)
            
            return {
                "fruit_mask": fruit_mask,
                "disease_mask": disease_mask,
                "fruit_count": fruit_count,
                "fruit_locations": fruit_locations,
                "ripeness_distribution": ripeness_distribution,
                "original_image": img,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _create_disease_mask(self, hsv, fruit_mask):
        brown_mask = cv2.inRange(hsv, *self.disease_brown)
        black_mask = cv2.inRange(hsv, *self.disease_black)
        combined = cv2.bitwise_or(brown_mask, black_mask)
        combined = cv2.bitwise_and(combined, fruit_mask)
        kernel = np.ones((3,3), np.uint8)
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    def _detect_fruits(self, hsv):
        green_mask = cv2.inRange(hsv, *self.fruit_green)
        yellow_mask = cv2.inRange(hsv, *self.fruit_yellow)
        orange_mask = cv2.inRange(hsv, *self.fruit_orange)
        red_mask = cv2.inRange(hsv, *self.fruit_red)
        
        fruit_mask = cv2.bitwise_or(green_mask, yellow_mask)
        fruit_mask = cv2.bitwise_or(fruit_mask, orange_mask)
        fruit_mask = cv2.bitwise_or(fruit_mask, red_mask)
        
        kernel = np.ones((7,7), np.uint8)
        cleaned = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_count = 0
        fruit_locations = []
        
        img_area = hsv.shape[0] * hsv.shape[1]
        min_fruit_area = max(img_area * 0.01, 500)
        max_fruit_area = img_area * 0.8
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_fruit_area < area < max_fruit_area:
                fruit_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                fruit_locations.append((x, y, w, h))
                
        return cleaned, fruit_count, fruit_locations

    def _get_color_distribution(self, hsv, fruit_mask):
        green_fruit = cv2.countNonZero(cv2.bitwise_and(cv2.inRange(hsv, *self.fruit_green), fruit_mask))
        yellow_fruit = cv2.countNonZero(cv2.bitwise_and(cv2.inRange(hsv, *self.fruit_yellow), fruit_mask))
        orange_fruit = cv2.countNonZero(cv2.bitwise_and(cv2.inRange(hsv, *self.fruit_orange), fruit_mask))
        red_fruit = cv2.countNonZero(cv2.bitwise_and(cv2.inRange(hsv, *self.fruit_red), fruit_mask))
        
        total = green_fruit + yellow_fruit + orange_fruit + red_fruit
        if total == 0:
            return {"green": 0, "yellow": 0, "orange": 0, "red": 0}
        
        return {
            "green": green_fruit / total,
            "yellow": yellow_fruit / total,
            "orange": orange_fruit / total,
            "red": red_fruit / total
        }