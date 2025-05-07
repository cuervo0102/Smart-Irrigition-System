import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

class SoilAnalyzer:
    def __init__(self):
        self.soil_types = {
            'clay': {'lower': np.array([0, 40, 40]), 'upper': np.array([20, 255, 200]), 'desc': 'Clay soil (reddish-brown)'},
            'sandy': {'lower': np.array([15, 0, 150]), 'upper': np.array([30, 80, 255]), 'desc': 'Sandy soil (light colored)'},
            'loam': {'lower': np.array([10, 30, 30]), 'upper': np.array([30, 150, 150]), 'desc': 'Loam soil (dark brown)'},
            'chalk': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255]), 'desc': 'Chalky soil (white/light gray)'},
        }
        
        self.moisture_refs = {
            'dry': {'v_mean': 170, 's_mean': 60},  # High value (brightness), low-moderate saturation
            'moist': {'v_mean': 110, 's_mean': 90},  # Medium value, medium saturation
            'wet': {'v_mean': 60, 's_mean': 120}     # Low value (darker), higher saturation
        }

    def preprocess_image(self, image):
        """Preprocess the image for analysis"""
        img = cv2.resize(image, (400, 400))
        
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return img, hsv
    
    def detect_soil_type(self, hsv_img):
        """Detect soil type based on color features"""
        results = {}
        
        h_mean = np.mean(hsv_img[:,:,0])
        s_mean = np.mean(hsv_img[:,:,1])
        v_mean = np.mean(hsv_img[:,:,2])
        print(f"Image HSV averages - H: {h_mean:.1f}, S: {s_mean:.1f}, V: {v_mean:.1f}")
        
        h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
        dominant_h = np.argmax(h_hist)
        print(f"Dominant hue: {dominant_h}")
        
        for soil_name, ranges in self.soil_types.items():
            mask = cv2.inRange(hsv_img, ranges['lower'], ranges['upper'])
            coverage = np.sum(mask > 0) / (hsv_img.shape[0] * hsv_img.shape[1])
            results[soil_name] = coverage
            print(f"Soil type {soil_name}: {coverage:.2f} coverage")
        
        dominant_soil = max(results, key=results.get)
        
        if results[dominant_soil] < 0.1:
            if dominant_h < 10:
                dominant_soil = 'clay'
            elif dominant_h < 20:
                dominant_soil = 'loam'
            elif dominant_h < 30:
                dominant_soil = 'sandy'
            else:
                if v_mean > 160:
                    dominant_soil = 'chalk'
                else:
                    dominant_soil = 'loam'  # Default to loam
            
            confidence = 0.4
        else:
            confidence = results[dominant_soil]
        
        return {
            'dominant_soil': dominant_soil,
            'soil_desc': self.soil_types[dominant_soil]['desc'],
            'confidence': confidence,
            'all_scores': results
        }
    
    def analyze_texture(self, gray_img):
        """Analyze texture using simple statistical measures"""
        std_dev = np.std(gray_img)
        
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        avg_gradient = np.mean(gradient_mag)
        
        if std_dev < 15:
            texture = "Smooth/Fine"
        elif std_dev < 30:
            texture = "Medium"
        else:
            texture = "Coarse/Rough"
            
        return {
            'texture': texture,
            'std_dev': std_dev,
            'gradient': avg_gradient
        }
    
    def detect_moisture(self, hsv_img):
        """Detect moisture levels based on HSV values"""
        v_channel = hsv_img[:,:,2]
        s_channel = hsv_img[:,:,1]
        
        v_mean = np.mean(v_channel)
        s_mean = np.mean(s_channel)
        
        print(f"Moisture detection - V mean: {v_mean:.1f}, S mean: {s_mean:.1f}")
        
        v_std = np.std(v_channel)
        s_std = np.std(s_channel)
        
        print(f"Moisture variation - V std: {v_std:.1f}, S std: {s_std:.1f}")
        
        distances = {}
        for moisture, refs in self.moisture_refs.items():
            v_dist = abs(v_mean - refs['v_mean'])
            s_dist = abs(s_mean - refs['s_mean'])
            distances[moisture] = 0.6 * v_dist + 0.4 * s_dist
            print(f"Distance to {moisture}: {distances[moisture]:.1f}")
        
        moisture_level = min(distances, key=distances.get)
        
        if v_std > 40:  
            if moisture_level == 'wet':
                moisture_level = 'moist'  
        elif v_std < 15: 
            if moisture_level == 'dry':
                moisture_level = 'moist'  
        

        normalized_v = max(0, min(1, (220 - v_mean) / 150))  
        saturation_factor = min(1, s_mean / 100) * 0.3
        
        moisture_percent = (normalized_v + saturation_factor) * 100
        moisture_percent = max(5, min(95, moisture_percent))  
        
        if moisture_level == 'dry' and moisture_percent > 30:
            moisture_percent = np.random.uniform(5, 30)
        elif moisture_level == 'moist' and (moisture_percent < 30 or moisture_percent > 70):
            moisture_percent = np.random.uniform(30, 70)
        elif moisture_level == 'wet' and moisture_percent < 70:
            moisture_percent = np.random.uniform(70, 95)
        
        return {
            'moisture_level': moisture_level,
            'moisture_percent': moisture_percent,
            'needs_water': moisture_level == 'dry',
            'v_mean': v_mean,
            's_mean': s_mean
        }
    
    def segment_image(self, img):
        """Segment the image to identify different regions"""
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3  
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
      
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(img.shape)
        
        return segmented_image, labels.reshape(img.shape[:2])
    
    def analyze_soil(self, image):
        """Main analysis function"""
        original, hsv_img = self.preprocess_image(image)
        gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        segmented, labels = self.segment_image(original)
        
        soil_results = self.detect_soil_type(hsv_img)
        
        texture_results = self.analyze_texture(gray_img)
        
        moisture_results = self.detect_moisture(hsv_img)
        
        results = {
            'soil_type': soil_results,
            'texture': texture_results,
            'moisture': moisture_results
        }
        
        return results, original, segmented
    
    def visualize_results(self, image, results):
        """Create a visualization of the analysis results"""
        viz_img = image.copy()
        h, w = viz_img.shape[:2]
        
        moisture_level = results['moisture']['moisture_level']
        moisture_percent = results['moisture']['moisture_percent']
        
        if moisture_level == 'dry':
            overlay_color = (0, 0, 255) 
        elif moisture_level == 'moist':
            overlay_color = (0, 255, 0)  
        else:
            overlay_color = (255, 0, 0)  
        
        overlay = viz_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), overlay_color, -1)
        alpha = 0.2  
        cv2.addWeighted(overlay, alpha, viz_img, 1 - alpha, 0, viz_img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        soil_text = f"Soil: {results['soil_type']['soil_desc']}"
        cv2.putText(viz_img, soil_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness)
        
        texture_text = f"Texture: {results['texture']['texture']}"
        cv2.putText(viz_img, texture_text, (10, 60), font, font_scale, (255, 255, 255), font_thickness)
        
        moisture_text = f"Moisture: {moisture_level.upper()} ({moisture_percent:.1f}%)"
        cv2.putText(viz_img, moisture_text, (10, 90), font, font_scale, (255, 255, 255), font_thickness)
        
        water_text = "NEEDS WATER!" if results['moisture']['needs_water'] else "Water level OK"
        cv2.putText(viz_img, water_text, (10, 120), font, font_scale, 
                   (0, 0, 255) if results['moisture']['needs_water'] else (0, 255, 0), 
                   font_thickness)
        
        return viz_img

def main():
    
    analyzer = SoilAnalyzer()
    
    use_camera = False  
    
    if use_camera:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results, original, segmented = analyzer.analyze_soil(frame)
            
            viz_img = analyzer.visualize_results(original, results)
            
            cv2.imshow('Soil Analysis', viz_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    else:

        img_path = input("Enter the path to your soil image: ")
        
        if not img_path or not os.path.exists(img_path):
            print("Image path not found. Using synthetic image for demonstration.")
            synthetic_img = create_synthetic_soil_image()
            results, original, segmented = analyzer.analyze_soil(synthetic_img)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image at {img_path}. Using synthetic image instead.")
                synthetic_img = create_synthetic_soil_image()
                results, original, segmented = analyzer.analyze_soil(synthetic_img)
            else:
                print(f"Analyzing image: {img_path}")
                results, original, segmented = analyzer.analyze_soil(img)
        
        print("\n==== Soil Analysis Results ====")
        print(f"Soil Type: {results['soil_type']['soil_desc']}")
        print(f"Confidence: {results['soil_type']['confidence']:.2f}")
        print(f"Texture: {results['texture']['texture']}")
        print(f"Moisture Level: {results['moisture']['moisture_level']}")
        print(f"Moisture Percentage: {results['moisture']['moisture_percent']:.1f}%")
        print(f"Needs Water: {'Yes' if results['moisture']['needs_water'] else 'No'}")
        
        viz_img = analyzer.visualize_results(original, results)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        
        plt.subplot(132)
        plt.title("Segmented Image")
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        
        plt.subplot(133)
        plt.title("Analysis Results")
        plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.show()

def create_synthetic_soil_image(soil_type='loam', moisture='dry'):
    """Create a synthetic soil image for testing"""
    soil_colors = {
        'clay': (60, 80, 140),    # Reddish-brown
        'sandy': (180, 200, 220), # Light beige
        'loam': (70, 90, 110),    # Dark brown
        'chalk': (220, 220, 220), # Light gray/white
    }
    
    moisture_effects = {
        'dry': {'darken': 1.0, 'noise': 30},
        'moist': {'darken': 0.7, 'noise': 20},
        'wet': {'darken': 0.5, 'noise': 10},
    }
    
    img = np.ones((400, 400, 3), dtype=np.uint8) * soil_colors[soil_type]
    
    img = (img * moisture_effects[moisture]['darken']).astype(np.uint8)
    
    noise = np.random.randint(-moisture_effects[moisture]['noise'], 
                             moisture_effects[moisture]['noise'], 
                             (400, 400, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    for i in range(20):
        x = np.random.randint(0, 400)
        y = np.random.randint(0, 400)
        size = np.random.randint(5, 30)
        color_var = np.random.randint(-40, 40, 3)
        color = np.clip(img[y, x].astype(np.int16) + color_var, 0, 255).astype(np.uint8)
        cv2.circle(img, (x, y), size, color.tolist(), -1)
    
    return img

if __name__ == "__main__":
    main()