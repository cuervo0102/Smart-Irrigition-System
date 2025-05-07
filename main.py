from fruitDetector import FruitDetector
from fruitAnalyzer import FruitAnalyzer
from Visualization import show_detection

def main():
    detector = FruitDetector()
    analyzer = FruitAnalyzer()
    
    image_path = r"C:\Users\DELL\Desktop\PFA\images\tomato.jpg"  
    
    detection_data = detector.detect(image_path)
    
    if detection_data.get("status") == "success":
        show_detection(
            detection_data["original_image"],
            detection_data["disease_mask"],
            detection_data["fruit_mask"],
            detection_data["fruit_locations"]
        )
        
        results = analyzer.analyze(detection_data)
        
        print(" FRUIT ANALYSIS RESULTS")
        print(f"DISEASE STATUS: {results['disease']}")
        print(f"RIPENESS: {results['ripeness']}")
        print(f"FRUIT COLOR DISTRIBUTION:")
        for color, percentage in results['ripeness_distribution'].items():
            print(f"  - {color.upper()}: {percentage:.1%}")
        print(f"FRUIT COUNT: {results['fruit_count']}")
        print(f"RECOMMENDATION: {results['recommendation']}")
    else:
        print(f"\nERROR: {detection_data.get('error', 'Detection failed')}")

if __name__ == "__main__":
    main()