import cv2
import numpy as np

def show_detection(img, disease_mask, fruit_mask, fruit_locations):
    debug_img = img.copy()
    
    disease_overlay = np.zeros_like(debug_img)
    disease_overlay[disease_mask > 0] = [0, 255, 255]
    cv2.addWeighted(disease_overlay, 0.5, debug_img, 1, 0, debug_img)
    
    for x, y, w, h in fruit_locations:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add legend
    cv2.putText(debug_img, "GREEN: Fruit boundaries", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug_img, "YELLOW: Disease areas", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Tomato Detection Preview", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()