import os
import cv2
from datetime import datetime

def save_first_detection(face_img, username):
    base_dir = os.path.join("data", "detected_people", username)
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"

    save_path = os.path.join(base_dir, filename)

    cv2.imwrite(save_path, face_img)

    print(f"[INFO] Screenshot saved: {save_path}")