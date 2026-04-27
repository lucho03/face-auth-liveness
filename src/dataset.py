import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

def load_dataset(base_dir: str = "data/enrolled") -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images = []
    labels = []
    label_map = {}

    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    current_label = 0

    for person_dir in sorted(base_path.iterdir()):
        if not person_dir.is_dir():
            continue

        label_map[current_label] = person_dir.name

        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))

            images.append(img)
            labels.append(current_label)

        current_label += 1

    if len(images) == 0:
        raise ValueError("No images found in dataset!")
    
    return images, np.array(labels, dtype=np.int32), label_map