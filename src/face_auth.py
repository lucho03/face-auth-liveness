import cv2
import json
import config
from typing import Tuple

class FaceAuthenticator:
    def __init__(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError("Install opencv-contrib-python")

        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}

    def train(self, images, labels, label_map):
        self.model.train(images, labels)
        self.label_map = label_map

    def save(self):
        self.model.write(config.LBPH_MODEL_PATH)

        with open(config.LABEL_MAP_PATH, "w") as f:
            json.dump(self.label_map, f, indent=2)

    def load(self):
        self.model.read(config.LBPH_MODEL_PATH)

        with open(config.LABEL_MAP_PATH, "r") as f:
            self.label_map = json.load(f)
        
        self.label_map = {int(k): v for k, v in self.label_map.items()}

    def predict(self, face) -> Tuple[str, float]:
        label, confidence = self.model.predict(face)
        name = self.label_map.get(label, "unknown")
        return name, confidence