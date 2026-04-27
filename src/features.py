from __future__ import annotations
import math
from typing import List, Tuple

Point = Tuple[float, float]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454

def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def eye_aspect_ratio(landmarks: List[Point], eye_indices: List[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def mouth_smile_ratio(landmarks: List[Point]) -> float:
    mouth_width = euclidean_distance(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    face_width = euclidean_distance(landmarks[LEFT_FACE], landmarks[RIGHT_FACE])
    if face_width == 0:
        return 0.0
    return mouth_width / face_width

def head_turn_score(landmarks: List[Point]) -> float:
    face_center_x = (landmarks[LEFT_FACE][0] + landmarks[RIGHT_FACE][0]) / 2.0
    face_width = euclidean_distance(landmarks[LEFT_FACE], landmarks[RIGHT_FACE])
    if face_width == 0:
        return 0.0
    return (landmarks[NOSE_TIP][0] - face_center_x) / face_width