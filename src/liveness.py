from typing import List, Tuple
import config
from features import (
    eye_aspect_ratio,
    head_turn_score,
    mouth_smile_ratio,
    LEFT_EYE,
    RIGHT_EYE,
)

Point = Tuple[float, float]

class LivenessVerifier:
    def __init__(self):
        self._ear_low_frames = 0

    def update(self, landmarks: List[Point]) -> List[str]:
        new_actions = []
        try:
            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0

            if ear < config.EAR_THRESHOLD:
                self._ear_low_frames += 1
            else:
                if self._ear_low_frames >= config.EAR_CONSEC_FRAMES:
                    new_actions.append("blink")
                self._ear_low_frames = 0

        except Exception:
            pass

        try:
            turn = head_turn_score(landmarks)
            if turn < -config.HEAD_TURN_THRESHOLD:
                new_actions.append("turn_left")

            elif turn > config.HEAD_TURN_THRESHOLD:
                new_actions.append("turn_right")

        except Exception:
            pass

        try:
            smile_ratio = mouth_smile_ratio(landmarks)
            if smile_ratio > config.SMILE_THRESHOLD:
                new_actions.append("smile")

        except Exception:
            pass

        return new_actions