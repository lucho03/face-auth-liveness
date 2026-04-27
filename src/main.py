import cv2
import random
import mediapipe as mp
from config import CHALLENGES
from liveness import LivenessVerifier
from face_auth import FaceAuthenticator
from screenshot import save_first_detection

current_challenge = random.choice(CHALLENGES)
_screenshot_taken = False

def extract_landmarks(face_landmarks, width, height):
    points = []
    for lm in face_landmarks.landmark:
        points.append((lm.x * width, lm.y * height))
    return points

def main():
    global current_challenge, _screenshot_taken

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh
    verifier = LivenessVerifier()
    auth = FaceAuthenticator()
    auth.load()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    is_live = False

    with mp_face.FaceMesh(max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            status_text = "No face"

            if results.multi_face_landmarks and is_live == False:
                landmarks = extract_landmarks(
                    results.multi_face_landmarks[0],
                    frame.shape[1],
                    frame.shape[0],
                )

                actions = verifier.update(landmarks)
                print(f"Detected actions: {actions}")
                
                if current_challenge in actions:
                    is_live = True
                    status_text = "Liveness OK"
                    print("Liveness check passed!")
            elif results.multi_face_landmarks is None and is_live == True:
                is_live = False
                current_challenge = random.choice(CHALLENGES)

            if is_live:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))

                    name, confidence = auth.predict(face)
                    
                    if _screenshot_taken == False:
                        save_first_detection(face, name)
                        _screenshot_taken = True

                    print(f"Predicted: {name} with confidence {confidence:.2f}")

                    if confidence < 85:
                        status_text = f"ACCESS GRANTED: {name} ({confidence:.2f})"
                    else:
                        status_text = f"ACCESS DENIED ({confidence:.2f})"
            else:
                status_text = f"Perform: {current_challenge}"
            
            gray_for_box = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_box = face_cascade.detectMultiScale(gray_for_box, 1.3, 5)

            for (x, y, w, h) in faces_box:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Auth System", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()