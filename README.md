# face-auth-liveness
Design and implementation of a Facial Authentication System with a Liveness Detection (Active-Challenge Response Mechanism)

# Face Authentication with Active Liveness Detection

## 📌 Overview

This project implements a lightweight face authentication system with **active liveness detection (challenge–response)**.

The system ensures that:

1. A real person is present (not a photo/video attack)
2. The person is authenticated against stored identities

---

## 🚀 Features

* Real-time face detection using MediaPipe
* Active liveness detection:
  * Blink detection
  * Head turn (left/right)
  * Smile detection
* Face authentication using LBPH (OpenCV)
* Lightweight and fast (runs on CPU)

---

## 📁 Project Structure

```
project/
├── data/
│   ├── detetcet_people/
│   │    ├── user_1/
│   │    └── user_2/
│   └── enrolled/
│       ├── user_1/
│       └── user_2/
├── models/
│   ├── lbph_model.yml
│   └── label_map.json
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── face_auth.py
│   ├── features.py
│   ├── liveness.py
│   ├── main.py
│   ├── screenshot.py
│   ├── snapshot.py
│   └── train.py
└── requirements.txt
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📷 Prepare Dataset

1. Create folders for each user:

```
data/enrolled/user_1/
data/enrolled/user_2/
```

2. Check line 22 in snapshot.py to set proper user folder
 
3. Run ```python src/snapshot ```

---

## 🧠 Train the Model

```bash
python src/train.py
```

This will generate:

* `models/lbph_model.yml`
* `models/label_map.json`

---

## ▶️ Run the System

```bash
python src/main.py
```

---

## 🎯 How It Works

### Step 1: Liveness Detection

User must complete actions:

* Blink
* Turn head
* Smile

### Step 2: Authentication

If liveness is confirmed:

* Face is compared with trained model

---

## ⚠️ Notes
* Good lighting improves accuracy
* Works best with frontal face

---