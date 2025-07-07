# 🖐️ AirDraw-Gesture-Canvas

A gesture-controlled virtual canvas for drawing, erasing, and saving using real-time hand tracking.

---

## 🚀 Features

- ✋ Real-time hand tracking using **MediaPipe**
- 🎨 Draw using index finger with smooth stroke filtering
- 🧽 Erase using two-finger gesture
- 📄 Multi-page canvas with swipe gesture to switch pages
- 🎯 Color palette and brush thickness control
- 🗑️ Clear canvas with a gesture tap
- 💾 Save all pages as a **multi-page PDF**
- 🖼️ Clean PyQt5 GUI with live webcam feed

---

## 🛠️ Tech Stack

- **Python 3.7+**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **PyQt5**
- **Pillow (PIL)**

---

## 🖥️ How It Works

- The system captures your webcam feed using OpenCV.
- MediaPipe detects and tracks hand landmarks in real-time.
- Gestures are interpreted:
  - **1 finger up** → Draw
  - **2 fingers up** → Erase
  - **5 fingers** → Select color / clear / save
  - **3 fingers (swipe)** → Switch between canvas pages
- A Kalman Filter smoothens cursor movement.
- Canvas is drawn on an OpenCV image layer and updated live in the PyQt interface.
