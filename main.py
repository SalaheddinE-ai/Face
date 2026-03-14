import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ---------------- CONFIG ----------------
DB_PATH = "./base_eleves"
LOG_FILE = "registre_presences.csv"
MODEL_YOLO = "yolo26n-seg.pt"

# ---------------- LIVENESS MANAGER ----------------
class LivenessManager:
    def __init__(self):
        BaseOptions = python.BaseOptions
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        
        self.EAR_THRESHOLD = 0.22
        self.blink_counter = 0
        self.eye_closed = False
        self.challenge_direction = "DROITE"
        self.timestamp_ms = 0  # For video mode

    def _calculate_ear(self, landmarks):
        def get_dist(p1, p2):
            return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

        # Left Eye EAR (same indices)
        v1 = get_dist(landmarks[159], landmarks[145])
        v2 = get_dist(landmarks[158], landmarks[153])
        h1 = get_dist(landmarks[33], landmarks[133])
        ear_left = (v1 + v2) / (2.0 * h1 + 1e-6)

        # Right Eye EAR
        v3 = get_dist(landmarks[385], landmarks[373])
        v4 = get_dist(landmarks[387], landmarks[380])
        h2 = get_dist(landmarks[362], landmarks[263])
        ear_right = (v3 + v4) / (2.0 * h2 + 1e-6)

        return (ear_left + ear_right) / 2.0

    def check_liveness(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.timestamp_ms += 33  # ~30 FPS
        
        results = self.face_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        status = {"blink_count": self.blink_counter, "direction": "CENTRE", "ready": False}

        if results.face_landmarks and len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]  # NormalizedLandmarkList
            
            ear = self._calculate_ear(landmarks)
            if ear < self.EAR_THRESHOLD:
                if not self.eye_closed:
                    self.blink_counter += 1
                    self.eye_closed = True
            else:
                self.eye_closed = False

            # Head pose (nose tip index 1)
            nose = landmarks[1].x
            l_bound, r_bound = landmarks[234].x, landmarks[454].x
            rel = (nose - l_bound) / (r_bound - l_bound + 1e-6)

            if rel < 0.38: status["direction"] = "DROITE"
            elif rel > 0.62: status["direction"] = "GAUCHE"

            if self.blink_counter >= 2 and status["direction"] == self.challenge_direction:
                status["ready"] = True
            
            status["blink_count"] = self.blink_counter

        return status

    def reset(self):
        self.blink_counter = 0
        self.eye_closed = False
# ---------------- APPLICATION ----------------
class AppPresence:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        self.window.geometry("1100x700")

        print("Booting Vision Engines...")
        self.yolo_model = YOLO(MODEL_YOLO)
        self.liveness = LivenessManager()
        self.eleves_deja_presents = set()

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        self.window.mainloop()

    def setup_ui(self):
        self.label_video = tk.Label(self.window)
        self.label_video.pack(side=tk.LEFT, padx=10, pady=10)

        self.sidebar = tk.Frame(self.window)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        tk.Label(self.sidebar, text="SYSTEM STATUS", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_instr = tk.Label(self.sidebar, text="Challenge:\nClignez 2 fois\nTournez à DROITE", 
                                  fg="#0055ff", font=("Arial", 11, "italic"))
        self.lbl_instr.pack(pady=10)

        self.tree = ttk.Treeview(self.sidebar, columns=("Nom", "Heure"), show="headings", height=15)
        self.tree.heading("Nom", text="Étudiant")
        self.tree.heading("Heure", text="Heure")
        self.tree.column("Nom", width=120)
        self.tree.column("Heure", width=80)
        self.tree.pack()

        tk.Button(self.sidebar, text="RESET LIVENESS", command=self.liveness.reset, bg="orange").pack(pady=10)
        tk.Button(self.sidebar, text="QUITTER", command=self.quitter, bg="#cc0000", fg="white").pack(pady=5)

    def recognize_face(self, face_crop):
        try:
            # Using ArcFace for superior embedding separation
            res = DeepFace.find(img_path=face_crop, db_path=DB_PATH, 
                                model_name="ArcFace", enforce_detection=False, silent=True)
            if len(res[0]) > 0:
                path = res[0]["identity"][0]
                return os.path.basename(path).split(".")[0]
        except Exception as e:
            print(f"DeepFace Engine Error: {e}")
        return None

    def update_frame(self):
        success, frame = self.cap.read()
        if not success: return

        status = self.liveness.check_liveness(frame)

        # UI Overlays
        hud_color = (0, 255, 0) if status["ready"] else (255, 255, 0)
        cv2.putText(frame, f"Blinks: {status['blink_count']}/2", (20,40), 1, 1.5, hud_color, 2)
        cv2.putText(frame, f"Pose: {status['direction']}", (20,80), 1, 1.5, hud_color, 2)

        if status["ready"]:
            # Draw visual feedback for authentication state
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 12)
            
            # Use YOLO only when liveness is confirmed (Performance Optimization)
            results = self.yolo_model(frame, conf=0.6, verbose=False)
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        nom = self.recognize_face(face_crop)
                        if nom:
                            self.enregistrer(nom)
                            cv2.putText(frame, f"VERIFIED: {nom}", (x1, y1-10), 1, 1.2, (0,255,0), 2)

        # Render to Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)
        self.window.after(10, self.update_frame)

    def enregistrer(self, nom):
        if nom in self.eleves_deja_presents: return
        
        maintenant = datetime.now()
        heure = maintenant.strftime("%H:%M:%S")
        date = maintenant.strftime("%Y-%m-%d")

        df = pd.DataFrame({"Nom": [nom], "Date": [date], "Heure": [heure]})
        df.to_csv(LOG_FILE, mode="a", index=False, header=not os.path.exists(LOG_FILE), sep=";")
        
        self.eleves_deja_presents.add(nom)
        self.tree.insert("", tk.END, values=(nom, heure))
        print(f"Access Granted: {nom}")

    def quitter(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
    root = tk.Tk()
    AppPresence(root, "Sentinel Presence 2026")