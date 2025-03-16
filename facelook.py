import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
import numpy as np
import pickle
from datetime import datetime

class FaceRecogApp:
    def __init__(self, window):
        self.window = window
        self.window.title("FaceRecog - DNN Face Recognition")
        self.window.geometry("800x600")

        # Load DNN face detector
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Initialize LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_running = False
        self.training_data_dir = "training_data"
        
        # Load existing model if available
        self.labels = {}
        self.current_id = 0
        if os.path.exists("trained_model.yml") and os.path.exists("labels.pickle"):
            self.recognizer.read("trained_model.yml")
            with open("labels.pickle", 'rb') as f:
                self.labels = pickle.load(f)
            self.current_id = max(self.labels.values(), default=-1) + 1
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # UI Elements
        self.video_label = tk.Label(window)
        self.video_label.pack(pady=10)
        
        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Start Camera", 
                 command=self.start_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop Camera", 
                 command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Add New Face", 
                 command=self.add_new_face).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Train Model", 
                 command=self.train_model).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", 
                 command=self.quit_app).pack(side=tk.LEFT, padx=5)
        
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
        
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret and self.face_running:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w-1, endX), min(h-1, endY)
                    
                    # Recognition
                    face = frame[startY:endY, startX:endX]
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    try:
                        id_, conf = self.recognizer.predict(gray_face)
                        label = [name for name, idx in self.labels.items() if idx == id_][0]
                        text = f"{label} ({conf:.2f})" if conf < 100 else "Unknown"
                    except:
                        text = "Unknown"
                    
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, startY-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_frame)

    def start_camera(self):
        self.face_running = True
        messagebox.showinfo("Info", "Camera started with DNN detection")

    def stop_camera(self):
        self.face_running = False
        messagebox.showinfo("Info", "Camera stopped")

    def add_new_face(self):
        name = simpledialog.askstring("Input", "Enter person's name:")
        if not name:
            return
        
        person_dir = os.path.join(self.training_data_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        count = 0
        messagebox.showinfo("Info", "Look at the camera. Capturing 30 samples...")
        
        while count < 30:
            ret, frame = self.cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
                )
                self.net.setInput(blob)
                detections = self.net.forward()
                
                if detections.shape[2] > 0:
                    i = np.argmax(detections[0, 0, :, 2])
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        (h, w) = frame.shape[:2]
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w-1, endX), min(h-1, endY)
                        
                        face = frame[startY:endY, startX:endX]
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        filename = f"{person_dir}/sample_{count}.jpg"
                        cv2.imwrite(filename, gray_face)
                        count += 1
                        
                        cv2.imshow("Capturing", face)
                        cv2.waitKey(100)
        
        cv2.destroyAllWindows()
        if name not in self.labels:
            self.labels[name] = self.current_id
            self.current_id += 1
        messagebox.showinfo("Success", f"Collected 30 samples for {name}")

    def train_model(self):
        faces = []
        ids = []
        
        for person_name, person_id in self.labels.items():
            person_dir = os.path.join(self.training_data_dir, person_name)
            for image_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                ids.append(person_id)
        
        if faces:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save("trained_model.yml")
            with open("labels.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
            messagebox.showinfo("Success", "Model trained successfully")
        else:
            messagebox.showwarning("Warning", "No training data found")

    def quit_app(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            self.window.destroy()

def main():
    if not (os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")):
        print("Error: DNN model files not found. Please download:")
        print("1. deploy.prototxt")
        print("2. res10_300x300_ssd_iter_140000.caffemodel")
        return
    
    root = tk.Tk()
    app = FaceRecogApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()