import cv2
import os
import json
import face_recognition
from ultralytics import YOLO

class FaceDetector:
    def __init__(self):
        # 1. Load YOLOv8 Face untuk deteksi posisi jarak jauh
        model_path = os.path.join('assets', 'face_yolov8n.pt')
        self.model_yolo = YOLO(model_path)
        
        # 2. Database lokal untuk menampung Face Embeddings
        self.known_face_encodings = []
        self.known_face_metadata = [] 
        
        self.load_karyawan_database()

    def load_karyawan_database(self):
        print("Menganalisis database foto_karyawan menggunakan Deep Learning")
        folder_foto = 'foto_karyawan'
        if not os.path.exists(folder_foto):
            os.makedirs(folder_foto)
            return
            
        for file_name in os.listdir(folder_foto):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                name_part, _ = os.path.splitext(file_name)
                if '_' in name_part:
                    str_id, nama = name_part.split('_', 1)
                    try:
                        int_id = int(str_id)
                        path_foto = os.path.join(folder_foto, file_name)
                        
                        image = face_recognition.load_image_file(path_foto)
                        encodings = face_recognition.face_encodings(image)
                        
                        if len(encodings) > 0:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_metadata.append({'id': int_id, 'nama': nama})
                            print(f"   + Berhasil mendaftarkan: {nama} (ID: {int_id})")
                        else:
                            print(f"   ⚠️ Wajah tidak terbaca pada file: {file_name}")
                    except ValueError:
                        print(f"   ⚠️ Format ID salah pada file: {file_name}")

    def detect(self, frame):
        # Jalankan deteksi posisi wajah via YOLOv8
        results = self.model_yolo(frame, verbose=False)
        active_ids = []
        
        # Konversi frame BGR (OpenCV) ke RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Pengaman koordinat box agar tidak keluar frame
                h_f, w_f, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_f, x2), min(h_f, y2)
                
                # --- PERBAIKAN UTAMA: JANGAN DI-CROP ---
                # Konversi koordinat YOLO ke format CSS (top, right, bottom, left) untuk dlib
                lokasi_wajah = [(y1, x2, y2, x1)]
                
                # Ekstrak embedding langsung dari frame utuh berbasis lokasi YOLOv8
                current_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=lokasi_wajah)
                
                nama = "Unknown"
                warna_box = (0, 0, 255) # Merah default
                
                if len(current_encodings) > 0:
                    face_to_compare = current_encodings[0]
                    
                    # Hitung jarak kedekatan vektor wajah
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_to_compare)
                    
                    if len(face_distances) > 0:
                        import numpy as np
                        best_match_index = np.argmin(face_distances)
                        
                        # Batas toleransi Euclidean Distance standar SOTA dlib (0.6)
                        if face_distances[best_match_index] < 0.48:
                            metadata = self.known_face_metadata[best_match_index]
                            nama = metadata['nama']
                            warna_box = (0, 255, 0) # Hijau jika cocok
                            active_ids.append(metadata['id'])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), warna_box, 2)
                cv2.putText(frame, nama, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_box, 2)
                
        return active_ids, frame