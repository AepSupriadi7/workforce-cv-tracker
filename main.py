import cv2
import os
from src.deteksi import FaceDetector
from src.hitung_waktu import TimeTracker

def main():
    # SETTING KAMERA: 0 = Webcam laptop. 
    sumber_kamera = 1 
    cap = cv2.VideoCapture(sumber_kamera)
    
    detector = FaceDetector()
    tracker = TimeTracker()
    
    # Sinkronisasi list nama dari detektor ke tracker waktu
    tracker.sync_labels(detector.known_face_metadata)
    
    # Otomatis reset file log lama agar bersih di awal sesi pengujian baru
    log_path = os.path.join('logs', 'data_absensi.csv')
    if os.path.exists(log_path):
        try: os.remove(log_path)
        except: pass

    print("SISTEM PEMANTAU KARYAWAN HYBRID (YOLOv8 + DL RESNET) AKTIF!")
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # 1. Jalankan Deteksi & Recognition
            active_ids, frame = detector.detect(frame)
            
            # 2. Update data timer real-time
            tracker.update(active_ids)
            
            # 3. HUD Kiri Atas: Menampilkan status kehadiran semua karyawan terdaftar
            y_offset = 40
            for int_id, nama in tracker.labels.items():
                is_present = int_id in active_ids
                status_text = f"STATUS {nama}: HADIR" if is_present else f"STATUS {nama}: TIDAK DI TEMPAT"
                status_color = (0, 255, 0) if is_present else (0, 0, 255)
                cv2.putText(frame, status_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                y_offset += 25
                
            # 4. HUD Bawah Layar: Menampilkan akumulasi waktu kerja karyawan
            y_bottom = frame.shape[0] - 30
            for int_id, nama in tracker.labels.items():
                waktu_kerja = tracker.get_formatted_time(int_id)
                cv2.putText(frame, f"Waktu {nama}: {waktu_kerja} / 08:00:00", (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_bottom -= 25

            cv2.imshow("Monitor Pemantau Karyawan", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Sistem dihentikan paksa via terminal.")
    finally:
        tracker.save_log()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Log absensi tersimpan dengan aman.")

if __name__ == "__main__":
    main()