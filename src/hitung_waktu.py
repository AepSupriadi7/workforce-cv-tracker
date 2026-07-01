import time
import os
import csv
from datetime import datetime

class TimeTracker:
    def __init__(self):
        self.log_file = os.path.join('logs', 'data_absensi.csv')
        self.total_seconds = {}       
        self.last_detected_time = {}  
        self.labels = {}              

    def sync_labels(self, metadata_list):
        # Sinkronisasi nama karyawan langsung dari folder foto yang terbaca detector
        self.labels = {m['id']: m['nama'] for m in metadata_list}

    def update(self, active_ids):
        waktu_sekarang = time.time()
        
        for int_id in self.labels.keys():
            if int_id in active_ids:
                if self.total_seconds.get(int_id) is None:
                    self.total_seconds[int_id] = 0.0
                    
                if self.last_detected_time.get(int_id) is not None:
                    selisih = waktu_sekarang - self.last_detected_time[int_id]
                    self.total_seconds[int_id] += selisih
                self.last_detected_time[int_id] = waktu_sekarang
            else:
                self.last_detected_time[int_id] = None

    def get_formatted_time(self, int_id):
        detik_total = self.total_seconds.get(int_id, 0.0)
        jam = int(detik_total // 3600)
        menit = int((detik_total % 3600) // 60)
        detik = int(detik_total % 60)
        return f"{jam:02d}:{menit:02d}:{detik:02d}"

    def save_log(self):
        if not self.total_seconds: return
        os.makedirs('logs', exist_ok=True)
        file_terbuat = os.path.isfile(self.log_file)
        tanggal_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_terbuat:
                writer.writerow(['Tanggal Sesi', 'ID', 'Nama Karyawan', 'Total Durasi Kerja'])
            
            for int_id, detik in self.total_seconds.items():
                if detik > 0:
                    nama = self.labels.get(int_id, f"ID {int_id}")
                    waktu_format = self.get_formatted_time(int_id)
                    writer.writerow([tanggal_sekarang, int_id, nama, waktu_format])
                    print(f"[RECORD SAVED] {nama} -> {waktu_format}")