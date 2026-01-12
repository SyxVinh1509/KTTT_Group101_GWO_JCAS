import numpy as np
import matplotlib.pyplot as plt
import os  # <--- Thư viện để xử lý đường dẫn file

# ==========================================
# CẤU HÌNH CỐ ĐỊNH (SEED)
# ==========================================
# Đặt seed để mỗi lần chạy kết quả ra Y HỆT nhau (tốt cho báo cáo)
# Bạn có thể đổi số 42 thành số khác nếu muốn thử vận may khác
np.random.seed(42) 

# ==========================================
# 1. CÁC HÀM HỖ TRỢ
# ==========================================
def get_steering_vector(theta_deg, N):
    theta_rad = np.radians(theta_deg)
    k = np.arange(N).reshape(-1, 1)
    sv = np.exp(1j * np.pi * k * np.sin(theta_rad))
    return sv

def calculate_full_pattern(w, N, resolution=1000):
    angles = np.linspace(-90, 90, resolution)
    k = np.arange(N).reshape(-1, 1)
    theta_rads = np.radians(angles).reshape(1, -1)
    A_scan = np.exp(1j * np.pi * k * np.sin(theta_rads))
    response = np.matmul(w.conj().T, A_scan)
    gains = np.abs(response).flatten()
    return angles, gains

# ==========================================
# 2. THUẬT TOÁN TS-ILS
# ==========================================
def optimize_jcas_ts_ils_original(N, target_angles, max_iter=50):
    M = len(target_angles)
    A = np.hstack([get_steering_vector(ang, N) for ang in target_angles])
    d_mag = np.ones((M, 1)) * N 
    
    # Khởi tạo ngẫu nhiên
    phi = np.random.rand(N, 1) * 2 * np.pi
    w = np.exp(1j * phi)
    
    cost_history = []

    # --- TÍNH SAI SỐ BAN ĐẦU (VÒNG 0) ---
    y_init = np.matmul(A.conj().T, w)
    initial_error = np.linalg.norm(np.abs(y_init) - d_mag)
    cost_history.append(initial_error)
    
    print("-" * 60)
    print(f"BẮT ĐẦU TS-ILS | N={N} | Targets={target_angles}")
    print(f"Iter 000 (Init) | Error: {initial_error:.6f}")
    
    # --- VÒNG LẶP ---
    for i in range(max_iter):
        y_current = np.matmul(A.conj().T, w)
        d_update = d_mag * np.exp(1j * np.angle(y_current))
        w_hat = np.matmul(np.linalg.pinv(A.conj().T), d_update)
        w = np.exp(1j * np.angle(w_hat))
        
        y_new = np.matmul(A.conj().T, w)
        gain_error = np.linalg.norm(np.abs(y_new) - d_mag) 
        cost_history.append(gain_error)
        
        print(f"Iter {i+1:03d}        | Error: {gain_error:.6f}")

    return w, cost_history

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # --- CẤU HÌNH ---
    N_ANTENNAS = 64     
    TARGETS = [0, -40]  
    ITERATIONS = 50
    
    w_opt, costs = optimize_jcas_ts_ils_original(N_ANTENNAS, TARGETS, ITERATIONS)
    
    # Vẽ đồ thị
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # HÌNH 1
    angles, gains = calculate_full_pattern(w_opt, N_ANTENNAS)
    gains_db = 20 * np.log10(gains / np.max(gains) + 1e-12)
    ax1.plot(angles, gains_db, linewidth=1.5, color='blue')
    ax1.set_title(f'JCAS Beam Pattern (N={N_ANTENNAS})')
    ax1.set_xlabel('Angle (degree)')
    ax1.set_ylabel('Normalized Gain (dB)')
    ax1.set_ylim([-60, 0])
    ax1.grid(True, linestyle='--', alpha=0.6)
    for t in TARGETS:
        ax1.axvline(x=t, color='red', linestyle=':', linewidth=1.5, label=f'Target {t}')
    ax1.legend()

    # HÌNH 2
    ax2.plot(range(0, ITERATIONS + 1), costs, marker='.', markersize=8, color='darkgreen', linewidth=2)
    ax2.set_title('Convergence Curve')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error')
    ax2.grid(True)
    
    # --- [QUAN TRỌNG] XỬ LÝ ĐƯỜNG DẪN LƯU FILE ---
    # Lấy đường dẫn của file code hiện tại (file jcas.py đang nằm ở đâu?)
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Tạo tên file cố định
    filename = "JCAS_original.png"
    
    # Ghép đường dẫn: thư mục chứa code + tên file
    full_save_path = os.path.join(current_script_path, filename)
    
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300)
    print(f"\n[DONE] Đã lưu ảnh vào chính xác thư mục chứa code:")
    print(f"Path: {full_save_path}")
    
    plt.show()