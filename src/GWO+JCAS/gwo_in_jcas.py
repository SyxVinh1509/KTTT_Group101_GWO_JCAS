import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CẤU HÌNH (SEED)
# ==========================================
np.random.seed(42) # Giữ cố định để kết quả lặp lại được

# ==========================================
# 1. CÁC HÀM VẬT LÝ (PHYSICS LAYER)
# ==========================================
def get_steering_vector(theta_deg, N):
    """Tạo vector lái pha"""
    theta_rad = np.radians(theta_deg)
    k = np.arange(N).reshape(-1, 1)
    sv = np.exp(1j * np.pi * k * np.sin(theta_rad))
    return sv

def calculate_full_pattern(phases, N, resolution=1000):
    """
    Tính biểu đồ bức xạ từ các góc pha (Input của GWO là pha)
    phases: Mảng 1 chiều chứa N góc pha
    """
    # Chuyển pha thành trọng số phức: w = e^(j*phi)
    w = np.exp(1j * phases).reshape(-1, 1)
    
    angles = np.linspace(-90, 90, resolution)
    k = np.arange(N).reshape(-1, 1)
    theta_rads = np.radians(angles).reshape(1, -1)
    A_scan = np.exp(1j * np.pi * k * np.sin(theta_rads))
    
    response = np.matmul(w.conj().T, A_scan)
    gains = np.abs(response).flatten()
    return angles, gains

# ==========================================
# 2. HÀM MỤC TIÊU (FITNESS FUNCTION)
# ==========================================
def fitness_function(phases, N, A_targets, d_mag):
    """
    Tính độ lệch (Error) của một con sói.
    phases: Vị trí của sói (Vector các góc pha).
    A_targets: Ma trận lái tại các hướng mục tiêu.
    d_mag: Biên độ mong muốn tại các hướng mục tiêu.
    """
    # 1. Giải mã con sói: Pha -> Trọng số phức
    w = np.exp(1j * phases).reshape(-1, 1)
    
    # 2. Tính phản hồi tại các hướng Target
    y_current = np.matmul(A_targets.conj().T, w)
    
    # 3. Tính sai số (L2 Norm) giữa biên độ thực tế và mong muốn
    # Fitness càng nhỏ càng tốt
    error = np.linalg.norm(np.abs(y_current) - d_mag)
    return error

# ==========================================
# 3. THUẬT TOÁN GWO (CORE)
# ==========================================
def run_GWO_JCAS(N, target_angles, n_wolves=50, max_iter=100):
    
    # --- A. CHUẨN BỊ DỮ LIỆU ---
    M = len(target_angles)
    # Tạo ma trận lái tại các hướng Target (tính 1 lần dùng mãi mãi)
    A_targets = np.hstack([get_steering_vector(ang, N) for ang in target_angles])
    # Biên độ mong muốn (Gain = N)
    d_mag = np.ones((M, 1)) * N
    
    # --- B. KHỞI TẠO BẦY SÓI ---
    # LB = 0, UB = 2*pi (Góc pha từ 0 đến 2pi)
    lb = 0
    ub = 2 * np.pi
    dim = N # Số chiều = Số ăng-ten
    
    # Vị trí bầy sói (n_wolves x 64)
    Positions = np.random.uniform(0, 1, (n_wolves, dim)) * (ub - lb) + lb
    
    # Khởi tạo 3 con đầu đàn
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    
    convergence_curve = []
    
    print("-" * 60)
    print(f"BẮT ĐẦU GWO CHO JCAS | Wolves={n_wolves} | Iterations={max_iter}")
    print("-" * 60)

    # --- C. VÒNG LẶP SĂN MỒI ---
    for l in range(0, max_iter):
        
        # 1. Đánh giá Fitness cho từng con sói
        for i in range(0, n_wolves):
            
            # Xử lý biên (Nếu pha < 0 hoặc > 2pi thì ép lại, hoặc để tự do cũng được vì cos tuần hoàn)
            # Nhưng để GWO ổn định, ta nên clip
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
            # Tính điểm
            fitness = fitness_function(Positions[i, :], N, A_targets, d_mag)
            
            # Cập nhật Alpha, Beta, Delta
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # 2. Cập nhật vị trí các con sói (GWO logic)
        a = 2 - l * ((2) / max_iter) # a giảm từ 2 xuống 0
        
        for i in range(0, n_wolves):
            for j in range(0, dim):
                
                # Sói Alpha
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                
                # Sói Beta
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                # Sói Delta
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                # Vị trí mới
                Positions[i, j] = (X1 + X2 + X3) / 3
        
        # Lưu lịch sử
        convergence_curve.append(Alpha_score)
        
        # In log ra terminal (Yêu cầu số 2)
        print(f"Iter {l+1:03d}/{max_iter} | Best Fitness: {Alpha_score:.6f}")

    return Alpha_pos, convergence_curve

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # --- Cấu hình ---
    N_ANTENNAS = 64
    TARGETS = [0, -40]
    ITERATIONS = 100    # GWO cần nhiều vòng lặp hơn TS-ILS một chút
    N_WOLVES = 50       # Số lượng sói
    
    # --- Chạy GWO ---
    # Kết quả trả về là best_phases (Alpha_pos)
    best_phases, costs = run_GWO_JCAS(N_ANTENNAS, TARGETS, n_wolves=N_WOLVES, max_iter=ITERATIONS)
    
    # --- Vẽ đồ thị (Yêu cầu số 3) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # HÌNH 1: Beam Pattern
    angles, gains = calculate_full_pattern(best_phases, N_ANTENNAS)
    gains_db = 20 * np.log10(gains / np.max(gains) + 1e-12)
    
    ax1.plot(angles, gains_db, linewidth=1.5, color='darkorange', label='GWO Optimized')
    ax1.set_title(f'JCAS Beam Pattern (GWO, N={N_ANTENNAS})')
    ax1.set_xlabel('Angle (degree)')
    ax1.set_ylabel('Normalized Gain (dB)')
    ax1.set_ylim([-60, 0])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for t in TARGETS:
        ax1.axvline(x=t, color='red', linestyle=':', linewidth=1.5, label=f'Target {t}')
    ax1.legend()

    # HÌNH 2: Convergence Curve
    ax2.plot(range(1, ITERATIONS + 1), costs, marker='.', markersize=6, color='purple', linewidth=1.5)
    ax2.set_title('Convergence Curve (GWO)')
    ax2.set_xlabel('Iteration (Vòng lặp)')
    ax2.set_ylabel('Fitness Value (Error)')
    ax2.grid(True)
    
    # --- Lưu file (Yêu cầu số 4) ---
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    filename = "GWO_in_JCAS_result.png"
    full_save_path = os.path.join(current_script_path, filename)
    
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300)
    
    print("\n" + "="*60)
    print(f"[HOÀN THÀNH] Đã chạy xong GWO.")
    print(f"Ảnh kết quả đã được lưu tại: {full_save_path}")
    print("="*60)
    
    plt.show()