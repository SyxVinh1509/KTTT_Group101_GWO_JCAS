import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CẤU HÌNH (SEED)
# ==========================================
np.random.seed(42) # Giữ cố định để kết quả ổn định

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
    Tính biểu đồ bức xạ từ các góc pha.
    phases: Mảng 1 chiều chứa N góc pha.
    """
    # Chuyển pha thành trọng số phức: w = e^(j*phi)
    w_complex = np.exp(1j * phases).reshape(-1, 1)
    
    angles = np.linspace(-90, 90, resolution)
    k = np.arange(N).reshape(-1, 1)
    theta_rads = np.radians(angles).reshape(1, -1)
    A_scan = np.exp(1j * np.pi * k * np.sin(theta_rads))
    
    response = np.matmul(w_complex.conj().T, A_scan)
    gains = np.abs(response).flatten()
    return angles, gains

# ==========================================
# 2. HÀM MỤC TIÊU (FITNESS FUNCTION)
# ==========================================
def fitness_function(phases, A_targets, d_mag):
    """
    Tính độ lệch (Error) của cấu hình pha.
    """
    w_complex = np.exp(1j * phases).reshape(-1, 1)
    
    # Tính phản hồi tại các hướng Target
    y_current = np.matmul(A_targets.conj().T, w_complex)
    
    # Tính sai số biên độ (L2 Norm)
    error = np.linalg.norm(np.abs(y_current) - d_mag)
    return error

# ==========================================
# 3. THUẬT TOÁN HYBRID GWO-PSO (CORE)
# ==========================================
def run_Hybrid_GWO_PSO_JCAS(N, target_angles, n_wolves=30, max_iter=50):
    
    # --- A. CHUẨN BỊ DỮ LIỆU ---
    M = len(target_angles)
    A_targets = np.hstack([get_steering_vector(ang, N) for ang in target_angles])
    d_mag = np.ones((M, 1)) * N
    
    # --- B. KHỞI TẠO PARAMETERS ---
    lb = 0
    ub = 2 * np.pi
    dim = N
    
    # PSO Parameters
    w_pso = 0.5    # Inertia weight
    c1 = 1.5       # Hệ số hướng về Alpha
    c2 = 1.5       # Hệ số hướng về Beta
    
    # Khởi tạo Vị trí (X) và Vận tốc (V)
    Positions = np.random.uniform(0, 1, (n_wolves, dim)) * (ub - lb) + lb
    Velocities = np.zeros((n_wolves, dim))
    
    # Khởi tạo Alpha, Beta, Delta
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    
    convergence_curve = []
    
    # Đánh giá ban đầu
    for i in range(n_wolves):
        fitness = fitness_function(Positions[i, :], A_targets, d_mag)
        if fitness < Alpha_score:
            Alpha_score = fitness
            Alpha_pos = Positions[i, :].copy()
        elif fitness < Beta_score:
            Beta_score = fitness
            Beta_pos = Positions[i, :].copy()
        elif fitness < Delta_score:
            Delta_score = fitness
            Delta_pos = Positions[i, :].copy()
            
    print("-" * 70)
    print(f"BẮT ĐẦU HYBRID GWO-PSO | Wolves={n_wolves} | Iter={max_iter}")
    print("-" * 70)

    # --- C. VÒNG LẶP SĂN MỒI ---
    for l in range(0, max_iter):
        
        # Cập nhật tham số a (giảm tuyến tính từ 2 về 0)
        a = 2 - l * ((2) / max_iter)
        
        for i in range(0, n_wolves):
            # ==========================================
            # PHASE 1: GREY WOLF UPDATING (Tính X_GWO)
            # ==========================================
            # Alpha
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * Alpha_pos - Positions[i, :])
            X1 = Alpha_pos - A1 * D_alpha
            
            # Beta
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * Beta_pos - Positions[i, :])
            X2 = Beta_pos - A2 * D_beta
            
            # Delta
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * Delta_pos - Positions[i, :])
            X3 = Delta_pos - A3 * D_delta
            
            # Vị trí dự đoán theo GWO
            X_GWO = (X1 + X2 + X3) / 3.0
            
            # ==========================================
            # PHASE 2: PSO VELOCITY UPDATE (Tính X_PSO)
            # ==========================================
            r1_pso = np.random.random(dim)
            r2_pso = np.random.random(dim)
            
            # Cập nhật vận tốc (Dùng Alpha và Beta để dẫn đường)
            Velocities[i, :] = (w_pso * Velocities[i, :] + 
                                c1 * r1_pso * (Alpha_pos - Positions[i, :]) + 
                                c2 * r2_pso * (Beta_pos - Positions[i, :]))
            
            # Vị trí dự đoán theo PSO
            X_PSO = Positions[i, :] + Velocities[i, :]
            
            # ==========================================
            # PHASE 3: HYBRID COMBINATION & BOUND CHECK
            # ==========================================
            # Kết hợp 50-50
            Positions[i, :] = 0.5 * X_GWO + 0.5 * X_PSO
            
            # Kiểm tra biên (0 đến 2pi)
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
        # --- Đánh giá lại Fitness ---
        for i in range(n_wolves):
            fitness = fitness_function(Positions[i, :], A_targets, d_mag)
            
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # Lưu lịch sử
        convergence_curve.append(Alpha_score)
        
        # [YÊU CẦU 2] In rõ từng bước ra terminal
        print(f"Iter {l+1:03d}/{max_iter} | Best Fitness (Error): {Alpha_score:.6f}")

    return Alpha_pos, convergence_curve

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # --- Cấu hình ---
    N_ANTENNAS = 64
    TARGETS = [0, -40]
    ITERATIONS = 100    
    N_WOLVES = 30       
    
    # [YÊU CẦU 1] Chạy Hybrid GWO-PSO thay vì TS-ILS
    best_phases, costs = run_Hybrid_GWO_PSO_JCAS(N_ANTENNAS, TARGETS, n_wolves=N_WOLVES, max_iter=ITERATIONS)
    
    # [YÊU CẦU 3] Vẽ 2 ảnh (Beam Pattern & Convergence)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # HÌNH 1: Beam Pattern
    angles, gains = calculate_full_pattern(best_phases, N_ANTENNAS)
    gains_db = 20 * np.log10(gains / np.max(gains) + 1e-12)
    
    ax1.plot(angles, gains_db, linewidth=1.5, color='magenta', label='Hybrid GWO-PSO')
    ax1.set_title(f'JCAS Beam Pattern (Hybrid GWO-PSO, N={N_ANTENNAS})')
    ax1.set_xlabel('Angle (degree)')
    ax1.set_ylabel('Normalized Gain (dB)')
    ax1.set_ylim([-60, 0])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for t in TARGETS:
        ax1.axvline(x=t, color='red', linestyle=':', linewidth=1.5, label=f'Target {t}')
    ax1.legend()

    # HÌNH 2: Convergence Curve
    ax2.plot(range(1, ITERATIONS + 1), costs, marker='.', markersize=6, color='darkblue', linewidth=1.5)
    ax2.set_title('Convergence Curve (Hybrid GWO-PSO)')
    ax2.set_xlabel('Iteration (Vòng lặp)')
    ax2.set_ylabel('Fitness Value (Error)')
    ax2.grid(True)
    
    # [YÊU CẦU 4] Lưu file ảnh với tên cố định
    # --- CƠ CHẾ TÌM ĐƯỜNG DẪN TUYỆT ĐỐI ---
    try:
        # Cách 1: Thử lấy đường dẫn của file script đang chạy
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Cách 2: Nếu chạy trong môi trường tương tác (như Jupyter), dùng thư mục làm việc hiện tại
        current_dir = os.getcwd()
        
    filename = "GWO_PSO_in_JCAS_result.png"
    full_save_path = os.path.join(current_dir, filename)
    
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300)
    
    print("\n" + "="*70)
    print(f"[THÀNH CÔNG] Ảnh đã được lưu tại:")
    print(f"{full_save_path}")  # <--- HÃY COPY ĐƯỜNG DẪN NÀY DÁN VÀO FILE EXPLORER
    print("="*70)
    
    plt.show()