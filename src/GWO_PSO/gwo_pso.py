import numpy as np

# ======================================================
# PHẦN 1: HÀM MỤC TIÊU (PLACEHOLDER)
# ======================================================
def fitness_function(position):
    """
    Hàm này nhận vào vị trí (bộ tham số).
    Trả về: Điểm số (Fitness).
    Hiện tại trả về tổng bình phương (Sphere function) để test.
    """
    return np.sum(position**2)

# ======================================================
# PHẦN 2: THUẬT TOÁN HYBRID GWO-PSO
# ======================================================
def Hybrid_GWO_PSO(N_WOLVES, MAX_ITER, DIM, LB, UB):
    """
    Phiên bản lai ghép: Vừa bao vây (GWO) vừa có quán tính (PSO)
    """
    
    # --- 1. Initialize parameters ---
    # PSO parameters
    w = 0.5      # Trọng số quán tính (Inertia weight)
    c1 = 1.5     # Hệ số học tập 1
    c2 = 1.5     # Hệ số học tập 2
    
    # --- 2. Randomly initialize wolf positions and velocities ---
    # Vị trí X
    Positions = np.random.uniform(0, 1, (N_WOLVES, DIM)) * (UB - LB) + LB
    # Vận tốc V (Khởi tạo bằng 0 hoặc random nhỏ)
    Velocities = np.zeros((N_WOLVES, DIM))
    
    # Khởi tạo Alpha, Beta, Delta
    Alpha_pos = np.zeros(DIM)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(DIM)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(DIM)
    Delta_score = float("inf")
    
    # Tính fitness ban đầu để tìm Alpha, Beta, Delta
    for i in range(N_WOLVES):
        fitness = fitness_function(Positions[i, :])
        if fitness < Alpha_score:
            Alpha_score = fitness
            Alpha_pos = Positions[i, :].copy()
        elif fitness < Beta_score:
            Beta_score = fitness
            Beta_pos = Positions[i, :].copy()
        elif fitness < Delta_score:
            Delta_score = fitness
            Delta_pos = Positions[i, :].copy()

    print("BẮT ĐẦU HYBRID GWO-PSO...")
    
    # --- 3. Main Loop (For iter = 1 to MAX_ITER) ---
    for l in range(0, MAX_ITER):
        
        # Update control parameter a (decrease from 2 to 0)
        a = 2 - l * ((2) / MAX_ITER)
        
        # For each wolf i
        for i in range(0, N_WOLVES):
            
            # Để tối ưu tốc độ Python, ta xử lý nguyên mảng (vectorization)
            # thay vì for j in range(DIM). Logic toán học vẫn y hệt.
            
            # ==========================================
            # PHASE 1: GREY WOLF UPDATING (Tính X_GWO)
            # ==========================================
            
            # --- Alpha ---
            r1 = np.random.random(DIM)
            r2 = np.random.random(DIM)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * Alpha_pos - Positions[i, :])
            X1 = Alpha_pos - A1 * D_alpha
            
            # --- Beta ---
            r1 = np.random.random(DIM)
            r2 = np.random.random(DIM)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * Beta_pos - Positions[i, :])
            X2 = Beta_pos - A2 * D_beta
            
            # --- Delta ---
            r1 = np.random.random(DIM)
            r2 = np.random.random(DIM)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * Delta_pos - Positions[i, :])
            X3 = Delta_pos - A3 * D_delta
            
            # Estimate new GWO position
            X_GWO = (X1 + X2 + X3) / 3.0
            
            # ==========================================
            # PHASE 2: PSO VELOCITY UPDATE (Tính X_PSO)
            # ==========================================
            
            r1_pso = np.random.random(DIM)
            r2_pso = np.random.random(DIM)
            
            # Cập nhật vận tốc dựa trên Alpha và Beta (theo mã giả)
            # V = w*V + c1*r1*(Alpha - X) + c2*r2*(Beta - X)
            Velocities[i, :] = (w * Velocities[i, :] + 
                                c1 * r1_pso * (Alpha_pos - Positions[i, :]) + 
                                c2 * r2_pso * (Beta_pos - Positions[i, :]))
            
            # Tính vị trí theo PSO
            X_PSO = Positions[i, :] + Velocities[i, :]
            
            # ==========================================
            # PHASE 3: HYBRID COMBINATION
            # ==========================================
            
            # Kết hợp 50-50
            Positions[i, :] = 0.5 * X_GWO + 0.5 * X_PSO
            
            # Bound control (Kiểm tra biên)
            Positions[i, :] = np.clip(Positions[i, :], LB, UB)
            
        # --- Evaluate new fitness & Update Leaders ---
        for i in range(0, N_WOLVES):
            fitness = fitness_function(Positions[i, :])
            
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
                
        print(f"Vòng lặp {l+1}: Best Fitness = {Alpha_score}")

    return Alpha_pos, Alpha_score

# ======================================================
# CHẠY THỬ
# ======================================================
if __name__ == "__main__":
    # Cấu hình
    dim = 64            
    n_wolves = 30 
    max_iter = 50       
    lb = -10            
    ub = 10             
    
    best_pos, best_score = Hybrid_GWO_PSO(n_wolves, max_iter, dim, lb, ub)
    
    print("-" * 30)
    print(f"Kết quả Hybrid GWO-PSO:")
    print(f"Giá trị tốt nhất: {best_score}")