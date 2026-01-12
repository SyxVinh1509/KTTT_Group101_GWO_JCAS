import numpy as np

# ======================================================
# PHẦN 1: HÀM MỤC TIÊU (PLACEHOLDER)
# Đây là nơi sau này bạn sẽ "cắm" bài toán JCAS vào
# ======================================================
def fitness_function(position):
    """
    Hàm này nhận vào vị trí của một con sói (1 bộ tham số).
    Trả về: Điểm số (Fitness). Càng nhỏ càng tốt.
    Hiện tại để trống (pass) hoặc trả về random để code không lỗi cú pháp.
    """
    # Sau này bạn sẽ viết code tính sai số ăng-ten ở đây
    return 0 

# ======================================================
# PHẦN 2: THUẬT TOÁN GWO (BÁM SÁT MÃ GIẢ)
# ======================================================
def GWO(SearchAgents_no, Max_iter, dim, lb, ub):
    """
    SearchAgents_no: Số lượng sói (n)
    Max_iter: Số vòng lặp tối đa
    dim: Số chiều bài toán (số biến cần tìm)
    lb, ub: Giới hạn dưới, giới hạn trên
    """

    # --- 1. Initialize the grey wolf population Xi ---
    # Tạo bầy sói ngẫu nhiên trong phạm vi [lb, ub]
    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    
    # Khởi tạo vị trí và điểm số của Alpha, Beta, Delta
    # Ban đầu cho điểm số là vô cùng lớn (inf) để dễ tìm min
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    # Biến đếm vòng lặp (t trong mã giả)
    t = 0 

    print("BẮT ĐẦU THUẬT TOÁN GWO...")

    # --- 2. While (t < Max number of iterations) ---
    while t < Max_iter:
        
        # --- 2a. Calculate the fitness of each search agent ---
        # (Trong mã giả bước này nằm trong vòng lặp, nhưng để tối ưu code,
        # ta thường đưa bước cập nhật Alpha/Beta/Delta lên đầu hoặc cuối vòng lặp)
        
        for i in range(0, SearchAgents_no):
            
            # Xử lý biên (Boundary check): Nếu sói chạy ra ngoài thì kéo lại
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
            # Tính fitness (gọi hàm mục tiêu)
            fitness = fitness_function(Positions[i, :])
            
            # --- Update X_alpha, X_beta, X_delta ---
            if fitness < Alpha_score:
                # Nếu tốt hơn Alpha thì Alpha xuống Beta, Beta xuống Delta...
                # Nhưng code đơn giản thì gán trực tiếp:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
                
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # --- 2b. Update a, A, and C ---
        # a giảm tuyến tính từ 2 về 0 theo công thức: a = 2 - t * (2/Max_iter)
        a = 2 - t * ((2) / Max_iter)
        
        # --- 2c. For each search agent (Update Position) ---
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                
                # Tính toán dựa trên Alpha
                r1 = np.random.random() # Random [0,1]
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                # D_alpha = |C1 * X_alpha - X|
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]) 
                # X1 = X_alpha - A1 * D_alpha
                X1 = Alpha_pos[j] - A1 * D_alpha 
                
                # Tính toán dựa trên Beta
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                # Tính toán dựa trên Delta
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                # --- Update position by equation (3.7) ---
                # Vị trí mới là trung bình cộng của 3 vector hướng
                Positions[i, j] = (X1 + X2 + X3) / 3
        
        # --- 2d. t = t + 1 ---
        t = t + 1
        print(f"Vòng lặp {t} hoàn thành.")

    # --- 3. Return X_alpha ---
    return Alpha_pos, Alpha_score

# ======================================================
# CHẠY THỬ (TEST)
# ======================================================
if __name__ == "__main__":
    # Các tham số giả định
    dim = 64            # Ví dụ: 64 biến số
    SearchAgents_no = 10 # 10 con sói
    Max_iter = 50       # 50 vòng lặp
    lb = -10            # Giới hạn dưới
    ub = 10             # Giới hạn trên
    
    # Gọi hàm
    best_pos, best_score = GWO(SearchAgents_no, Max_iter, dim, lb, ub)
    
    print("\nThuật toán đã chạy xong cấu trúc (chưa có bài toán thực tế).")