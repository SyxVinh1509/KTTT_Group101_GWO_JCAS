# Nghiên cứu và Ứng dụng Thuật toán Hybrid GWO-PSO cho Bài toán Tối ưu hóa Truyền thông Đa chùm (JCAS)

**Học phần:** Nhập môn Kỹ thuật Truyền thông  
**Nhóm thực hiện:** 101  
**Trường:** Đại học Bách Khoa Hà Nội (HUST)

## 📖 Giới thiệu (Introduction)

Dự án này tập trung nghiên cứu và giải quyết bài toán tối ưu hóa vector pha cho hệ thống anten trong truyền thông tích hợp cảm nhận (Joint Communication and Sensing - **JCAS**). 

Chúng tôi so sánh hiệu năng của ba phương pháp:
1.  **TS-ILS (Two-Step Iterative Least Squares):** Thuật toán gốc dựa trên phương pháp giải tích.
2.  **GWO (Grey Wolf Optimizer):** Thuật toán bầy đàn cơ bản.
3.  **Hybrid GWO-PSO:** Thuật toán lai ghép đề xuất, kết hợp khả năng tìm kiếm toàn cục của GWO và tốc độ hội tụ của PSO.

## 📂 Cấu trúc Thư mục (Project Structure)

```text
PRJ_GWO/
├── SRC/
│   ├── GWO/
│   │   └── gwo.py                # Cài đặt thuật toán GWO cơ bản (Benchmark)
│   │
│   ├── GWO_PSO/
│   │   └── gwo_pso.py            # Cài đặt thuật toán Hybrid GWO-PSO (Benchmark)
│   │
│   ├── JCAS/
│   │   ├── jcas.py               # Bài toán JCAS gốc + Thuật toán TS-ILS
│   │   └── JCAS_original.png     # Kết quả chạy của thuật toán gốc
│   │
│   ├── GWO+JCAS/
│   │   ├── gwo_in_jcas.py        # Áp dụng GWO giải bài toán JCAS
│   │   └── GWO_in_JCAS_result.png
│   │
│   └── GWO_PSO+JCAS/
│       ├── gwo_pso_in_jcas.py    # Áp dụng Hybrid GWO-PSO giải bài toán JCAS (Đề xuất)
│       └── GWO_PSO_in_JCAS_result.png
│
└── README.md                     # Tài liệu hướng dẫn

```
## ⚙️ Cài đặt (Installation)

```text

Dự án yêu cầu Python 3.x và các thư viện tính toán khoa học cơ bản.

1.clone dự án
    git clone https://github.com/SyxVinh1509/KTTT_Group101_GWO_JCAS
    cd PRJ_GWO

2.Cài đặt thư viện:
    pip install numpy matplotlib

🚀 Hướng dẫn chạy (Usage)
Bạn có thể chạy từng file độc lập để xem kết quả của từng thuật toán.

1. Chạy thuật toán gốc (Baseline TS-ILS)
Mô phỏng phương pháp truyền thống giải bài toán JCAS.
    python SRC/JCAS/jcas.py

Kết quả: Sẽ lưu file ảnh vào thư mục SRC/JCAS/.

2. Chạy thuật toán GWO áp dụng vào JCAS
    python SRC/GWO+JCAS/gwo_in_jcas.py

3. Chạy thuật toán Hybrid GWO-PSO (Đề xuất)
Đây là phần trọng tâm của đồ án, thể hiện sự cải tiến về hiệu năng.
    python SRC/GWO_PSO+JCAS/gwo_pso_in_jcas.py

```

## 📊 Kết quả (Results)
```text
Dưới đây là tóm tắt so sánh hiệu năng giữa 3 thuật toán trên hệ thống 64 anten (N=64) với hướng mục tiêu tại 0 và -40 độ:

| Thuật toán | Tốc độ hội tụ | Khả năng tìm kiếm toàn cục | Chất lượng nghiệm (Error) |
| :--- | :--- | :--- | :--- |
| **TS-ILS (Gốc)** | Rất nhanh (ngay lập tức) | Thấp (Dễ kẹt cục bộ) | Trung bình |
| **GWO** | Chậm | Cao | Tốt |
| **Hybrid GWO-PSO** | **Trung bình - Nhanh** | **Rất Cao** | **Tốt nhất** |

Hybrid GWO-PSO khắc phục được nhược điểm hội tụ sớm của TS-ILS và tốc độ chậm của GWO nhờ cơ chế cập nhật vận tốc có hướng.
```

## 👥 Tác giả (Authors)
```text
- Lê Minh Trọng (20239675)
- Nguyễn Sỹ Vinh (20235875)
```