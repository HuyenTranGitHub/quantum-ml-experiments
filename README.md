# Quantum Machine Learning Experiments

Các thí nghiệm tự học về Quantum Machine Learning, thực hiện trong quá trình học thạc sĩ ngành AI Convergence tại Pukyong National University (Hàn Quốc).

**Công cụ:** Python, PennyLane, scikit-learn, NumPy, SciPy, Matplotlib

Điểm chung của hai thí nghiệm: **cấu trúc mạch quyết định mạch có thể học được gì.** Một cổng đặt sai trục hay một cặp cổng tự triệt tiêu có thể khiến mô hình "lượng tử" thực chất chỉ là một công thức cổ điển, hoặc không học được gì cả.

---

## 1. Variational Quantum Circuit phân loại Iris
📓 [01_vqc_iris_classification.ipynb](01_vqc_iris_classification.ipynb)

- Mạch 4 qubit, 3 lớp trọng số `RY` xen kẽ `CNOT`, huấn luyện bằng Adam với trọng số khởi tạo nhỏ để tránh barren plateau.
- Chia 70/30 train/test, chuẩn hóa theo thống kê của tập train.

**Phát hiện:** chỉ đổi cổng mã hóa dữ liệu từ `RX` sang `RY`, kết quả thay đổi hoàn toàn.

| Cặp hoa | Mã hóa | Test accuracy |
|---|---|---|
| Setosa / Versicolor | `RX` | ~43% (gần như đoán bừa) |
| Setosa / Versicolor | `RY` | **100%** |
| Versicolor / Virginica | `RY` | **~90%** |

**Lý do:** phép đo `PauliZ` chỉ thấy `cos(x)`, mà `cos(−x) = cos(x)`. Với mã hóa `RX` và trọng số `RY` (khác trục), mạch không thể phân biệt dữ liệu âm với dữ liệu dương, trong khi Setosa và Versicolor sau khi chuẩn hóa nằm đối xứng qua gốc tọa độ. Mã hóa `RY` cho dữ liệu và trọng số quay cùng trục, cộng lại thành `x + w`, nên trọng số có thể phá vỡ sự đối xứng.

## 2. Quantum Kernel vs RBF Kernel
📓 [02_quantum_kernel_vs_rbf.ipynb](02_quantum_kernel_vs_rbf.ipynb)

Tái hiện ý tưởng *geometric difference* từ Huang et al. (2021), [*Power of data in quantum machine learning*](https://arxiv.org/abs/2011.01938).

- Tính quantum kernel (PennyLane) và RBF kernel (scikit-learn), rồi tính geometric difference g.
- **Phát hiện:** trong feature map ban đầu (`RX, RX, CNOT`), cổng `CNOT` bị triệt tiêu với `CNOT†` trong mạch kernel. Kernel thu được trùng khít với công thức cổ điển `cos²(Δx₀/2)·cos²(Δx₁/2)` (kiểm chứng bằng số trong notebook).
- So sánh với feature map sửa đổi (thêm `RZ(x₀·x₁)` sau `CNOT`):

| N | g/√N (feature map gốc) | g/√N (feature map sửa) |
|---|---|---|
| 5 | 39.9% | 40.5% |
| 20 | 24.3% | 24.5% |
| 50 | 16.2% | 16.8% |

- **SVM:** Setosa/Versicolor đều đạt 100% với mọi kernel. Versicolor/Virginica chỉ quanh 67–70% với cả RBF lẫn quantum kernel.
- **Nhận xét:** g/√N giảm khi N tăng và chỉ đổi rất ít khi sửa feature map. Với 2 qubit và feature map đơn giản, quantum kernel nhìn dữ liệu gần giống RBF, nên chưa thấy quantum advantage.

---

## Cài đặt và chạy

```bash
pip install -r requirements.txt
jupyter notebook
```

Tất cả thí nghiệm chạy trên simulator (`default.qubit`), chưa chạy trên phần cứng lượng tử thật.
