# FPT Stock Price Prediction - CONQ007

Project sử dụng một số thuật toán để dự đoán giá cổ phiếu và hiển thị kết quả lên Web Dashboard.

## Cấu trúc dự án
- `data/`: Chứa dữ liệu lịch sử giá.
- `output/`: Chứa kết quả mô hình dự đoán
- `requirements.txt`: tổng hợp các thư viện cần thiết cho Project
- `dash_UI.py`: Giao diện Web App.
- `model_train.py`: Code huấn luyện mô hình, dành cho những ai muốn chạy local trên máy.
- `stock_price_prediction.ipynb`: Notebook bao gồm cả phần code huấn luyện mô hình và giao diện dành cho những ai muốn huấn luyện mô hình ngay trên Kaggle 

## Cách chạy dự án cho local
1. Download các file và để chúng vào cùng 1 thư mục, gồm `data/`, `output/`, `requirements.txt`, `dash_UI.py` và `model_train.py`
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Thay đổi các đường dẫn INPUT và OUTPUT trong file dash_UI.py và model_train.py
4. Chạy giao diện: `python dash_UI.py`. Sau khi chạy, click vào link ở trong terminal hoặc truy cập http://127.0.0.1:8050/
5.  Click vào "train model" chờ 1 ít phút. Sau khi model đã train xong, kết quả sẽ được hiển thị

## Cách chạy dự án trên Kaggle

1. Import file stock_price_prediction.ipyb lên Kaggle.
2. Truy cập [dashboard.ngrok.com](https://dashboard.ngrok.com/signup) và đăng ký tài khoản.
3. Sau khi đăng nhập, nhìn menu bên trái chọn **Your Authtoken**.
4. Copy đoạn mã token bắt đầu bằng `2...`. Giữ bí mật mã này.
5. Ở cell cuối cùng, chỉnh sửa biến tên là `NGROK_AUTH_TOKEN` chính là token của người dùng. 
6. Run all để Kaggle chạy qua tất cả các cell cho ra kết quả.
7. Truy cập link kết quả trả về trong cell cuối cùng để hiển thị kết quả.
