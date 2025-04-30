# Hệ Thống Đề Xuất Nhạc (Music Recommendation System)

Hệ thống đề xuất nhạc sử dụng dữ liệu từ Spotify và thuật toán máy học để gợi ý bài hát phù hợp với sở thích của người dùng.

## Tính năng

- **Đăng nhập với Spotify**: Kết nối với tài khoản Spotify của bạn
- **Phát nhạc**: Phát bài hát trực tiếp từ ứng dụng web
- **Đề xuất dựa trên thể loại**: Chọn thể loại bạn yêu thích để nhận gợi ý
- **Đề xuất dựa trên bài hát**: Tìm bài hát tương tự với bài hát bạn đang nghe
- **Đề xuất dựa trên đặc tính âm nhạc**: Điều chỉnh các thông số như danceability, energy, acousticness để tìm bài hát phù hợp
- **Khám phá ngẫu nhiên**: Tìm nhạc ngẫu nhiên trong kho dữ liệu

## Công nghệ sử dụng

- **Backend**: Flask (Python)
- **Phát nhạc**: Spotify Web Playback SDK
- **Thuật toán đề xuất**: 
  - Content-based filtering (dựa trên đặc tính âm nhạc)
  - Collaborative filtering (dựa trên lịch sử người dùng)
  - Hybrid recommendation (kết hợp các phương pháp)

## Dữ liệu

Hệ thống sử dụng các dữ liệu sau:
- `spotify_artist_data_2023.csv`: Thông tin về nghệ sĩ và thể loại
- `spotify_data_12_20_2023.csv`: Dữ liệu tổng hợp
- `spotify_features_data_2023.csv`: Đặc tính âm thanh của bài hát
- `spotify_tracks_data_2023.csv`: Thông tin cơ bản về bài hát
- `spotify-albums_data_2023.csv`: Thông tin về album
- Tải với link sau: 
Lưu ý: Tải các file vào thư mục "data"

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```

2. Tạo file `.env` với nội dung:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

3. Chạy ứng dụng:
```
python app.py
```

4. Truy cập ứng dụng tại `http://127.0.0.1:5000`

## Phát triển trong tương lai

- Cải thiện độ chính xác của các đề xuất
- Thêm tính năng tạo playlist từ các đề xuất
- Tăng tốc độ xử lý dữ liệu lớn
- Thêm khả năng đề xuất dựa trên tâm trạng và thời điểm trong ngày
