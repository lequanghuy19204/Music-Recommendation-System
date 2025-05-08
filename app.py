from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import uuid
import pandas as pd
import base64
import cv2
import numpy as np
from io import BytesIO
from deepface import DeepFace
from recommendation import MusicRecommender, get_available_genres

load_dotenv()

app = Flask(__name__)
app.secret_key = 'H550'

# Cấu hình Spotify
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'user-library-read user-library-modify streaming user-read-playback-state user-modify-playback-state user-read-currently-playing'

# Cache cho từng người dùng
caches_folder = './.spotify_caches/'
if not os.path.exists(caches_folder):
    os.makedirs(caches_folder)

# Tạo thư mục tạm thời để lưu ảnh phân tích
temp_folder = './temp_images/'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# Khởi tạo hệ thống gợi ý
recommender = MusicRecommender()
genres = get_available_genres()

def session_cache_path():
    return caches_folder + session.get('uuid')

@app.route('/')
def login():
    if not session.get('uuid'):
        # Tạo ID duy nhất cho mỗi session
        session['uuid'] = str(uuid.uuid4())

    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=cache_handler,
        show_dialog=True  # Luôn hiện dialog đăng nhập
    )

    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        auth_url = auth_manager.get_authorize_url()
        return render_template('login.html', auth_url=auth_url)
    
    return redirect(url_for('player'))

@app.route('/callback')
def callback():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=cache_handler
    )
    
    if request.args.get("code"):
        auth_manager.get_access_token(request.args.get("code"))
        return redirect(url_for('player'))

    return redirect(url_for('login'))

@app.route('/player')
def player():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return redirect(url_for('login'))
    
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    results = spotify.current_user_saved_tracks(limit=20)
    tracks = results['items']
    
    # Lấy thông tin người dùng
    user = spotify.current_user()
    
    # Lấy các thể loại nhạc có sẵn cho dropdown
    available_genres = genres[:30]  # Giới hạn số lượng thể loại hiển thị
    
    return render_template('player.html', 
                         tracks=tracks, 
                         access_token=auth_manager.get_access_token()['access_token'],
                         user=user,
                         genres=available_genres)

@app.route('/play', methods=['POST'])
def play():
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.json
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    
    try:
        spotify.start_playback(
            device_id=data['device_id'],
            uris=[data['uri']]
        )
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint to get music recommendations"""
    data = request.json
    
    # Các loại gợi ý khác nhau
    rec_type = data.get('type', 'random')
    num_recs = data.get('num_recommendations', 5)
    offset = data.get('offset', 0)  # Dùng offset cho tất cả các loại gợi ý
    
    # Gợi ý dựa trên bài hát
    if rec_type == 'track':
        track_id = data.get('track_id')
        if not track_id:
            return jsonify({"status": "error", "message": "No track ID provided"}), 400
        
        # Bắt lỗi khi track_id không hợp lệ
        try:
            rec_tracks = recommender.recommend_by_track_id(track_id, num_recs, offset)
        except Exception as e:
            print(f"Lỗi khi đề xuất từ track_id {track_id}: {str(e)}")
            # Nếu lỗi, trả về gợi ý ngẫu nhiên để tránh crash
            rec_tracks = recommender.recommend_random(num_recs, offset)
    
    # Gợi ý dựa trên thể loại
    elif rec_type == 'genre':
        genre = data.get('genre')
        if not genre:
            return jsonify({"status": "error", "message": "No genre provided"}), 400
        
        rec_tracks = recommender.recommend_by_genre(genre, num_recs, offset)
    
    # Gợi ý dựa trên đặc điểm âm nhạc
    elif rec_type == 'features':
        features = data.get('features', {})
        rec_tracks = recommender.recommend_by_features(features, num_recs, offset)
    
    # Gợi ý ngẫu nhiên
    else:  # random
        rec_tracks = recommender.recommend_random(num_recs, offset)
    
    # Lấy chi tiết về các bài hát được gợi ý
    tracks_details = recommender.get_track_details(rec_tracks)
    
    # Tìm thêm thông tin từ Spotify API nếu đang đăng nhập
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    # Thêm thông tin để hiển thị trên giao diện
    result_tracks = []
    if auth_manager.validate_token(cache_handler.get_cached_token()):
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        
        # Nhóm các ID bài hát thành các nhóm nhỏ vì API giới hạn số lượng
        track_ids = [t['id'] for t in tracks_details if 'id' in t]
        
        # Chia thành các nhóm 50 bài hát (giới hạn API của Spotify)
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            
            try:
                spotify_tracks = spotify.tracks(batch)['tracks']
                
                for spotify_track in spotify_tracks:
                    if spotify_track:
                        track_info = {
                            'id': spotify_track['id'],
                            'name': spotify_track['name'],
                            'artist': spotify_track['artists'][0]['name'] if spotify_track['artists'] else 'Unknown',
                            'album': spotify_track['album']['name'],
                            'image': spotify_track['album']['images'][0]['url'] if spotify_track['album']['images'] else '',
                            'uri': spotify_track['uri']
                        }
                        result_tracks.append(track_info)
            except Exception as e:
                print(f"Error fetching Spotify data: {e}")
    
    # Nếu không thể lấy dữ liệu từ Spotify hoặc không ở trạng thái đăng nhập, sử dụng dữ liệu cục bộ
    if not result_tracks:
        for track in tracks_details:
            result_tracks.append({
                'id': track.get('id', ''),
                'name': track.get('name', 'Unknown Track'),
                'artist': track.get('artists', 'Unknown Artist'),
                'album': track.get('album_name', 'Unknown Album'),
                'image': '/static/default-album.jpg',  # Hình mặc định
                'uri': f"spotify:track:{track.get('id', '')}"
            })
    
    # Thêm thông tin về lượt nghe cho tất cả các bài hát
    for track in result_tracks:
        track_id = track.get('id', '')
        play_count = recommender.get_play_count(track_id)
        track['play_count'] = int(play_count) if play_count is not None else 0
    
    return jsonify({
        "status": "success",
        "recommendations": result_tracks
    })

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """API endpoint để phân tích cảm xúc từ hình ảnh sử dụng DeepFace"""
    try:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400
        
        # Chuyển đổi dữ liệu hình ảnh từ base64 sang numpy array
        image_data = image_data.split(',')[1] 
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Lưu ảnh tạm thời để DeepFace phân tích
        session_id = session.get('uuid', 'temp')
        temp_image_path = os.path.join(temp_folder, f'{session_id}.jpg')
        cv2.imwrite(temp_image_path, image)
        
        # Phân tích cảm xúc với DeepFace
        try:
            result = DeepFace.analyze(temp_image_path, actions=['emotion'], enforce_detection=False)
            
            # DeepFace có thể trả về một từ điển hoặc một danh sách từ điển
            if isinstance(result, list):
                if not result:
                    raise ValueError("No face detected")
                result = result[0]
            
            # Lấy thông tin cảm xúc
            emotions = result['emotion']
            
            # Chỉ giữ lại ba cảm xúc: happy, sad, angry
            filtered_emotions = {
                'happy': emotions.get('happy', 0) / 100,
                'sad': emotions.get('sad', 0) / 100,
                'angry': emotions.get('angry', 0) / 100,
            }
            
            # Chuẩn hóa lại tổng bằng 1
            total = sum(filtered_emotions.values())
            if total > 0:
                for key in filtered_emotions:
                    filtered_emotions[key] /= total
            
            # Xác định cảm xúc chính từ các cảm xúc đã lọc
            dominant_emotion = max(filtered_emotions, key=filtered_emotions.get)
            
            return jsonify({
                "status": "success",
                "emotions": filtered_emotions,
                "dominant_emotion": dominant_emotion
            })
            
        except Exception as analyze_error:
            print(f"DeepFace error: {analyze_error}")
            return jsonify({"status": "error", "message": str(analyze_error)}), 400
        
        finally:
            # Xóa file tạm sau khi xử lý
            try:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")
                
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/recommend_emotion', methods=['POST'])
def recommend_emotion():
    """API endpoint để gợi ý bài hát dựa trên cảm xúc"""
    data = request.json
    emotion = data.get('emotion')
    num_recs = data.get('num_recommendations', 5)
    offset = data.get('offset', 0)  # Thêm tham số offset
    
    if not emotion:
        return jsonify({"status": "error", "message": "No emotion provided"}), 400
    
    # Ánh xạ cảm xúc sang đặc tính âm nhạc
    features = map_emotion_to_features(emotion)
    
    # Gọi hàm gợi ý dựa trên đặc tính
    rec_tracks = recommender.recommend_by_features(features, num_recs, offset)
    
    # Lấy chi tiết về các bài hát được gợi ý
    tracks_details = recommender.get_track_details(rec_tracks)
    
    # Lấy thông tin từ Spotify API
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    result_tracks = []
    if auth_manager.validate_token(cache_handler.get_cached_token()):
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        
        track_ids = [t['id'] for t in tracks_details if 'id' in t]
        
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            
            try:
                spotify_tracks = spotify.tracks(batch)['tracks']
                
                for spotify_track in spotify_tracks:
                    if spotify_track:
                        track_info = {
                            'id': spotify_track['id'],
                            'name': spotify_track['name'],
                            'artist': spotify_track['artists'][0]['name'] if spotify_track['artists'] else 'Unknown',
                            'album': spotify_track['album']['name'],
                            'image': spotify_track['album']['images'][0]['url'] if spotify_track['album']['images'] else '',
                            'uri': spotify_track['uri']
                        }
                        result_tracks.append(track_info)
            except Exception as e:
                print(f"Error fetching Spotify data: {e}")
    
    if not result_tracks:
        for track in tracks_details:
            result_tracks.append({
                'id': track.get('id', ''),
                'name': track.get('name', 'Unknown Track'),
                'artist': track.get('artists', 'Unknown Artist'),
                'album': track.get('album_name', 'Unknown Album'),
                'image': '/static/default-album.jpg',
                'uri': f"spotify:track:{track.get('id', '')}"
            })
    
    # Thêm thông tin về lượt nghe cho tất cả các bài hát dựa trên cảm xúc
    for track in result_tracks:
        track_id = track.get('id', '')
        play_count = recommender.get_play_count(track_id)
        track['play_count'] = int(play_count) if play_count is not None else 0
    
    return jsonify({
        "status": "success",
        "recommendations": result_tracks
    })

def map_emotion_to_features(emotion):
    """
    Ánh xạ một cảm xúc sang các đặc tính âm nhạc phù hợp
    Dựa trên nghiên cứu về tâm lý âm nhạc
    """
    # Các giá trị cơ bản dựa trên cảm xúc
    base_values = {
        'happy': {
            'valence': 0.8,     # Cao - tích cực
            'energy': 0.7,      # Trung bình-cao
            'danceability': 0.7, # Cao
            'tempo': 0.7,       # Nhanh
            'acousticness': 0.3  # Thấp
        },
        'sad': {
            'valence': 0.2,     # Thấp - tiêu cực
            'energy': 0.3,      # Thấp
            'danceability': 0.3, # Thấp
            'tempo': 0.3,       # Chậm
            'acousticness': 0.7  # Cao
        },
        'angry': {
            'valence': 0.3,     # Thấp - tiêu cực
            'energy': 0.9,      # Rất cao
            'danceability': 0.5, # Trung bình
            'tempo': 0.8,       # Nhanh
            'acousticness': 0.2  # Rất thấp
        },
        'fear': {
            'valence': 0.2,     # Thấp - tiêu cực
            'energy': 0.5,      # Trung bình
            'danceability': 0.3, # Thấp
            'tempo': 0.4,       # Trung bình-chậm
            'acousticness': 0.6  # Cao
        },
        'surprise': {
            'valence': 0.6,     # Trung bình-cao
            'energy': 0.7,      # Cao
            'danceability': 0.6, # Trung bình-cao
            'tempo': 0.7,       # Nhanh
            'acousticness': 0.4  # Trung bình
        },
        'neutral': {
            'valence': 0.5,     # Trung bình
            'energy': 0.5,      # Trung bình
            'danceability': 0.5, # Trung bình
            'tempo': 0.5,       # Trung bình
            'acousticness': 0.5  # Trung bình
        },
        'disgust': {
            'valence': 0.3,     # Thấp
            'energy': 0.6,      # Trung bình-cao
            'danceability': 0.4, # Trung bình-thấp
            'tempo': 0.5,       # Trung bình
            'acousticness': 0.4  # Trung bình-thấp
        }
    }
    
    # Lấy giá trị cơ bản tương ứng với cảm xúc
    features = base_values.get(emotion.lower(), base_values['neutral'])
    
    # Bổ sung thêm các đặc trưng còn thiếu với giá trị mặc định
    all_features = {
        'valence': features.get('valence', 0.5),
        'energy': features.get('energy', 0.5),
        'danceability': features.get('danceability', 0.5),
        'tempo': features.get('tempo', 0.5),
        'acousticness': features.get('acousticness', 0.5),
        'liveness': 0.5,
        'speechiness': 0.5,
        'instrumentalness': 0.5,
        'key': 0.5,
        'loudness': 0.5,
        'mode': 0.5,
        # Thêm các đặc trưng còn lại để đủ 16 đặc trưng
        'time_signature': 0.5,
        'duration_ms': 0.5,
        'chorus_hit': 0.5,
        'sections': 0.5,
        'popularity': 0.5
    }
    
    return all_features

@app.route('/track_data/<track_id>')
def track_data(track_id):
    """API endpoint để lấy chi tiết về một bài hát"""
    track_details = recommender.get_track_details([track_id])
    
    if not track_details:
        return jsonify({"status": "error", "message": "Track not found"}), 404
    
    return jsonify({
        "status": "success",
        "track": track_details[0]
    })

@app.route('/logout')
def logout():
    # Xóa cache của người dùng
    try:
        os.remove(session_cache_path())
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    
    # Xóa session
    session.clear()
    return redirect(url_for('login'))

@app.route('/popular_tracks', methods=['POST'])
def popular_tracks():
    """API endpoint để lấy bài hát phổ biến từ dữ liệu triplets"""
    data = request.json
    num_recs = data.get('num_recommendations', 10)
    offset = data.get('offset', 0)
    
    # Gọi phương thức mới từ recommender
    rec_tracks = recommender.recommend_popular_tracks(num_recs, offset)
    
    # Lấy chi tiết về các bài hát được gợi ý
    tracks_details = recommender.get_track_details(rec_tracks)
    
    # Tìm thêm thông tin từ Spotify API nếu đang đăng nhập
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    # Thêm thông tin để hiển thị trên giao diện
    result_tracks = []
    if auth_manager.validate_token(cache_handler.get_cached_token()):
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        
        # Nhóm các ID bài hát thành các nhóm nhỏ vì API giới hạn số lượng
        track_ids = [t['id'] for t in tracks_details if 'id' in t]
        
        # Chia thành các nhóm 50 bài hát (giới hạn API của Spotify)
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            
            try:
                spotify_tracks = spotify.tracks(batch)['tracks']
                
                for spotify_track in spotify_tracks:
                    if spotify_track:
                        # Lấy số lượt nghe thực tế
                        play_count = recommender.get_play_count(spotify_track['id'])
                        
                        track_info = {
                            'id': spotify_track['id'],
                            'name': spotify_track['name'],
                            'artist': spotify_track['artists'][0]['name'] if spotify_track['artists'] else 'Unknown',
                            'album': spotify_track['album']['name'],
                            'image': spotify_track['album']['images'][0]['url'] if spotify_track['album']['images'] else '',
                            'uri': spotify_track['uri'],
                            'play_count': int(play_count) if play_count is not None else 0
                        }
                        result_tracks.append(track_info)
            except Exception as e:
                print(f"Error fetching Spotify data: {e}")
    
    # Nếu không thể lấy dữ liệu từ Spotify hoặc không ở trạng thái đăng nhập, sử dụng dữ liệu cục bộ
    if not result_tracks:
        for track in tracks_details:
            spotify_id = track.get('id', '')
            play_count = recommender.get_play_count(spotify_id)
                
            result_tracks.append({
                'id': spotify_id,
                'name': track.get('name', 'Unknown Track'),
                'artist': track.get('artists', 'Unknown Artist'),
                'album': track.get('album_name', 'Unknown Album'),
                'image': '/static/default-album.jpg',  # Hình mặc định
                'uri': f"spotify:track:{spotify_id}",
                'play_count': int(play_count) if play_count is not None else 0
            })
    
    return jsonify({
        "status": "success",
        "recommendations": result_tracks
    })

@app.route('/save_track', methods=['POST'])
def save_track():
    """API endpoint để lưu bài hát vào thư viện Spotify của người dùng"""
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE + " user-library-modify"  # Thêm quyền truy cập để sửa đổi thư viện
    )
    
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return jsonify({"status": "error", "message": "Bạn cần đăng nhập lại để thực hiện chức năng này"}), 401
    
    data = request.json
    track_id = data.get('track_id')
    
    if not track_id:
        return jsonify({"status": "error", "message": "Không có ID bài hát"}), 400
    
    try:
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        
        # Thêm bài hát vào thư viện của người dùng
        spotify.current_user_saved_tracks_add(tracks=[track_id])
        
        return jsonify({
            "status": "success",
            "message": "Đã lưu bài hát vào thư viện của bạn"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi lưu bài hát: {str(e)}"
        }), 500

@app.route('/get_user_library')
def get_user_library():
    """API endpoint để lấy danh sách bài hát đã lưu của người dùng"""
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    try:
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        results = spotify.current_user_saved_tracks(limit=20)
        tracks = results['items']
        
        formatted_tracks = []
        for item in tracks:
            track = item['track']
            formatted_tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'image': track['album']['images'][0]['url'] if track['album']['images'] else '/static/default-album.jpg',
                'uri': track['uri'],
                'added_at': item['added_at']  # Thêm thời gian được thêm vào
            })
        
        return jsonify({
            "status": "success",
            "tracks": formatted_tracks
        })
    except Exception as e:
        print(f"Error getting user library: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/recommend_spotify', methods=['POST'])
def recommend_spotify():
    """API endpoint để gợi ý bài hát dựa trên đặc trưng âm nhạc từ Spotify"""
    cache_handler = spotipy.cache_handler.CacheFileHandler(cache_path=session_cache_path())
    auth_manager = SpotifyOAuth(
        cache_handler=cache_handler,
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    )
    
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    data = request.json
    track_id = data.get('track_id')
    num_recs = data.get('num_recommendations', 5)
    
    if not track_id:
        return jsonify({"status": "error", "message": "No track ID provided"}), 400
    
    try:
        spotify = spotipy.Spotify(auth_manager=auth_manager)
        
        # Lấy đặc trưng âm nhạc của bài hát nguồn
        track_features = spotify.audio_features(track_id)[0]
        if not track_features:
            return jsonify({"status": "error", "message": "Cannot get audio features"}), 404
        
        # Sử dụng API đề xuất của Spotify để lấy bài hát tương tự
        recommendations = spotify.recommendations(
            seed_tracks=[track_id],
            limit=num_recs,
            target_danceability=track_features.get('danceability'),
            target_energy=track_features.get('energy'),
            target_key=track_features.get('key'),
            target_loudness=track_features.get('loudness'),
            target_tempo=track_features.get('tempo'),
            target_valence=track_features.get('valence'),
            target_acousticness=track_features.get('acousticness'),
            target_instrumentalness=track_features.get('instrumentalness'),
            target_liveness=track_features.get('liveness'),
            target_speechiness=track_features.get('speechiness')
        )
        
        # Định dạng kết quả
        result_tracks = []
        for track in recommendations['tracks']:
            # Lấy số lượt nghe từ dataset nếu có
            play_count = recommender.get_play_count(track['id'])
            
            result_tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'album': track['album']['name'],
                'image': track['album']['images'][0]['url'] if track['album']['images'] else '',
                'uri': track['uri'],
                'play_count': int(play_count) if play_count is not None else 0
            })
        
        return jsonify({
            "status": "success",
            "recommendations": result_tracks
        })
        
    except Exception as e:
        print(f"Error getting Spotify recommendations: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)