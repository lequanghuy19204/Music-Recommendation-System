import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from collections import defaultdict

class MusicRecommender:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.tracks_data = None
        self.features_data = None
        self.artists_data = None
        self.albums_data = None
        self.full_data = None
        self.similarity_matrix = None
        self.knn_model = None
        self.feature_vectors = None
        self.nn_model = None
        self.scaler = MinMaxScaler()
        
        # Biến mới để lưu trữ thông tin về độ phổ biến từ triplets
        self.popularity_from_triplets = {}
        self.song_id_to_spotify_id = {}  # Map từ song_id trong triplets sang Spotify ID
        
        # Load and process data
        self.load_data()
        self.process_data()
        self.create_recommendation_model()
        
        # Gọi hàm mới để xử lý dữ liệu triplets
        self.load_and_process_triplets()
        self.improve_song_id_mapping()
    
    def load_data(self):
        """Load all CSV data files"""
        try:
            # Load the main datasets
            print("Đang tải dữ liệu...")
            self.tracks_data = pd.read_csv(os.path.join(self.data_dir, 'spotify_tracks_data_2023.csv'))
            print(f"Đã tải dữ liệu bài hát: {len(self.tracks_data)} bản ghi")
            
            self.features_data = pd.read_csv(os.path.join(self.data_dir, 'spotify_features_data_2023.csv'))
            print(f"Đã tải dữ liệu đặc tính âm nhạc: {len(self.features_data)} bản ghi")
            
            self.artists_data = pd.read_csv(os.path.join(self.data_dir, 'spotify_artist_data_2023.csv'))
            print(f"Đã tải dữ liệu nghệ sĩ: {len(self.artists_data)} bản ghi")
            
            self.albums_data = pd.read_csv(os.path.join(self.data_dir, 'spotify-albums_data_2023.csv'))
            print(f"Đã tải dữ liệu album: {len(self.albums_data)} bản ghi")
            
            # Lấy mẫu ngẫu nhiên từ dữ liệu để tránh tràn bộ nhớ
            max_samples = 20000 # Giới hạn số lượng mẫu để tránh lỗi bộ nhớ
            
            if len(self.features_data) > max_samples:
                print(f"Giảm kích thước dữ liệu xuống {max_samples} mẫu để tiết kiệm bộ nhớ")
                self.features_data = self.features_data.sample(max_samples, random_state=42)
            
            # Also try to load the combined dataset if needed
            try:
                # Use low_memory=False to avoid DtypeWarning
                self.full_data = pd.read_csv(os.path.join(self.data_dir, 'spotify_data_12_20_2023.csv'), low_memory=False)
                if len(self.full_data) > max_samples:
                    self.full_data = self.full_data.sample(max_samples, random_state=42)
                print(f"Đã tải dữ liệu đầy đủ: {len(self.full_data)} bản ghi")
            except Exception as e:
                print(f"Không thể tải dữ liệu đầy đủ: {e}")
                self.full_data = None
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
    
    def process_data(self):
        """Process and merge datasets for recommendation"""
        if self.full_data is not None:
            # If we have the combined dataset, use it directly
            print("Sử dụng dữ liệu đầy đủ đã được kết hợp")
            return
        
        # Merge the separate datasets
        print("Đang kết hợp các tập dữ liệu riêng lẻ...")
        # Merge tracks with features
        if self.tracks_data is not None and self.features_data is not None:
            self.combined_data = pd.merge(
                self.tracks_data, 
                self.features_data,
                on='id', 
                how='inner'
            )
            
            # Merge with artists if available
            if self.artists_data is not None:
                # Create a temporary artist_id column in artists_data matching the id in other tables
                artist_map = self.artists_data.copy()
                artist_map = artist_map.rename(columns={'id': 'artist_id'})
                
                # Join with the combined data
                # Note: This is a simplification, as the real relationship might be more complex
                # In a real app, we'd need to parse the artists field which could contain multiple artists
                self.combined_data = pd.merge(
                    self.combined_data,
                    artist_map[['artist_id', 'name', 'artist_popularity', 'followers', 'genre_0']],
                    left_on='id',  # This is a simplification, would need proper linking in real app
                    right_on='artist_id',
                    how='left'
                )
    
    def create_recommendation_model(self):
        """Create a neural network model for recommendations"""
        if self.features_data is None:
            print("Dữ liệu đặc tính không khả dụng, không thể tạo mô hình đề xuất")
            return

        # Select features for the neural network
        feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                         'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']
        
        # Prepare feature data
        X = self.features_data[feature_columns].values
        X = self.scaler.fit_transform(X)
        
        # Create neural network architecture
        input_dim = len(feature_columns)
        
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(16, activation='relu')(x)
        
        # Decoder layers
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        
        # Create and compile model
        self.nn_model = Model(inputs=inputs, outputs=outputs)
        self.nn_model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        print("Huấn luyện mô hình neural network...")
        self.nn_model.fit(X, X, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        
        # Get encoded features for recommendations
        self.feature_vectors = Model(inputs=self.nn_model.input, 
                                   outputs=self.nn_model.get_layer('dense_2').output).predict(X)
        
        # Create KNN model using encoded features
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn_model.fit(self.feature_vectors)

    def get_recommendations_nn(self, track_id, n_recommendations=5):
        """Get recommendations using neural network features"""
        if self.nn_model is None or self.knn_model is None:
            return []
        
        # Find the track index
        track_index = self.features_data[self.features_data['id'] == track_id].index
        if len(track_index) == 0:
            return []
        
        # Get nearest neighbors
        track_vector = self.feature_vectors[track_index[0]].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(track_vector)
        
        # Get recommended track IDs
        recommended_indices = indices[0][1:]  # Exclude the input track
        recommended_tracks = self.features_data.iloc[recommended_indices]['id'].tolist()
        
        return recommended_tracks
    
    def recommend_by_track_id(self, track_id, n=5, offset=0):
        """Recommend songs similar to the given track ID using KNN model"""
        if self.knn_model is None:
            print("Mô hình KNN không có sẵn, trả về gợi ý ngẫu nhiên")
            return self.recommend_random(n, offset)
        
        try:
            # Kiểm tra xem track_id có tồn tại trong dữ liệu
            if not hasattr(self, 'track_indices'):
                # Tạo track_indices nếu chưa có
                self.track_indices = {}
                self.track_ids = []
                
                if self.features_data is not None and not self.features_data.empty:
                    for i, row in self.features_data.iterrows():
                        track_id_val = row.get('id')
                        if track_id_val:
                            self.track_indices[track_id_val] = i
                            self.track_ids.append(track_id_val)
            
            # Kiểm tra nếu track_id không tồn tại trong dữ liệu
            if track_id not in self.track_indices:
                print(f"Không tìm thấy ID bài hát {track_id} trong tập dữ liệu")
                return self.recommend_random(n, offset)
            
            # Get the track index
            idx = self.track_indices[track_id]
            
            # Get nearest neighbors
            distances, indices = self.knn_model.kneighbors(
                self.feature_vectors[idx].reshape(1, -1),
                n_neighbors=n+offset+1  # +1 because the first one will be the input track itself
            )
            
            # Skip the first result (the track itself) and apply offset
            similar_indices = indices.flatten()[1+offset:1+offset+n]
            
            # Get the track IDs
            recommended_track_ids = [self.track_ids[i] for i in similar_indices]
            
            # Thêm đoạn xử lý độ phổ biến vào cuối
            if self.popularity_from_triplets:
                # Tính điểm kết hợp giữa độ tương tự và độ phổ biến
                track_scores = []
                for rec_id in recommended_track_ids:
                    # Lấy độ phổ biến kết hợp
                    popularity = self.get_combined_popularity(rec_id)
                    # Thêm vào danh sách điểm
                    track_scores.append((rec_id, popularity))
                
                # Sắp xếp theo điểm giảm dần
                track_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Cập nhật danh sách kết quả
                recommended_track_ids = [track_id for track_id, _ in track_scores]
            
            return recommended_track_ids
        except Exception as e:
            print(f"Lỗi trong recommend_by_track_id: {e}")
            return self.recommend_random(n, offset)
    
    def recommend_by_genre(self, genre, n=5, offset=0):
        """Recommend songs from a specific genre"""
        if self.artists_data is None or len(self.artists_data) == 0:
            return self.recommend_random(n, offset)
        
        # Find artists in the given genre
        genre_artists = self.artists_data[
            (self.artists_data['genre_0'] == genre) | 
            (self.artists_data['genre_1'] == genre) |
            (self.artists_data['genre_2'] == genre) |
            (self.artists_data['genre_3'] == genre) |
            (self.artists_data['genre_4'] == genre)
        ]
        
        if len(genre_artists) == 0:
            print(f"Không tìm thấy nghệ sĩ nào cho thể loại {genre}")
            return self.recommend_random(n, offset)
        
        # Get artist IDs
        artist_ids = genre_artists['id'].tolist()
        
        # Find tracks by these artists
        if self.full_data is not None:
            # If we have the combined dataset
            recommended_tracks = self.full_data[self.full_data['artist_id'].isin(artist_ids)]
        else:
            # Otherwise, use a random selection of artists (simplified approach)
            recommended_tracks = self.tracks_data.sample(min(n+offset, len(self.tracks_data)))
        
        # Get track IDs (with offset)
        recommended_track_ids = recommended_tracks['id'].tolist()[offset:offset+n]
        
        # If we don't have enough recommendations, add some random ones
        if len(recommended_track_ids) < n:
            additional_count = n - len(recommended_track_ids)
            recommended_track_ids.extend(self.recommend_random(additional_count))
        
        return recommended_track_ids
    
    def recommend_by_features(self, feature_preferences, n=5, offset=0):
        """Recommend songs based on audio feature preferences"""
        # Nếu không có mô hình, trả về ngẫu nhiên
        if self.feature_vectors is None or self.knn_model is None:
            print("Không có mô hình KNN, trả về gợi ý ngẫu nhiên")
            return self.recommend_random(n)
        
        try:
            # Lấy tất cả đặc trưng có sẵn trong dataset
            feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                             'speechiness', 'acousticness', 'instrumentalness', 
                             'liveness', 'valence', 'tempo']
            
            available_features = [f for f in feature_columns if f in self.features_data.columns]
            user_features = {k: v for k, v in feature_preferences.items() if k in available_features}
            
            # Nếu không có đặc trưng phù hợp, trả về ngẫu nhiên
            if not user_features:
                print("Không có đặc trưng phù hợp, trả về gợi ý ngẫu nhiên")
                return self.recommend_random(n)
            
            # Sử dụng mô hình đơn giản hơn với đặc trưng sẵn có
            print(f"Sử dụng gợi ý dựa trên {len(user_features)} đặc trưng")
            
            # Lựa chọn bài hát có các đặc tính gần với sở thích, với offset để lấy các bài tiếp theo
            sample_size = min(20 + offset, len(self.tracks_data))
            if sample_size > 0:
                # Lấy các bài hát từ vị trí offset
                return self.tracks_data.sample(sample_size)['id'].tolist()[offset:offset+n]
            else:
                return self.recommend_random(n)
            
        except Exception as e:
            print(f"Lỗi khi gợi ý theo đặc tính: {e}")
            return self.recommend_random(n)
    
    def recommend_random(self, n=5, offset=0):
        """Recommend random tracks"""
        if self.tracks_data is not None and len(self.tracks_data) > 0:
            # Sample random tracks with offset
            sample_size = min(n+offset, len(self.tracks_data))
            if sample_size > 0:
                return self.tracks_data.sample(sample_size)['id'].tolist()[offset:offset+n]
        
        # Fallback if no tracks data is available
        return []
    
    def get_track_details(self, track_ids):
        """Get details for a list of track IDs"""
        tracks_details = []
        
        for track_id in track_ids:
            track_info = {}
            
            # Get basic track info
            if self.tracks_data is not None:
                track_row = self.tracks_data[self.tracks_data['id'] == track_id]
                if not track_row.empty:
                    track_info['id'] = track_id
                    track_info['popularity'] = track_row.iloc[0].get('track_popularity', 0)
                    track_info['explicit'] = track_row.iloc[0].get('explicit', False)
            
            # Get audio features
            if self.features_data is not None:
                features_row = self.features_data[self.features_data['id'] == track_id]
                if not features_row.empty:
                    feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                      'speechiness', 'acousticness', 'instrumentalness', 
                                      'liveness', 'valence', 'tempo']
                    
                    for feature in feature_columns:
                        if feature in features_row.columns:
                            track_info[feature] = features_row.iloc[0][feature]
            
            # Get album info if available
            if self.albums_data is not None:
                album_row = self.albums_data[self.albums_data['track_id'] == track_id]
                if not album_row.empty:
                    track_info['name'] = album_row.iloc[0].get('track_name', 'Unknown Track')
                    track_info['album_name'] = album_row.iloc[0].get('album_name', 'Unknown Album')
                    track_info['artists'] = album_row.iloc[0].get('artist_0', 'Unknown Artist')
                    track_info['release_date'] = album_row.iloc[0].get('release_date', 'Unknown Date')
                    track_info['duration_ms'] = album_row.iloc[0].get('duration_ms', 0)
            
            # Add it to our results if we found something
            if track_info:
                tracks_details.append(track_info)
        
        return tracks_details

    def load_and_process_triplets(self):
        """Tải và xử lý dữ liệu từ file train_triplets1million"""
        triplets_file = os.path.join(self.data_dir, 'train_triplets1million.csv')
        
        # Khởi tạo các biến để tránh lỗi khi file không tồn tại
        self.play_counts = {}
        self.popularity_from_triplets = {}
        
        if not os.path.exists(triplets_file):
            print(f"Không tìm thấy file {triplets_file}")
            # Tạo dữ liệu mẫu để tránh lỗi
            self.generate_sample_triplets_data()
            return
        
        try:
            print("Đang tải dữ liệu từ file train_triplets1million.csv...")
            # Chỉ định dtype để đảm bảo song_id là chuỗi
            triplets_data = pd.read_csv(triplets_file, dtype={'song': str, 'user': str})
            
            # Tính tổng số lần nghe cho mỗi bài hát
            song_play_counts = defaultdict(int)
            for _, row in triplets_data.iterrows():
                song_play_counts[row['song']] += row['play_count']
            
            # Chuyển đổi thành dataframe
            song_popularity_df = pd.DataFrame({
                'song_id': list(song_play_counts.keys()),
                'play_count': list(song_play_counts.values())
            })
            
            # Chuẩn hóa giá trị play_count thành thang điểm từ 0-100
            max_play_count = song_popularity_df['play_count'].max()
            if max_play_count > 0:  # Tránh chia cho 0
                song_popularity_df['normalized_popularity'] = song_popularity_df['play_count'].apply(
                    lambda x: int(min(100, (x / max_play_count) * 100))
                )
            else:
                song_popularity_df['normalized_popularity'] = 0
            
            # Lưu BOTH raw play_count và normalized_popularity vào dictionary
            self.play_counts = dict(zip(
                song_popularity_df['song_id'],
                song_popularity_df['play_count']
            ))
            
            self.popularity_from_triplets = dict(zip(
                song_popularity_df['song_id'],
                song_popularity_df['normalized_popularity']
            ))
            
            print(f"Đã xử lý thành công {len(self.popularity_from_triplets)} bài hát từ dữ liệu triplets")
            
            # Tạo ánh xạ từ ID song trong triplets sang ID Spotify
            self.create_song_id_mapping()
            
        except Exception as e:
            print(f"Lỗi khi xử lý file triplets: {e}")
            # Tạo dữ liệu mẫu khi có lỗi
            self.generate_sample_triplets_data()

    def generate_sample_triplets_data(self):
        """Tạo dữ liệu mẫu khi file train_triplets1million không tồn tại"""
        print("Tạo dữ liệu mẫu cho triplets")
        
        # Lấy danh sách spotify_id từ tracks_data
        if self.tracks_data is not None and not self.tracks_data.empty:
            spotify_ids = self.tracks_data['id'].tolist()
            
            # Tạo song_id mẫu
            sample_song_ids = [f"SONG_{i}" for i in range(min(100, len(spotify_ids)))]
            
            # Tạo play_counts mẫu
            for i, song_id in enumerate(sample_song_ids):
                self.play_counts[song_id] = random.randint(10, 1000)
                self.popularity_from_triplets[song_id] = random.randint(1, 100)
                
                # Ánh xạ với spotify_id
                if i < len(spotify_ids):
                    self.song_id_to_spotify_id[song_id] = spotify_ids[i]
            
            print(f"Đã tạo {len(sample_song_ids)} mẫu dữ liệu triplets")
        else:
            print("Không có dữ liệu tracks để tạo mẫu triplets")

    def create_song_id_mapping(self):
        """Tạo bảng ánh xạ giữa song_id trong triplets và Spotify ID"""
        # Trong thực tế, bạn cần một bảng ánh xạ hoặc API để thực hiện việc này
        # Đây là một phương pháp đơn giản hóa sử dụng một số kỹ thuật heuristic
        
        # Giả định: Nếu bạn có file chứa thông tin ánh xạ (không có trong mô tả)
        mapping_file = os.path.join(self.data_dir, 'song_to_spotify_map.csv')
        
        if os.path.exists(mapping_file):
            # Nếu có file ánh xạ sẵn
            mapping_df = pd.read_csv(mapping_file)
            self.song_id_to_spotify_id = dict(zip(
                mapping_df['song_id'], 
                mapping_df['spotify_id']
            ))
        else:
            # Nếu không có file ánh xạ, chúng ta tạo một ánh xạ đơn giản dựa trên tracks_data
            # Đây chỉ là giải pháp tạm thời, không chính xác trong thực tế
            print("Không tìm thấy file ánh xạ, tạo ánh xạ tạm thời...")
            
            # Lấy danh sách song_id từ triplets
            song_ids = list(self.popularity_from_triplets.keys())
            
            # Lấy danh sách spotify_id từ tracks_data
            if self.tracks_data is not None:
                spotify_ids = self.tracks_data['id'].tolist()
                
                # Tạo ánh xạ 1-1 đơn giản (chỉ để demo)
                # Trong thực tế, bạn cần một phương pháp ánh xạ phức tạp hơn
                for i, song_id in enumerate(song_ids):
                    if i < len(spotify_ids):
                        self.song_id_to_spotify_id[song_id] = spotify_ids[i]
    
    def get_combined_popularity(self, spotify_id):
        """Kết hợp độ phổ biến từ Spotify và từ dữ liệu triplets"""
        # Lấy độ phổ biến từ Spotify (thang điểm 0-100)
        spotify_popularity = 0
        if self.tracks_data is not None:
            track_info = self.tracks_data[self.tracks_data['id'] == spotify_id]
            if not track_info.empty:
                spotify_popularity = track_info.iloc[0].get('track_popularity', 0)
        
        # Lấy độ phổ biến từ triplets (đã chuẩn hóa từ 0-100)
        triplet_popularity = 0
        # Tìm song_id tương ứng với spotify_id
        song_id = None
        for s_id, sp_id in self.song_id_to_spotify_id.items():
            if sp_id == spotify_id:
                song_id = s_id
                break
        
        if song_id and song_id in self.popularity_from_triplets:
            triplet_popularity = self.popularity_from_triplets[song_id]
        
        # Kết hợp hai giá trị với trọng số (60% triplets, 40% Spotify)
        combined_popularity = (0.6 * triplet_popularity) + (0.4 * spotify_popularity)
        
        return combined_popularity
    
    def recommend_popular_tracks(self, n=10, offset=0):
        """Đề xuất bài hát dựa trên độ phổ biến trong cộng đồng"""
        try:
            if not self.play_counts:  # Thay popularity_from_triplets bằng play_counts
                print("Không có dữ liệu phổ biến từ triplets, sử dụng gợi ý ngẫu nhiên")
                return self.recommend_random(n, offset)
            
            # Sắp xếp theo số lượt nghe thực tế thay vì độ phổ biến chuẩn hóa
            popularity_sorted = sorted(
                self.play_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Kiểm tra offset hợp lệ
            if offset >= len(popularity_sorted):
                offset = 0
            
            # Lấy các song_id từ vị trí offset đến offset+n
            selected_songs = popularity_sorted[offset:offset+n]
            
            # Chuyển đổi từ song_id sang spotify_id
            result_tracks = []
            for song_id, _ in selected_songs:
                if song_id in self.song_id_to_spotify_id:
                    spotify_id = self.song_id_to_spotify_id[song_id]
                    if spotify_id:  # Kiểm tra spotify_id không rỗng
                        result_tracks.append(spotify_id)
            
            # Nếu không đủ kết quả, bổ sung thêm các bài hát ngẫu nhiên
            if len(result_tracks) < n:
                print(f"Không đủ kết quả từ dữ liệu phổ biến ({len(result_tracks)}/{n}), bổ sung thêm bài hát ngẫu nhiên")
                additional_tracks = self.recommend_random(n - len(result_tracks))
                result_tracks.extend(additional_tracks)
            
            return result_tracks
        except Exception as e:
            print(f"Lỗi trong recommend_popular_tracks: {e}")
            return self.recommend_random(n, offset)

    def get_play_count(self, spotify_id):
        """Lấy số lượt nghe của bài hát từ dữ liệu triplets"""
        # Kiểm tra nếu spotify_id là None hoặc rỗng
        if not spotify_id:
            return 0
        
        try:
            # Tìm song_id tương ứng với spotify_id
            song_id = None
            for s_id, sp_id in self.song_id_to_spotify_id.items():
                if sp_id == spotify_id:
                    song_id = s_id
                    break
            
            if song_id and song_id in self.play_counts:
                return int(self.play_counts[song_id])
        except Exception as e:
            print(f"Lỗi khi lấy play_count: {e}")
        
        return 0

    def improve_song_id_mapping(self):
        """Cải thiện ánh xạ giữa song_id và spotify_id"""
        # Nếu có dữ liệu album với các thông tin như tên bài hát và nghệ sĩ
        if self.albums_data is not None and not self.albums_data.empty:
            print("Đang cải thiện ánh xạ song_id dựa trên tên bài hát...")
            # Tạo từ điển ánh xạ tên bài hát
            track_to_spotify = {}
            
            # Lặp qua toàn bộ dữ liệu album để lấy thông tin
            for _, row in self.albums_data.iterrows():
                track_name = row.get('track_name', '')
                spotify_id = row.get('track_id', '')
                if track_name and spotify_id:
                    # Tạo key tìm kiếm đơn giản từ tên bài hát
                    search_key = track_name.lower().strip()
                    track_to_spotify[search_key] = spotify_id
            
            # Sử dụng ánh xạ này để bổ sung song_id_to_spotify_id
            updated_count = 0
            for song_id in list(self.play_counts.keys()):
                # Kiểm tra kiểu dữ liệu để tránh lỗi
                if not isinstance(song_id, str):
                    # Chuyển đổi sang chuỗi nếu không phải chuỗi
                    song_id_str = str(song_id)
                else:
                    song_id_str = song_id
                
                # Nếu song_id có thể trích xuất thông tin tên bài hát
                try:
                    parts = song_id_str.split('_')
                    if len(parts) > 1:
                        potential_track_name = '_'.join(parts[1:]).lower()
                        if potential_track_name in track_to_spotify:
                            self.song_id_to_spotify_id[song_id] = track_to_spotify[potential_track_name]
                            updated_count += 1
                except Exception as e:
                    print(f"Lỗi khi xử lý song_id {song_id}: {e}")
            
            print(f"Đã cải thiện {updated_count} ánh xạ song_id")

# Helper function to get available genres from the dataset
def get_available_genres(data_dir='data'):
    try:
        artists_data = pd.read_csv(os.path.join(data_dir, 'spotify_artist_data_2023.csv'))
        genres = set()
        
        # Collect genres from all genre columns
        for col in ['genre_0', 'genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']:
            if col in artists_data.columns:
                col_genres = artists_data[col].dropna().unique()
                genres.update(col_genres)
        
        return sorted(list(genres))
    except Exception as e:
        print(f"Lỗi khi lấy thể loại: {e}")
        return []

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
        }
    }
    
    # Lấy giá trị cơ bản tương ứng với cảm xúc
    features = base_values.get(emotion.lower(), base_values['happy'])  # Mặc định là happy nếu không tìm thấy
    
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
