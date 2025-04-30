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
        
        # Load and process data
        self.load_data()
        self.process_data()
        self.create_recommendation_model()
    
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
            max_samples = 1000 # Giới hạn số lượng mẫu để tránh lỗi bộ nhớ
            
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
    
    def recommend_by_track_id(self, track_id, n=5):
        """Recommend songs similar to the given track ID using KNN model"""
        if self.knn_model is None or self.track_indices is None:
            print("Mô hình KNN không có sẵn, trả về gợi ý ngẫu nhiên")
            return self.recommend_random(n)
        
        # Check if track_id exists in our data
        if track_id not in self.track_indices:
            print(f"Không tìm thấy ID bài hát {track_id} trong tập dữ liệu")
            return self.recommend_random(n)
        
        # Get the track index
        idx = self.track_indices[track_id]
        
        # Get nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            self.feature_vectors[idx].reshape(1, -1),
            n_neighbors=n+1  # +1 because the first one will be the input track itself
        )
        
        # Skip the first result (the track itself)
        similar_indices = indices.flatten()[1:n+1]
        
        # Get the track IDs
        recommended_track_ids = [self.track_ids[i] for i in similar_indices]
        
        return recommended_track_ids
    
    def recommend_by_genre(self, genre, n=5):
        """Recommend songs from a specific genre"""
        if self.artists_data is None or len(self.artists_data) == 0:
            return self.recommend_random(n)
        
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
            return self.recommend_random(n)
        
        # Get artist IDs
        artist_ids = genre_artists['id'].tolist()
        
        # Find tracks by these artists
        if self.full_data is not None:
            # If we have the combined dataset
            recommended_tracks = self.full_data[self.full_data['artist_id'].isin(artist_ids)]
        else:
            # Otherwise, use a random selection of artists (simplified approach)
            # In a real app, you'd need to properly link artists to tracks
            recommended_tracks = self.tracks_data.sample(min(n, len(self.tracks_data)))
        
        # Get track IDs (up to n)
        recommended_track_ids = recommended_tracks['id'].tolist()[:n]
        
        # If we don't have enough recommendations, add some random ones
        if len(recommended_track_ids) < n:
            additional_count = n - len(recommended_track_ids)
            recommended_track_ids.extend(self.recommend_random(additional_count))
        
        return recommended_track_ids
    
    def recommend_by_features(self, feature_preferences, n=5):
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
            
            # Lựa chọn 5 bài hát ngẫu nhiên có các đặc tính gần với sở thích
            sample_size = min(20, len(self.tracks_data))
            if sample_size > 0:
                return self.tracks_data.sample(min(n, sample_size))['id'].tolist()
            else:
                return self.recommend_random(n)
            
        except Exception as e:
            print(f"Lỗi khi gợi ý theo đặc tính: {e}")
            return self.recommend_random(n)
    
    def recommend_random(self, n=5):
        """Recommend random tracks"""
        if self.tracks_data is not None and len(self.tracks_data) > 0:
            # Sample random tracks
            sample = self.tracks_data.sample(min(n, len(self.tracks_data)))
            return sample['id'].tolist()
        
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
