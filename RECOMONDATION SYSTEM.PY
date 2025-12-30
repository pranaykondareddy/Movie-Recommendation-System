import numpy as np
import pandas as pd
from collections import defaultdict

class RecommendationSystem:
    def __init__(self):
        self.users_data = {}
        self.items_data = {}
        self.ratings_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.load_sample_data()
        
    def load_sample_data(self):
        self.items_data = {
            1: {'title': 'The Shawshank Redemption', 'genres': ['Drama', 'Crime'], 'year': 1994, 'rating': 9.3},
            2: {'title': 'The Godfather', 'genres': ['Crime', 'Drama'], 'year': 1972, 'rating': 9.2},
            3: {'title': 'The Dark Knight', 'genres': ['Action', 'Crime', 'Drama'], 'year': 2008, 'rating': 9.0},
            4: {'title': 'Pulp Fiction', 'genres': ['Crime', 'Drama'], 'year': 1994, 'rating': 8.9},
            5: {'title': 'Forrest Gump', 'genres': ['Drama', 'Romance'], 'year': 1994, 'rating': 8.8},
            6: {'title': 'Inception', 'genres': ['Action', 'Sci-Fi', 'Thriller'], 'year': 2010, 'rating': 8.8},
            7: {'title': 'The Matrix', 'genres': ['Action', 'Sci-Fi'], 'year': 1999, 'rating': 8.7},
            8: {'title': 'Interstellar', 'genres': ['Adventure', 'Drama', 'Sci-Fi'], 'year': 2014, 'rating': 8.6},
            9: {'title': 'The Silence of the Lambs', 'genres': ['Crime', 'Drama', 'Thriller'], 'year': 1991, 'rating': 8.6},
            10: {'title': 'Saving Private Ryan', 'genres': ['Drama', 'War'], 'year': 1998, 'rating': 8.6},
            11: {'title': 'Gladiator', 'genres': ['Action', 'Adventure', 'Drama'], 'year': 2000, 'rating': 8.5},
            12: {'title': 'The Departed', 'genres': ['Crime', 'Drama', 'Thriller'], 'year': 2006, 'rating': 8.5},
            13: {'title': 'The Prestige', 'genres': ['Drama', 'Mystery', 'Thriller'], 'year': 2006, 'rating': 8.5},
            14: {'title': 'Titanic', 'genres': ['Drama', 'Romance'], 'year': 1997, 'rating': 7.9},
            15: {'title': 'Avatar', 'genres': ['Action', 'Adventure', 'Fantasy'], 'year': 2009, 'rating': 7.8},
            16: {'title': 'The Avengers', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'year': 2012, 'rating': 8.0},
            17: {'title': 'Joker', 'genres': ['Crime', 'Drama', 'Thriller'], 'year': 2019, 'rating': 8.4},
            18: {'title': 'Parasite', 'genres': ['Comedy', 'Drama', 'Thriller'], 'year': 2019, 'rating': 8.6},
            19: {'title': 'The Lion King', 'genres': ['Animation', 'Adventure', 'Drama'], 'year': 1994, 'rating': 8.5},
            20: {'title': 'Toy Story', 'genres': ['Animation', 'Adventure', 'Comedy'], 'year': 1995, 'rating': 8.3}
        }
        self.users_data = {
            'User1': {1: 5, 2: 5, 3: 4, 9: 4, 12: 5},
            'User2': {1: 4, 2: 5, 4: 5, 9: 3, 12: 4},
            'User3': {3: 5, 6: 5, 7: 4, 8: 5, 16: 4},
            'User4': {3: 4, 6: 5, 7: 5, 15: 4, 16: 5},
            'User5': {5: 5, 10: 4, 14: 5, 19: 5, 20: 4},
            'User6': {5: 4, 10: 5, 14: 4, 19: 4, 20: 5},
            'User7': {1: 5, 4: 4, 13: 5, 17: 5, 18: 5},
            'User8': {2: 5, 4: 5, 9: 4, 12: 5, 17: 4}
        }
        self.build_ratings_matrix()
    
    def build_ratings_matrix(self):
        users = list(self.users_data.keys())
        items = list(self.items_data.keys())
        self.ratings_matrix = pd.DataFrame(np.nan, index=users, columns=items)
        for user, ratings in self.users_data.items():
            for item, rating in ratings.items():
                self.ratings_matrix.loc[user, item] = rating
    
    def cosine_similarity(self, vec1, vec2):
        mask = ~(np.isnan(vec1) | np.isnan(vec2))
        if not mask.any(): return 0
        vec1_clean, vec2_clean = vec1[mask], vec2[mask]
        dot_product = np.dot(vec1_clean, vec2_clean)
        norm1, norm2 = np.linalg.norm(vec1_clean), np.linalg.norm(vec2_clean)
        if norm1 == 0 or norm2 == 0: return 0
        return dot_product / (norm1 * norm2)
    
    def pearson_correlation(self, vec1, vec2):
        mask = ~(np.isnan(vec1) | np.isnan(vec2))
        if not mask.any() or mask.sum() < 2: return 0
        vec1_clean, vec2_clean = vec1[mask], vec2[mask]
        if len(vec1_clean) < 2: return 0
        return np.corrcoef(vec1_clean, vec2_clean)[0, 1]
    
    def calculate_user_similarity(self):
        users = self.ratings_matrix.index
        n_users = len(users)
        self.user_similarity = pd.DataFrame(np.zeros((n_users, n_users)), index=users, columns=users)
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users):
                if i != j:
                    vec1, vec2 = self.ratings_matrix.loc[user1].values, self.ratings_matrix.loc[user2].values
                    self.user_similarity.loc[user1, user2] = self.pearson_correlation(vec1, vec2)
                else:
                    self.user_similarity.loc[user1, user2] = 1.0
    
    def calculate_item_similarity(self):
        items = self.ratings_matrix.columns
        n_items = len(items)
        self.item_similarity = pd.DataFrame(np.zeros((n_items, n_items)), index=items, columns=items)
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i != j:
                    vec1, vec2 = self.ratings_matrix[item1].values, self.ratings_matrix[item2].values
                    self.item_similarity.loc[item1, item2] = self.cosine_similarity(vec1, vec2)
                else:
                    self.item_similarity.loc[item1, item2] = 1.0
    
    def collaborative_filtering_user_based(self, user_id, n_recommendations=5):
        if self.user_similarity is None: self.calculate_user_similarity()
        if user_id not in self.ratings_matrix.index: return []
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index
        predictions = {}
        for item in unrated_items:
            rated_users = self.ratings_matrix[item].dropna().index
            if len(rated_users) == 0: continue
            numerator, denominator = 0, 0
            for other_user in rated_users:
                if other_user == user_id: continue
                similarity = self.user_similarity.loc[user_id, other_user]
                rating = self.ratings_matrix.loc[other_user, item]
                numerator += similarity * rating
                denominator += abs(similarity)
            if denominator > 0: predictions[item] = numerator / denominator
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for item_id, predicted_rating in sorted_predictions[:n_recommendations]:
            item_info = self.items_data[item_id].copy()
            item_info['predicted_rating'] = round(predicted_rating, 2)
            item_info['method'] = 'User-Based CF'
            recommendations.append(item_info)
        return recommendations
    
    def collaborative_filtering_item_based(self, user_id, n_recommendations=5):
        if self.item_similarity is None: self.calculate_item_similarity()
        if user_id not in self.ratings_matrix.index: return []
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_items = user_ratings.dropna()
        if len(rated_items) == 0: return []
        unrated_items = user_ratings[user_ratings.isna()].index
        predictions = {}
        for item in unrated_items:
            numerator, denominator = 0, 0
            for rated_item, rating in rated_items.items():
                similarity = self.item_similarity.loc[item, rated_item]
                numerator += similarity * rating
                denominator += abs(similarity)
            if denominator > 0: predictions[item] = numerator / denominator
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for item_id, predicted_rating in sorted_predictions[:n_recommendations]:
            item_info = self.items_data[item_id].copy()
            item_info['predicted_rating'] = round(predicted_rating, 2)
            item_info['method'] = 'Item-Based CF'
            recommendations.append(item_info)
        return recommendations
    
    def content_based_filtering(self, user_id, n_recommendations=5):
        if user_id not in self.ratings_matrix.index: return []
        user_ratings = self.ratings_matrix.loc[user_id].dropna()
        if len(user_ratings) == 0: return []
        liked_items = user_ratings[user_ratings >= 4].index
        liked_genres = []
        for item_id in liked_items: liked_genres.extend(self.items_data[item_id]['genres'])
        genre_counts = defaultdict(int)
        for genre in liked_genres: genre_counts[genre] += 1
        unrated_items = self.ratings_matrix.loc[user_id][self.ratings_matrix.loc[user_id].isna()].index
        item_scores = {}
        for item_id in unrated_items:
            score = 0
            item = self.items_data[item_id]
            for genre in item['genres']: score += genre_counts.get(genre, 0)
            if len(item['genres']) > 0: score = score / len(item['genres'])
            score += item['rating'] * 0.1
            item_scores[item_id] = score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            item_info = self.items_data[item_id].copy()
            item_info['content_score'] = round(score, 2)
            item_info['method'] = 'Content-Based'
            recommendations.append(item_info)
        return recommendations
    
    def hybrid_recommendation(self, user_id, n_recommendations=5):
        user_cf = self.collaborative_filtering_user_based(user_id, n_recommendations*2)
        item_cf = self.collaborative_filtering_item_based(user_id, n_recommendations*2)
        content = self.content_based_filtering(user_id, n_recommendations*2)
        combined_scores = defaultdict(float)
        item_info_map = {}
        for rec in user_cf:
            title = rec['title']
            combined_scores[title] += rec.get('predicted_rating', 0) * 0.4
            item_info_map[title] = rec
        for rec in item_cf:
            title = rec['title']
            combined_scores[title] += rec.get('predicted_rating', 0) * 0.4
            item_info_map[title] = rec
        for rec in content:
            title = rec['title']
            combined_scores[title] += rec.get('content_score', 0) * 0.2
            item_info_map[title] = rec
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for title, score in sorted_items[:n_recommendations]:
            item_info = item_info_map[title].copy()
            item_info['hybrid_score'] = round(score, 2)
            item_info['method'] = 'Hybrid'
            recommendations.append(item_info)
        return recommendations
    
    def add_user_rating(self, user_id, item_id, rating):
        if user_id not in self.users_data: self.users_data[user_id] = {}
        self.users_data[user_id][item_id] = rating
        self.build_ratings_matrix()
        self.user_similarity = None
        self.item_similarity = None
    
    def display_recommendations(self, recommendations, title="Recommendations"):
        print("\n" + "="*80)
        print(f"  🎬 {title}")
        print("="*80)
        if not recommendations:
            print("\n  ❌ No recommendations available.\n")
            return
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']} ({rec['year']})")
            print(f"   Genres: {', '.join(rec['genres'])}")
            print(f"   IMDb Rating: ⭐ {rec['rating']}/10")
            if 'predicted_rating' in rec: print(f"   Predicted Rating: {rec['predicted_rating']}/5.0")
            if 'content_score' in rec: print(f"   Content Score: {rec['content_score']}")
            if 'hybrid_score' in rec: print(f"   Hybrid Score: {rec['hybrid_score']}")
            print(f"   Method: {rec['method']}")
        print("\n" + "="*80 + "\n")
    
    def display_all_movies(self):
        print("\n" + "="*80)
        print("  📚 AVAILABLE MOVIES")
        print("="*80 + "\n")
        for movie_id, movie in self.items_data.items():
            print(f"{movie_id:2d}. {movie['title']} ({movie['year']})")
            print(f"    Genres: {', '.join(movie['genres'])} | Rating: ⭐ {movie['rating']}/10\n")
    
    def interactive_mode(self):
        print("\n" + "="*80)
        print("  🎯 PROFESSIONAL RECOMMENDATION SYSTEM")
        print("  Powered by Collaborative & Content-Based Filtering")
        print("="*80)
        while True:
            print("\n📋 MENU:")
            print("1. View all movies")
            print("2. Rate movies (create new user)")
            print("3. Get recommendations for existing user")
            print("4. Compare recommendation methods")
            print("5. Exit")
            choice = input("\n👉 Select option (1-5): ").strip()
            if choice == '1':
                self.display_all_movies()
            elif choice == '2':
                user_id = input("\n👤 Enter your username: ").strip()
                while True:
                    self.display_all_movies()
                    movie_input = input("\n🎬 Enter movie ID to rate (or 'done'): ").strip()
                    if movie_input.lower() == 'done': break
                    try:
                        movie_id = int(movie_input)
                        rating = float(input(f"⭐ Rate '{self.items_data[movie_id]['title']}' (1-5): ").strip())
                        self.add_user_rating(user_id, movie_id, rating)
                        print(f"✅ Rating added!")
                    except: print("❌ Invalid input!")
                recs = self.hybrid_recommendation(user_id, 5)
                self.display_recommendations(recs, f"Recommendations for {user_id}")
            elif choice == '3':
                user_id = input("\n👤 Enter existing username: ").strip()
                recs = self.hybrid_recommendation(user_id, 5)
                self.display_recommendations(recs, f"Recommendations for {user_id}")
            elif choice == '4':
                user_id = input("\n👤 Enter existing username: ").strip()
                self.display_recommendations(self.collaborative_filtering_user_based(user_id,3), "User-Based CF")
                self.display_recommendations(self.collaborative_filtering_item_based(user_id,3), "Item-Based CF")
                self.display_recommendations(self.content_based_filtering(user_id,3), "Content-Based")
                self.display_recommendations(self.hybrid_recommendation(user_id,3), "Hybrid Approach")
            elif choice == '5':
                print("\n👋 Goodbye!"); break
            else:
                print("❌ Invalid choice!")

if __name__ == "__main__":
    system = RecommendationSystem()
    system.interactive_mode()
