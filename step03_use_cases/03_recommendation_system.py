#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 3단계: 추천 시스템

실제 추천 시스템을 구현합니다:
- 콘텐츠 기반 추천 (Content-Based Filtering)
- 협업 필터링 (Collaborative Filtering)
- 하이브리드 추천 시스템
- 실시간 개인화 추천
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationSystem:
    """벡터 기반 추천 시스템"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        self.collections = {}
        
    def create_item_collection(self, collection_name: str = "items") -> Collection:
        """아이템(상품/콘텐츠) 컬렉션 생성"""
        print(f"\n📁 아이템 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 - 다양한 아이템 정보
        fields = [
            FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="rating", dtype=DataType.FLOAT),
            FieldSchema(name="review_count", dtype=DataType.INT64),
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="purchase_count", dtype=DataType.INT64),
            FieldSchema(name="release_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="availability", dtype=DataType.BOOL),
            FieldSchema(name="popularity_score", dtype=DataType.FLOAT),
            FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=384),  # 콘텐츠 기반 벡터
            FieldSchema(name="behavior_vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # 행동 기반 벡터
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="추천 시스템용 아이템 컬렉션",
            enable_dynamic_field=True
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def create_user_collection(self, collection_name: str = "users") -> Collection:
        """사용자 프로필 컬렉션 생성"""
        print(f"\n📁 사용자 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 - 사용자 프로필 정보
        fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="age_group", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="preferred_categories", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="price_range", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="join_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="last_active", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="total_purchases", dtype=DataType.INT64),
            FieldSchema(name="total_spent", dtype=DataType.FLOAT),
            FieldSchema(name="avg_rating_given", dtype=DataType.FLOAT),
            FieldSchema(name="is_premium", dtype=DataType.BOOL),
            FieldSchema(name="preference_vector", dtype=DataType.FLOAT_VECTOR, dim=384),  # 사용자 선호도 벡터
            FieldSchema(name="behavior_vector", dtype=DataType.FLOAT_VECTOR, dim=128)     # 행동 패턴 벡터
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="추천 시스템용 사용자 컬렉션"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def create_interaction_collection(self, collection_name: str = "interactions") -> Collection:
        """사용자-아이템 상호작용 컬렉션 생성"""
        print(f"\n📁 상호작용 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 - 사용자-아이템 상호작용
        fields = [
            FieldSchema(name="interaction_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="item_id", dtype=DataType.INT64),
            FieldSchema(name="interaction_type", dtype=DataType.VARCHAR, max_length=50),  # view, like, purchase, rating
            FieldSchema(name="rating", dtype=DataType.FLOAT),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=30),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="device_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="duration", dtype=DataType.INT64),  # 초 단위
            FieldSchema(name="context_vector", dtype=DataType.FLOAT_VECTOR, dim=64)  # 상황 정보 벡터
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="사용자-아이템 상호작용 컬렉션"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def generate_sample_items(self, count: int = 1000) -> List[Dict[str, Any]]:
        """샘플 아이템 데이터 생성"""
        print(f"\n📊 샘플 아이템 {count}개 생성 중...")
        
        categories = {
            "Electronics": ["Smartphone", "Laptop", "Tablet", "Headphones", "Smart Watch"],
            "Fashion": ["Clothing", "Shoes", "Accessories", "Bags", "Jewelry"],
            "Books": ["Fiction", "Non-Fiction", "Academic", "Children", "Comics"],
            "Home": ["Furniture", "Kitchen", "Decor", "Garden", "Tools"],
            "Sports": ["Equipment", "Apparel", "Fitness", "Outdoor", "Supplements"],
            "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance", "Tools"],
            "Food": ["Snacks", "Beverages", "Fresh", "Organic", "International"]
        }
        
        brands = {
            "Electronics": ["Apple", "Samsung", "Sony", "LG", "Xiaomi", "Google"],
            "Fashion": ["Nike", "Adidas", "Zara", "H&M", "Uniqlo", "Gucci"],
            "Books": ["Penguin", "Harper", "Random House", "Scholastic", "Marvel"],
            "Home": ["IKEA", "Home Depot", "Wayfair", "Target", "Amazon Basics"],
            "Sports": ["Nike", "Adidas", "Under Armour", "Puma", "Reebok"],
            "Beauty": ["L'Oreal", "Maybelline", "MAC", "Clinique", "Sephora"],
            "Food": ["Nestle", "Coca Cola", "PepsiCo", "General Mills", "Kellogg's"]
        }
        
        items = []
        
        for i in range(count):
            item_id = i + 1
            category = np.random.choice(list(categories.keys()))
            subcategory = np.random.choice(categories[category])
            brand = np.random.choice(brands[category])
            
            # 제목 생성
            title = f"{brand} {subcategory} Model {np.random.randint(100, 9999)}"
            
            # 설명 생성
            description = f"High-quality {subcategory.lower()} from {brand}. "
            description += f"Perfect for {category.lower()} enthusiasts. "
            description += f"Features advanced technology and superior design. "
            description += f"Ideal for both professionals and casual users."
            
            # 태그 생성
            base_tags = [category.lower(), subcategory.lower(), brand.lower()]
            additional_tags = ["popular", "trending", "bestseller", "premium", "eco-friendly", "limited-edition"]
            selected_tags = np.random.choice(additional_tags, size=np.random.randint(1, 4), replace=False)
            tags = ", ".join(base_tags + list(selected_tags))
            
            # 가격 생성 (카테고리별 차등)
            price_ranges = {
                "Electronics": (50, 2000),
                "Fashion": (20, 500),
                "Books": (10, 100),
                "Home": (15, 800),
                "Sports": (25, 300),
                "Beauty": (10, 200),
                "Food": (5, 50)
            }
            min_price, max_price = price_ranges[category]
            price = round(np.random.uniform(min_price, max_price), 2)
            
            # 평점 및 리뷰 수
            rating = round(np.random.uniform(3.0, 5.0), 1)
            review_count = np.random.randint(10, 1000)
            
            # 조회수, 구매수
            view_count = np.random.randint(100, 10000)
            purchase_count = int(view_count * np.random.uniform(0.01, 0.1))  # 1-10% 구매율
            
            # 인기도 점수 (복합적 계산)
            popularity_score = (rating * 0.3 + 
                             min(review_count / 100, 10) * 0.2 + 
                             min(view_count / 1000, 10) * 0.3 + 
                             min(purchase_count / 50, 10) * 0.2)
            
            # 출시일
            release_date = f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            
            item = {
                "item_id": item_id,
                "title": title,
                "description": description,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "tags": tags,
                "price": price,
                "rating": rating,
                "review_count": review_count,
                "view_count": view_count,
                "purchase_count": purchase_count,
                "release_date": release_date,
                "availability": np.random.random() > 0.05,  # 95% 재고 있음
                "popularity_score": popularity_score
            }
            
            items.append(item)
        
        print(f"  ✅ {count}개 아이템 생성 완료")
        return items
    
    def generate_sample_users(self, count: int = 500) -> List[Dict[str, Any]]:
        """샘플 사용자 데이터 생성"""
        print(f"\n👥 샘플 사용자 {count}개 생성 중...")
        
        age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        genders = ["Male", "Female", "Other"]
        locations = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon"]
        price_ranges = ["Budget", "Mid-range", "Premium", "Luxury"]
        
        categories = ["Electronics", "Fashion", "Books", "Home", "Sports", "Beauty", "Food"]
        
        users = []
        
        for i in range(count):
            user_id = i + 1
            age_group = np.random.choice(age_groups)
            gender = np.random.choice(genders)
            location = np.random.choice(locations)
            price_range = np.random.choice(price_ranges)
            
            # 선호 카테고리 (1-3개)
            num_prefs = np.random.randint(1, 4)
            preferred_cats = np.random.choice(categories, size=num_prefs, replace=False)
            preferred_categories = ", ".join(preferred_cats)
            
            # 가입일 및 마지막 활동일
            join_date = f"2024-{np.random.randint(1, 8):02d}-{np.random.randint(1, 29):02d}"
            last_active = f"2024-{np.random.randint(8, 13):02d}-{np.random.randint(1, 29):02d}"
            
            # 구매 이력
            total_purchases = np.random.randint(0, 50)
            total_spent = round(total_purchases * np.random.uniform(50, 500), 2)
            
            # 평점 평균 (평점을 주는 사용자만)
            avg_rating_given = round(np.random.uniform(3.0, 5.0), 1) if total_purchases > 0 else 0.0
            
            # 프리미엄 여부
            is_premium = np.random.random() > 0.8  # 20% 프리미엄
            
            user = {
                "user_id": user_id,
                "age_group": age_group,
                "gender": gender,
                "location": location,
                "preferred_categories": preferred_categories,
                "price_range": price_range,
                "join_date": join_date,
                "last_active": last_active,
                "total_purchases": total_purchases,
                "total_spent": total_spent,
                "avg_rating_given": avg_rating_given,
                "is_premium": is_premium
            }
            
            users.append(user)
        
        print(f"  ✅ {count}개 사용자 생성 완료")
        return users
    
    def generate_sample_interactions(self, users: List[Dict], items: List[Dict], count: int = 5000) -> List[Dict[str, Any]]:
        """샘플 상호작용 데이터 생성"""
        print(f"\n🔄 샘플 상호작용 {count}개 생성 중...")
        
        interaction_types = ["view", "like", "add_to_cart", "purchase", "rating", "review"]
        device_types = ["mobile", "desktop", "tablet"]
        
        interactions = []
        
        # 사용자별로 realistic한 상호작용 패턴 생성
        for _ in range(count):
            user = np.random.choice(users)
            item = np.random.choice(items)
            
            user_id = user["user_id"]
            item_id = item["item_id"]
            
            # 사용자의 선호 카테고리와 아이템 카테고리 매칭으로 상호작용 확률 조정
            user_prefs = user["preferred_categories"].split(", ")
            interaction_prob = 0.3  # 기본 확률
            if item["category"] in user_prefs:
                interaction_prob = 0.8  # 선호 카테고리면 높은 확률
            
            if np.random.random() > interaction_prob:
                continue
            
            # 상호작용 타입 선택 (순차적 확률)
            interaction_type = "view"  # 기본적으로 조회
            if np.random.random() > 0.7:  # 30% 확률로 좋아요
                interaction_type = "like"
            if np.random.random() > 0.85:  # 15% 확률로 장바구니
                interaction_type = "add_to_cart"
            if np.random.random() > 0.95:  # 5% 확률로 구매
                interaction_type = "purchase"
            if interaction_type == "purchase" and np.random.random() > 0.7:  # 구매 후 30% 확률로 평점
                interaction_type = "rating"
            
            # 평점 (rating인 경우에만)
            rating = 0.0
            if interaction_type == "rating":
                # 아이템 평점 근처에서 생성
                rating = max(1.0, min(5.0, item["rating"] + np.random.normal(0, 0.5)))
                rating = round(rating, 1)
            
            # 타임스탬프
            timestamp = f"2024-{np.random.randint(9, 13):02d}-{np.random.randint(1, 29):02d} {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
            
            # 세션 ID
            session_id = f"sess_{user_id}_{np.random.randint(1000, 9999)}"
            
            # 디바이스 타입
            device_type = np.random.choice(device_types)
            
            # 지속 시간 (초)
            duration_ranges = {
                "view": (10, 300),      # 10초 ~ 5분
                "like": (5, 60),        # 5초 ~ 1분
                "add_to_cart": (30, 180), # 30초 ~ 3분
                "purchase": (120, 600),  # 2분 ~ 10분
                "rating": (60, 300)      # 1분 ~ 5분
            }
            min_dur, max_dur = duration_ranges.get(interaction_type, (10, 300))
            duration = np.random.randint(min_dur, max_dur)
            
            interaction = {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": timestamp,
                "session_id": session_id,
                "device_type": device_type,
                "duration": duration
            }
            
            interactions.append(interaction)
        
        print(f"  ✅ {len(interactions)}개 상호작용 생성 완료")
        return interactions
    
    def insert_items_with_vectors(self, collection: Collection, items: List[Dict[str, Any]]) -> None:
        """아이템 데이터와 벡터 삽입"""
        print(f"\n💾 아이템 데이터 삽입 중...")
        
        # 콘텐츠 벡터 생성 (제목 + 설명 + 태그)
        print("  콘텐츠 벡터 생성 중...")
        content_texts = []
        for item in items:
            content_text = f"{item['title']} {item['description']} {item['tags']} {item['category']} {item['brand']}"
            content_texts.append(content_text)
        
        content_vectors = self.vector_utils.texts_to_vectors(content_texts)
        
        # 행동 기반 벡터 생성 (간단한 수치 특성 기반)
        print("  행동 벡터 생성 중...")
        behavior_vectors = []
        for item in items:
            # 수치적 특징들을 정규화하여 128차원 벡터 생성
            features = [
                item['price'] / 1000,  # 가격 정규화
                item['rating'] / 5,    # 평점 정규화
                min(item['review_count'] / 100, 10) / 10,  # 리뷰수 정규화
                min(item['view_count'] / 1000, 10) / 10,   # 조회수 정규화
                min(item['purchase_count'] / 50, 10) / 10,  # 구매수 정규화
                item['popularity_score'] / 10  # 인기도 정규화
            ]
            
            # 128차원으로 확장 (패딩 + 노이즈)
            extended_features = features + [0] * (128 - len(features))
            extended_features = np.array(extended_features) + np.random.normal(0, 0.01, 128)
            behavior_vectors.append(extended_features)
        
        behavior_vectors = np.array(behavior_vectors)
        
        # 데이터를 2단계 패턴으로 구성 (List[List])
        data = [
            [item["item_id"] for item in items],
            [item["title"] for item in items],
            [item["description"] for item in items],
            [item["category"] for item in items],
            [item["subcategory"] for item in items],
            [item["brand"] for item in items],
            [item["tags"] for item in items],
            [item["price"] for item in items],
            [item["rating"] for item in items],
            [item["review_count"] for item in items],
            [item["view_count"] for item in items],
            [item["purchase_count"] for item in items],
            [item["release_date"] for item in items],
            [item["availability"] for item in items],
            [item["popularity_score"] for item in items],
            content_vectors.tolist(),
            behavior_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(items)
        print(f"  ✅ {total_inserted}개 아이템 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 아이템 삽입 완료")
    
    def insert_users_with_vectors(self, collection: Collection, users: List[Dict[str, Any]]) -> None:
        """사용자 데이터와 벡터 삽입"""
        print(f"\n💾 사용자 데이터 삽입 중...")
        
        # 선호도 벡터 생성 (선호 카테고리 기반)
        print("  사용자 선호도 벡터 생성 중...")
        preference_texts = []
        for user in users:
            pref_text = f"{user['preferred_categories']} {user['age_group']} {user['gender']} {user['price_range']}"
            preference_texts.append(pref_text)
        
        preference_vectors = self.vector_utils.texts_to_vectors(preference_texts)
        
        # 행동 벡터 생성 (구매 이력 기반)
        print("  사용자 행동 벡터 생성 중...")
        behavior_vectors = []
        for user in users:
            features = [
                min(user['total_purchases'] / 10, 10) / 10,  # 구매수 정규화
                min(user['total_spent'] / 1000, 10) / 10,    # 지출액 정규화
                user['avg_rating_given'] / 5 if user['avg_rating_given'] > 0 else 0,  # 평점 정규화
                1 if user['is_premium'] else 0  # 프리미엄 여부
            ]
            
            # 128차원으로 확장
            extended_features = features + [0] * (128 - len(features))
            extended_features = np.array(extended_features) + np.random.normal(0, 0.01, 128)
            behavior_vectors.append(extended_features)
        
        behavior_vectors = np.array(behavior_vectors)
        
        # 데이터를 2단계 패턴으로 구성 (List[List])
        data = [
            [user["user_id"] for user in users],
            [user["age_group"] for user in users],
            [user["gender"] for user in users],
            [user["location"] for user in users],
            [user["preferred_categories"] for user in users],
            [user["price_range"] for user in users],
            [user["join_date"] for user in users],
            [user["last_active"] for user in users],
            [user["total_purchases"] for user in users],
            [user["total_spent"] for user in users],
            [user["avg_rating_given"] for user in users],
            [user["is_premium"] for user in users],
            preference_vectors.tolist(),
            behavior_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(users)
        print(f"  ✅ {total_inserted}개 사용자 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 사용자 삽입 완료")
    
    def insert_interactions_with_vectors(self, collection: Collection, interactions: List[Dict[str, Any]]) -> None:
        """상호작용 데이터와 벡터 삽입"""
        print(f"\n💾 상호작용 데이터 삽입 중...")
        
        # 상황 벡터 생성 (시간, 디바이스, 행동 패턴 기반)
        print("  상황 벡터 생성 중...")
        context_vectors = []
        
        for interaction in interactions:
            # 시간 정보 파싱
            timestamp = interaction['timestamp']
            hour = int(timestamp.split()[1].split(':')[0])
            
            # 상황 특징
            features = [
                hour / 24,  # 시간 정규화 (0-1)
                1 if interaction['device_type'] == 'mobile' else 0,
                1 if interaction['device_type'] == 'desktop' else 0,
                1 if interaction['device_type'] == 'tablet' else 0,
                min(interaction['duration'] / 300, 1),  # 지속시간 정규화 (0-1)
                1 if interaction['interaction_type'] == 'view' else 0,
                1 if interaction['interaction_type'] == 'like' else 0,
                1 if interaction['interaction_type'] == 'purchase' else 0,
                interaction['rating'] / 5 if interaction['rating'] > 0 else 0
            ]
            
            # 64차원으로 확장
            extended_features = features + [0] * (64 - len(features))
            extended_features = np.array(extended_features) + np.random.normal(0, 0.001, 64)
            context_vectors.append(extended_features)
        
        context_vectors = np.array(context_vectors)
        
        # 데이터를 2단계 패턴으로 구성 (List[List] - auto_id 필드 제외)
        data = [
            [interaction["user_id"] for interaction in interactions],
            [interaction["item_id"] for interaction in interactions],
            [interaction["interaction_type"] for interaction in interactions],
            [interaction["rating"] for interaction in interactions],
            [interaction["timestamp"] for interaction in interactions],
            [interaction["session_id"] for interaction in interactions],
            [interaction["device_type"] for interaction in interactions],
            [interaction["duration"] for interaction in interactions],
            context_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(interactions)
        print(f"  ✅ {total_inserted}개 상호작용 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 상호작용 삽입 완료")
    
    def create_indexes(self, collection: Collection, vector_fields: List[str]) -> None:
        """인덱스 생성"""
        print(f"\n🔍 인덱스 생성 중...")
        
        for field_name in vector_fields:
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            print(f"  {field_name} 인덱스 생성 중...")
            collection.create_index(field_name, index_params)
            print(f"  ✅ {field_name} 인덱스 생성 완료")
        
        print(f"  ✅ 모든 인덱스 생성 완료")
    
    def content_based_recommendation_demo(self, item_collection: Collection) -> None:
        """콘텐츠 기반 추천 데모"""
        print("\n" + "="*80)
        print(" 📚 콘텐츠 기반 추천 시스템 데모")
        print("="*80)
        
        item_collection.load()
        
        # 사용자가 관심 있어 할 만한 아이템을 텍스트로 표현
        user_interests = [
            {
                "description": "최신 스마트폰과 모바일 기술에 관심이 많은 사용자",
                "query": "smartphone mobile technology latest features premium",
                "category_filter": "Electronics"
            },
            {
                "description": "패션과 스타일에 관심이 많은 사용자",
                "query": "fashion clothing style trendy designer premium",
                "category_filter": "Fashion"
            },
            {
                "description": "건강과 운동에 관심이 많은 사용자",
                "query": "fitness sports health workout equipment training",
                "category_filter": "Sports"
            },
            {
                "description": "독서와 지식 습득을 좋아하는 사용자",
                "query": "books learning education knowledge fiction academic",
                "category_filter": "Books"
            }
        ]
        
        for i, interest in enumerate(user_interests, 1):
            print(f"\n{i}. {interest['description']}")
            print(f"   관심사: '{interest['query']}'")
            
            # 관심사 벡터화
            query_vectors = self.vector_utils.text_to_vector(interest['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 콘텐츠 기반 유사도 검색
            start_time = time.time()
            results = item_collection.search(
                data=[query_vector.tolist()],
                anns_field="content_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["title", "category", "brand", "price", "rating", "description"]
            )
            search_time = time.time() - start_time
            
            print(f"   검색 시간: {search_time:.4f}초")
            print(f"   추천 아이템 수: {len(results[0])}")
            
            for j, hit in enumerate(results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')}")
                print(f"        카테고리: {entity.get('category')}, 브랜드: {entity.get('brand')}")
                print(f"        가격: ${entity.get('price'):.2f}, 평점: {entity.get('rating')}")
                print(f"        설명: {entity.get('description')[:80]}...")
                print(f"        유사도: {similarity:.3f}")
            
            # 카테고리 필터링 추천
            if interest.get('category_filter'):
                print(f"\n   📁 카테고리 필터링 추천 ({interest['category_filter']})")
                
                category_results = item_collection.search(
                    data=[query_vector.tolist()],
                    anns_field="content_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 200}},
                    limit=3,
                    expr=f"category == '{interest['category_filter']}' and availability == True",
                    output_fields=["title", "brand", "price", "rating", "popularity_score"]
                )
                
                print(f"   카테고리 내 추천 수: {len(category_results[0])}")
                for j, hit in enumerate(category_results[0]):
                    similarity = 1 - hit.distance
                    entity = hit.entity
                    print(f"     {j+1}. {entity.get('title')}")
                    print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
                    print(f"        평점: {entity.get('rating')}, 인기도: {entity.get('popularity_score'):.2f}")
                    print(f"        유사도: {similarity:.3f}")
    
    def collaborative_filtering_demo(self, user_collection: Collection, item_collection: Collection) -> None:
        """협업 필터링 데모"""
        print("\n" + "="*80)
        print(" 👥 협업 필터링 추천 시스템 데모")
        print("="*80)
        
        user_collection.load()
        item_collection.load()
        
        # 타겟 사용자 시뮬레이션
        target_users = [
            {
                "description": "젊은 전자제품 애호가",
                "profile": "18-25 Electronics premium mobile technology",
                "details": "18-25세, 전자제품 선호, 프리미엄 구매력"
            },
            {
                "description": "패션에 관심 많은 중년 여성",
                "profile": "36-45 Fashion Female premium clothing style",
                "details": "36-45세, 패션 관심, 여성, 프리미엄 스타일"
            },
            {
                "description": "운동을 좋아하는 남성",
                "profile": "26-35 Sports Male fitness equipment training",
                "details": "26-35세, 스포츠 관심, 남성, 피트니스"
            }
        ]
        
        for i, target in enumerate(target_users, 1):
            print(f"\n{i}. {target['description']}")
            print(f"   프로필: {target['details']}")
            
            # 타겟 사용자 프로필 벡터화
            profile_vectors = self.vector_utils.text_to_vector(target['profile'])
            profile_vector = profile_vectors[0] if len(profile_vectors.shape) > 1 else profile_vectors
            
            # 유사한 사용자 찾기
            print(f"\n   👥 유사한 사용자 찾기:")
            similar_users = user_collection.search(
                data=[profile_vector.tolist()],
                anns_field="preference_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["user_id", "age_group", "gender", "preferred_categories", "total_purchases", "is_premium"]
            )
            
            user_ids = []
            for j, hit in enumerate(similar_users[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                user_ids.append(entity.get('user_id'))
                print(f"     사용자 {entity.get('user_id')}: {entity.get('age_group')}, {entity.get('gender')}")
                print(f"       선호: {entity.get('preferred_categories')}")
                print(f"       구매: {entity.get('total_purchases')}회, 프리미엄: {entity.get('is_premium')}")
                print(f"       유사도: {similarity:.3f}")
            
            # 유사한 사용자들이 선호하는 아이템 패턴 분석
            print(f"\n   🎯 유사 사용자 기반 추천:")
            
            # 유사 사용자들의 선호 카테고리 집계
            all_preferences = []
            for hit in similar_users[0]:
                try:
                    prefs = (hit.entity.get('preferred_categories') or '').split(', ')
                except:
                    prefs = []
                all_preferences.extend([p for p in prefs if p.strip()])
            
            # 가장 인기 있는 카테고리 찾기
            if all_preferences:
                from collections import Counter
                pref_counter = Counter(all_preferences)
                top_categories = [cat for cat, count in pref_counter.most_common(2)]
                
                print(f"   인기 카테고리: {', '.join(top_categories)}")
                
                # 인기 카테고리에서 고품질 아이템 추천
                for category in top_categories[:1]:  # 가장 인기 있는 카테고리만
                    category_items = item_collection.search(
                        data=[[0.0] * 384],  # 더미 벡터
                        anns_field="content_vector",
                        param={"metric_type": "COSINE", "params": {"ef": 100}},
                        limit=10,
                        expr=f"category == '{category}' and rating >= 4.0 and availability == True",
                        output_fields=["title", "brand", "price", "rating", "popularity_score"]
                    )
                    
                    if category_items and len(category_items[0]) > 0:
                        # 인기도 순으로 정렬
                        def get_popularity(x):
                            try:
                                return x.entity.get('popularity_score') or 0
                            except:
                                return 0
                        sorted_items = sorted(category_items[0], 
                                            key=get_popularity, 
                                            reverse=True)
                        
                        print(f"\n     📈 {category} 카테고리 인기 아이템:")
                        for k, hit in enumerate(sorted_items[:3], 1):
                            entity = hit.entity
                            print(f"       {k}. {entity.get('title')}")
                            print(f"          브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
                            print(f"          평점: {entity.get('rating')}, 인기도: {entity.get('popularity_score'):.2f}")
    
    def hybrid_recommendation_demo(self, user_collection: Collection, item_collection: Collection) -> None:
        """하이브리드 추천 데모"""
        print("\n" + "="*80)
        print(" 🔄 하이브리드 추천 시스템 데모")
        print("="*80)
        
        user_collection.load()
        item_collection.load()
        
        # 타겟 사용자 정의
        target_user = {
            "description": "테크 관심사와 프리미엄 취향을 가진 사용자",
            "interests": "smartphone laptop premium technology innovation",
            "profile": "26-35 Electronics premium technology gadget",
            "budget": "premium",
            "details": "26-35세, 전자제품 애호가, 프리미엄 선호, 최신 기술 관심"
        }
        
        print(f"타겟 사용자: {target_user['description']}")
        print(f"상세 정보: {target_user['details']}")
        
        # 1. 콘텐츠 기반 추천
        print(f"\n1️⃣ 콘텐츠 기반 추천")
        interests_vector = self.vector_utils.text_to_vector(target_user['interests'])
        interests_vector = interests_vector[0] if len(interests_vector.shape) > 1 else interests_vector
        
        content_results = item_collection.search(
            data=[interests_vector.tolist()],
            anns_field="content_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=10,
            output_fields=["item_id", "title", "category", "brand", "price", "rating", "popularity_score"]
        )
        
        content_recommendations = {}
        print("   콘텐츠 기반 추천 결과:")
        for j, hit in enumerate(content_results[0][:5]):
            similarity = 1 - hit.distance
            entity = hit.entity
            item_id = entity.get('item_id')
            content_recommendations[item_id] = {
                'similarity': similarity,
                'method': 'content',
                'item': entity
            }
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        가격: ${entity.get('price'):.2f}, 평점: {entity.get('rating')}")
            print(f"        콘텐츠 유사도: {similarity:.3f}")
        
        # 2. 협업 필터링 추천
        print(f"\n2️⃣ 협업 필터링 추천")
        profile_vector = self.vector_utils.text_to_vector(target_user['profile'])
        profile_vector = profile_vector[0] if len(profile_vector.shape) > 1 else profile_vector
        
        # 유사 사용자 찾기
        similar_users = user_collection.search(
            data=[profile_vector.tolist()],
            anns_field="preference_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=3,
            output_fields=["user_id", "preferred_categories", "total_purchases", "is_premium"]
        )
        
        print("   유사한 사용자들:")
        for hit in similar_users[0]:
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     사용자 {entity.get('user_id')}: 선호 {entity.get('preferred_categories')}")
            print(f"       구매 {entity.get('total_purchases')}회, 프리미엄: {entity.get('is_premium')}, 유사도: {similarity:.3f}")
        
        # 유사 사용자 선호 기반 아이템 추천
        collaborative_results = item_collection.search(
            data=[[0.0] * 384],  # 더미 벡터
            anns_field="content_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=10,
            expr="category == 'Electronics' and rating >= 4.0 and price >= 500",  # 프리미엄 조건
            output_fields=["item_id", "title", "brand", "price", "rating", "popularity_score"]
        )
        
        collaborative_recommendations = {}
        print("\n   협업 필터링 추천 결과:")
        for j, hit in enumerate(collaborative_results[0][:5]):
            entity = hit.entity
            item_id = entity.get('item_id')
            # 인기도를 유사도로 사용
            try:
                popularity_score = entity.get('popularity_score') or 0
            except:
                popularity_score = 0
            popularity_sim = min(popularity_score / 10, 1.0)
            collaborative_recommendations[item_id] = {
                'similarity': popularity_sim,
                'method': 'collaborative',
                'item': entity
            }
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
            print(f"        인기도 기반 점수: {popularity_sim:.3f}")
        
        # 3. 하이브리드 결합
        print(f"\n3️⃣ 하이브리드 추천 (가중 결합)")
        
        # 모든 추천 아이템 수집
        all_recommendations = {}
        
        # 콘텐츠 기반 결과 추가 (가중치 0.6)
        for item_id, rec in content_recommendations.items():
            all_recommendations[item_id] = {
                'item': rec['item'],
                'content_score': rec['similarity'] * 0.6,
                'collaborative_score': 0.0,
                'methods': ['content']
            }
        
        # 협업 필터링 결과 추가 (가중치 0.4)
        for item_id, rec in collaborative_recommendations.items():
            if item_id in all_recommendations:
                all_recommendations[item_id]['collaborative_score'] = rec['similarity'] * 0.4
                all_recommendations[item_id]['methods'].append('collaborative')
            else:
                all_recommendations[item_id] = {
                    'item': rec['item'],
                    'content_score': 0.0,
                    'collaborative_score': rec['similarity'] * 0.4,
                    'methods': ['collaborative']
                }
        
        # 최종 점수 계산 및 정렬
        for item_id in all_recommendations:
            rec = all_recommendations[item_id]
            rec['final_score'] = rec['content_score'] + rec['collaborative_score']
            # 두 방법 모두에서 추천된 경우 보너스
            if len(rec['methods']) > 1:
                rec['final_score'] *= 1.2
        
        # 최종 점수 순으로 정렬
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        print("   최종 하이브리드 추천 결과:")
        for j, (item_id, rec) in enumerate(sorted_recommendations[:5], 1):
            entity = rec['item']
            methods_str = ' + '.join(rec['methods'])
            print(f"     {j}. {entity.get('title')}")
            print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
            print(f"        평점: {entity.get('rating')}")
            print(f"        콘텐츠 점수: {rec['content_score']:.3f}, 협업 점수: {rec['collaborative_score']:.3f}")
            print(f"        최종 점수: {rec['final_score']:.3f} ({methods_str})")
    
    def real_time_recommendation_demo(self, user_collection: Collection, item_collection: Collection, interaction_collection: Collection) -> None:
        """실시간 개인화 추천 데모"""
        print("\n" + "="*80)
        print(" ⚡ 실시간 개인화 추천 시스템 데모")
        print("="*80)
        
        user_collection.load()
        item_collection.load()
        interaction_collection.load()
        
        # 실시간 사용자 행동 시뮬레이션
        current_user = {
            "user_id": 123,
            "current_session": "sess_123_realtime",
            "recent_actions": [
                {"action": "view", "item_category": "Electronics", "item_type": "Smartphone", "duration": 180},
                {"action": "view", "item_category": "Electronics", "item_type": "Laptop", "duration": 240},
                {"action": "like", "item_category": "Electronics", "item_type": "Headphones", "duration": 30},
                {"action": "add_to_cart", "item_category": "Electronics", "item_type": "Smartphone", "duration": 120}
            ]
        }
        
        print(f"사용자 ID: {current_user['user_id']}")
        print(f"현재 세션: {current_user['current_session']}")
        print("최근 행동:")
        for action in current_user['recent_actions']:
            print(f"  - {action['action']}: {action['item_category']} > {action['item_type']} ({action['duration']}초)")
        
        # 1. 현재 세션 기반 즉시 추천
        print(f"\n1️⃣ 현재 세션 기반 즉시 추천")
        
        # 현재 관심사 추출
        current_interests = []
        category_weights = defaultdict(float)
        action_weights = {"view": 1.0, "like": 2.0, "add_to_cart": 3.0, "purchase": 5.0}
        
        for action in current_user['recent_actions']:
            weight = action_weights.get(action['action'], 1.0)
            duration_bonus = min(action['duration'] / 120, 2.0)  # 최대 2배 보너스
            final_weight = weight * duration_bonus
            
            category_weights[action['item_category']] += final_weight
            current_interests.append(f"{action['item_category']} {action['item_type']}")
        
        # 가장 관심 있는 카테고리
        top_category = max(category_weights.items(), key=lambda x: x[1])[0]
        interest_text = " ".join(current_interests)
        
        print(f"   추출된 관심사: {interest_text}")
        print(f"   주요 관심 카테고리: {top_category} (가중치: {category_weights[top_category]:.2f})")
        
        # 관심사 기반 즉시 추천
        interest_vector = self.vector_utils.text_to_vector(interest_text)
        interest_vector = interest_vector[0] if len(interest_vector.shape) > 1 else interest_vector
        
        immediate_results = item_collection.search(
            data=[interest_vector.tolist()],
            anns_field="content_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=5,
            expr=f"category == '{top_category}' and availability == True",
            output_fields=["title", "brand", "price", "rating", "view_count"]
        )
        
        print("   즉시 추천 결과:")
        for j, hit in enumerate(immediate_results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
            print(f"        평점: {entity.get('rating')}, 조회수: {entity.get('view_count')}")
            print(f"        유사도: {similarity:.3f}")
        
        # 2. 시간 기반 트렌딩 추천
        print(f"\n2️⃣ 실시간 트렌딩 추천")
        
        # 최근 인기 아이템 (조회수 기반)
        trending_results = item_collection.search(
            data=[[0.0] * 384],  # 더미 벡터
            anns_field="content_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=10,
            expr=f"category == '{top_category}' and view_count >= 1000",
            output_fields=["title", "brand", "view_count", "purchase_count", "popularity_score"]
        )
        
        if trending_results and len(trending_results[0]) > 0:
            # 조회수 기준 정렬
            def get_view_count(x):
                try:
                    return x.entity.get('view_count') or 0
                except:
                    return 0
            sorted_trending = sorted(trending_results[0], 
                                   key=get_view_count, 
                                   reverse=True)
            
            print("   실시간 트렌딩 아이템:")
            for j, hit in enumerate(sorted_trending[:3], 1):
                entity = hit.entity
                try:
                    purchase_count = entity.get('purchase_count') or 0
                    view_count = entity.get('view_count') or 1
                except:
                    purchase_count = 0
                    view_count = 1
                conversion_rate = purchase_count / max(view_count, 1) * 100
                print(f"     {j}. {entity.get('title')}")
                print(f"        브랜드: {entity.get('brand')}")
                print(f"        조회수: {entity.get('view_count'):,}, 구매수: {entity.get('purchase_count')}")
                print(f"        전환율: {conversion_rate:.2f}%, 인기도: {entity.get('popularity_score'):.2f}")
        
        # 3. 상황 인식 추천
        print(f"\n3️⃣ 상황 인식 추천 (시간/디바이스 기반)")
        
        current_hour = datetime.now().hour
        current_device = "mobile"  # 시뮬레이션
        
        print(f"   현재 시간: {current_hour}시, 디바이스: {current_device}")
        
        # 시간대별 추천 로직
        if 9 <= current_hour <= 18:  # 업무 시간
            context_desc = "업무용 제품"
            context_query = "professional work business productivity"
        elif 19 <= current_hour <= 23:  # 저녁/여가 시간
            context_desc = "여가/엔터테인먼트 제품"
            context_query = "entertainment leisure gaming relaxation"
        else:  # 밤/새벽
            context_desc = "개인용/프라이빗 제품"
            context_query = "personal private quiet comfort"
        
        print(f"   상황 기반 추천: {context_desc}")
        
        # 상황 + 관심사 결합 검색
        contextual_query = f"{interest_text} {context_query}"
        contextual_vector = self.vector_utils.text_to_vector(contextual_query)
        contextual_vector = contextual_vector[0] if len(contextual_vector.shape) > 1 else contextual_vector
        
        contextual_results = item_collection.search(
            data=[contextual_vector.tolist()],
            anns_field="content_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=4,
            expr=f"category == '{top_category}' and rating >= 4.0",
            output_fields=["title", "brand", "price", "rating"]
        )
        
        print("   상황 인식 추천 결과:")
        for j, hit in enumerate(contextual_results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
            print(f"        평점: {entity.get('rating')}, 상황 적합도: {similarity:.3f}")
        
        # 4. 개인화 점수 종합
        print(f"\n4️⃣ 최종 개인화 추천 점수")
        
        print("   개인화 점수 계산 요소:")
        print(f"     - 현재 세션 관심도: {category_weights[top_category]:.2f}")
        print(f"     - 시간대 상황 점수: {0.8 if 9 <= current_hour <= 23 else 0.5:.1f}")
        print(f"     - 디바이스 적합성: {0.9 if current_device == 'mobile' else 0.7:.1f}")
        print(f"     - 실시간 트렌드 반영: 활성화")
        
        final_score = (category_weights[top_category] * 0.4 + 
                      (0.8 if 9 <= current_hour <= 23 else 0.5) * 0.3 + 
                      (0.9 if current_device == 'mobile' else 0.7) * 0.3)
        
        print(f"   최종 개인화 점수: {final_score:.3f}")
        print("   ✅ 실시간 추천 준비 완료!")


def main():
    """메인 실행 함수"""
    print("🚀 벡터 기반 추천 시스템 실습")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 추천 시스템 초기화
            rec_system = RecommendationSystem(conn)
            
            # 1. 데이터 수집 및 저장
            print("\n" + "="*80)
            print(" 🗃️ 추천 시스템 데이터 구축")
            print("="*80)
            
            # 컬렉션 생성
            item_collection = rec_system.create_item_collection()
            user_collection = rec_system.create_user_collection()
            interaction_collection = rec_system.create_interaction_collection()
            
            # 샘플 데이터 생성
            items = rec_system.generate_sample_items(1000)
            users = rec_system.generate_sample_users(500)
            interactions = rec_system.generate_sample_interactions(users, items, 5000)
            
            # 데이터 삽입
            rec_system.insert_items_with_vectors(item_collection, items)
            rec_system.insert_users_with_vectors(user_collection, users)
            rec_system.insert_interactions_with_vectors(interaction_collection, interactions)
            
            # 인덱스 생성
            rec_system.create_indexes(item_collection, ["content_vector", "behavior_vector"])
            rec_system.create_indexes(user_collection, ["preference_vector", "behavior_vector"])
            rec_system.create_indexes(interaction_collection, ["context_vector"])
            
            # 2. 추천 시스템 데모
            rec_system.content_based_recommendation_demo(item_collection)
            rec_system.collaborative_filtering_demo(user_collection, item_collection)
            rec_system.hybrid_recommendation_demo(user_collection, item_collection)
            rec_system.real_time_recommendation_demo(user_collection, item_collection, interaction_collection)
            
            # 컬렉션 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            collections_to_clean = ["items", "users", "interactions"]
            for collection_name in collections_to_clean:
                if utility.has_collection(collection_name):
                    utility.drop_collection(collection_name)
            print("✅ 정리 완료")
            
        print("\n🎉 벡터 기반 추천 시스템 실습 완료!")
        
        print("\n💡 학습 포인트:")
        print("  • 콘텐츠 기반 추천: 아이템 특성 유사도로 개인화")
        print("  • 협업 필터링: 유사한 사용자 패턴 기반 추천")
        print("  • 하이브리드 추천: 여러 방법론의 효과적 결합")
        print("  • 실시간 개인화: 현재 세션과 상황 정보 활용")
        print("  • 벡터 기반 유사도로 확장성 있는 추천 시스템 구현")
        
        print("\n🎉 3단계 모든 실습 완료!")
        print("🚀 다음 단계: 4단계 고급 기능 및 최적화")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 