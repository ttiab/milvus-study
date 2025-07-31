"""
 추천 시스템

 1. 데이터 수집 및 저장
 1-1. 컬렉션 생성
 1-2. 샘플 데이터 생성
 1-3. 데이터 삽입
 1-4. 인덱스 생성

 2. 추천 시스템 데모
 2-1. 콘텐츠 기반 추천 데모
 2-2. 협업 필터링 데모
 2-3. 하이브리드 추천 데모
 2-4. 실시간 개인화 추천 데모


"""
import time
from typing import List, Dict, Any

import numpy as np
from tutorial.common.vector_utils import VectorUtils
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema


def generate_sample_items(count: int = 1000) -> List[Dict[str, Any]]:
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


def generate_sample_users(count: int = 500) -> List[Dict[str, Any]]:
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


def generate_sample_interactions(users: List[Dict], items: List[Dict], count: int = 5000) -> List[Dict[str, Any]]:
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
            "view": (10, 300),  # 10초 ~ 5분
            "like": (5, 60),  # 5초 ~ 1분
            "add_to_cart": (30, 180),  # 30초 ~ 3분
            "purchase": (120, 600),  # 2분 ~ 10분
            "rating": (60, 300)  # 1분 ~ 5분
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


"""
 추천 시스템
"""
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

vector_utils = VectorUtils()

"""
[ 컬렉션 생성 ]
 - items
 - users
 - interactions
"""

# 상품
collection_name = "items"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

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

client.create_collection(collection_name=collection_name, schema=schema)

# 유저
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
    FieldSchema(name="behavior_vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # 행동 패턴 벡터
]

collection_name = "users"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

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
    FieldSchema(name="behavior_vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # 행동 패턴 벡터
]

schema = CollectionSchema(
    fields=fields,
    description="추천 시스템용 아이템 컬렉션",
    enable_dynamic_field=True
)

client.create_collection(collection_name=collection_name, schema=schema)

# 사용자-아이템 상호작용
collection_name = "interactions"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

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

client.create_collection(collection_name=collection_name, schema=schema)

"""
[ 샘플 데이터 생성 ]
1. 샘플 아이템 데이터 생성
2. 샘플 사용자 데이터 생성
3. 샘플 상호작용 데이터 생성
"""
items = generate_sample_items(1000)
users = generate_sample_users(500)
interactions = generate_sample_interactions(users, items, 5000)


"""
[ 데이터 삽입 ]
1. 아이템 데이터와 벡터 삽입
2. 사용자 데이터와 벡터 삽입
3. 상호작용 데이터와 벡터 삽입
"""

# 1. 아이템 데이터와 벡터 삽입
# 콘텐츠 벡터 생성 (제목 + 설명 + 태그)
print("  콘텐츠 벡터 생성 중...")
content_texts = []
for item in items:
    content_text = f"{item['title']} {item['description']} {item['tags']} {item['category']} {item['brand']}"
    content_texts.append(content_text)

content_vectors = vector_utils.texts_to_vectors(content_texts)

# 행동 기반 벡터 생성 (간단한 수치 특성 기반)
print("  행동 벡터 생성 중...")
behavior_vectors = []
for item in items:
    # 수치적 특징들을 정규화하여 128차원 벡터 생성
    features = [
        item['price'] / 1000,  # 가격 정규화
        item['rating'] / 5,  # 평점 정규화
        min(item['review_count'] / 100, 10) / 10,  # 리뷰수 정규화
        min(item['view_count'] / 1000, 10) / 10,  # 조회수 정규화
        min(item['purchase_count'] / 50, 10) / 10,  # 구매수 정규화
        item['popularity_score'] / 10  # 인기도 정규화
    ]

    # 128차원으로 확장 (패딩 + 노이즈)
    extended_features = features + [0] * (128 - len(features))
    extended_features = np.array(extended_features) + np.random.normal(0, 0.01, 128)
    behavior_vectors.append(extended_features)

behavior_vectors = np.array(behavior_vectors)

# 데이터를 딕셔너리 리스트 형태로 구성
data = []
for i, item in enumerate(items):
    data_item = {
        "item_id": item["item_id"],
        "title": item["title"],
        "description": item["description"],
        "category": item["category"],
        "subcategory": item["subcategory"],
        "brand": item["brand"],
        "tags": item["tags"],
        "price": item["price"],
        "rating": item["rating"],
        "review_count": item["review_count"],
        "view_count": item["view_count"],
        "purchase_count": item["purchase_count"],
        "release_date": item["release_date"],
        "availability": item["availability"],
        "popularity_score": item["popularity_score"],
        "content_vector": content_vectors[i].tolist(),
        "behavior_vector": behavior_vectors[i].tolist()
    }
    data.append(data_item)

collection_name = "items"
# 삽입 (딕셔너리 리스트 형태)
result = client.upsert(collection_name=collection_name, data=data)
total_inserted = len(items)
print(f"  ✅ {total_inserted}개 아이템 삽입 완료")

# 데이터 플러시
print("  데이터 플러시 중...")
client.flush(collection_name=collection_name)
print(f"  ✅ 총 {total_inserted}개 아이템 삽입 완료")

# 2. 사용자 데이터와 벡터 삽입
"""사용자 데이터와 벡터 삽입"""
print(f"\n💾 사용자 데이터 삽입 중...")

# 선호도 벡터 생성 (선호 카테고리 기반)
print("  사용자 선호도 벡터 생성 중...")
preference_texts = []
for user in users:
    pref_text = f"{user['preferred_categories']} {user['age_group']} {user['gender']} {user['price_range']}"
    preference_texts.append(pref_text)

preference_vectors = vector_utils.texts_to_vectors(preference_texts)

# 행동 벡터 생성 (구매 이력 기반)
print("  사용자 행동 벡터 생성 중...")
behavior_vectors = []
for user in users:
    features = [
        min(user['total_purchases'] / 10, 10) / 10,  # 구매수 정규화
        min(user['total_spent'] / 1000, 10) / 10,  # 지출액 정규화
        user['avg_rating_given'] / 5 if user['avg_rating_given'] > 0 else 0,  # 평점 정규화
        1 if user['is_premium'] else 0  # 프리미엄 여부
    ]

    # 128차원으로 확장
    extended_features = features + [0] * (128 - len(features))
    extended_features = np.array(extended_features) + np.random.normal(0, 0.01, 128)
    behavior_vectors.append(extended_features)

behavior_vectors = np.array(behavior_vectors)

# 데이터를 딕셔너리 리스트 형태로 구성
data = []
for i, user in enumerate(users):
    data_item = {
        "user_id": user["user_id"],
        "age_group": user["age_group"],
        "gender": user["gender"],
        "location": user["location"],
        "preferred_categories": user["preferred_categories"],
        "price_range": user["price_range"],
        "join_date": user["join_date"],
        "last_active": user["last_active"],
        "total_purchases": user["total_purchases"],
        "total_spent": user["total_spent"],
        "avg_rating_given": user["avg_rating_given"],
        "is_premium": user["is_premium"],
        "preference_vector": preference_vectors[i].tolist(),
        "behavior_vector": behavior_vectors[i].tolist()
    }
    data.append(data_item)

collection_name = "users"  # 컬렉션 이름 지정
# 삽입 (딕셔너리 리스트 형태 & 최신 MilvusClient API 사용)
result = client.upsert(collection_name=collection_name, data=data)
total_inserted = len(users)
print(f"  ✅ {total_inserted}개 사용자 삽입 완료")

# 데이터 플러시
print("  데이터 플러시 중...")
client.flush(collection_name=collection_name)
print(f"  ✅ 총 {total_inserted}개 사용자 삽입 완료")

# 3. 상호작용 데이터와 벡터 삽입
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

# 데이터를 딕셔너리 리스트 형태로 구성 (auto_id 필드 제외)
data = []
for i, interaction in enumerate(interactions):
    data_item = {
        "interaction_id": i,
        "user_id": interaction["user_id"],
        "item_id": interaction["item_id"],
        "interaction_type": interaction["interaction_type"],
        "rating": interaction["rating"],
        "timestamp": interaction["timestamp"],
        "session_id": interaction["session_id"],
        "device_type": interaction["device_type"],
        "duration": interaction["duration"],
        "context_vector": context_vectors[i].tolist()
    }
    data.append(data_item)

collection_name = "interactions"  # 컬렉션 이름 지정
# 삽입 (딕셔너리 리스트 형태 & 최신 MilvusClient API 사용)
result = client.upsert(collection_name=collection_name, data=data)
total_inserted = len(interactions)
print(f"  ✅ {total_inserted}개 상호작용 삽입 완료")

# 데이터 플러시
print("  데이터 플러시 중...")
client.flush(collection_name=collection_name)
print(f"  ✅ 총 {total_inserted}개 상호작용 삽입 완료")


"""

[ 인덱스 생성 ]

"""

# rec_system.create_indexes(item_collection, ["content_vector", "behavior_vector"])
# rec_system.create_indexes(user_collection, ["preference_vector", "behavior_vector"])
# rec_system.create_indexes(interaction_collection, ["context_vector"])

collection_name = "items"
vector_fields = ["content_vector", "behavior_vector"]

index_params = MilvusClient.prepare_index_params()

for field_name in vector_fields:
    index_params.add_index(
        index_type="HNSW",  # Name of the vector field to be indexed
        field_name=field_name,  # Type of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "M": 16,
            "efConstruction": 200
        }  # Index building params
    )
    print(f"  {field_name} 인덱스 생성 중...")
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    print(f"  ✅ {field_name} 인덱스 생성 완료")


collection_name = "users"
vector_fields = ["preference_vector", "behavior_vector"]

index_params = MilvusClient.prepare_index_params()

for field_name in vector_fields:
    index_params.add_index(
        index_type="HNSW",  # Name of the vector field to be indexed
        field_name=field_name,  # Type of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "M": 16,
            "efConstruction": 200
        }  # Index building params
    )
    print(f"  {field_name} 인덱스 생성 중...")
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    print(f"  ✅ {field_name} 인덱스 생성 완료")

collection_name = "interactions"
vector_fields = ["context_vector"]

index_params = MilvusClient.prepare_index_params()

for field_name in vector_fields:
    index_params.add_index(
        index_type="HNSW",  # Name of the vector field to be indexed
        field_name=field_name,  # Type of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "M": 16,
            "efConstruction": 200
        }  # Index building params
    )
    print(f"  {field_name} 인덱스 생성 중...")
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    print(f"  ✅ {field_name} 인덱스 생성 완료")


"""
[ 추천 서비스 데모 ]
 - 콘텐츠 기반 추천 데모
 - 협업 필터링 데모
 - 하이브리드 추천 데모
 - 실시간 개인화 추천 데모
 
"""
"""콘텐츠 기반 추천 데모"""
print("\n" + "=" * 80)
print(" 📚 콘텐츠 기반 추천 시스템 데모")
print("=" * 80)

client.load_collection(collection_name="items")

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
    query_vectors = vector_utils.text_to_vector(interest['query'])
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors

    # 콘텐츠 기반 유사도 검색
    start_time = time.time()
    results = client.search(
        collection_name="items",
        anns_field="content_vector",
        data=[query_vector.tolist()],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 200}},
        output_fields=["title", "category", "brand", "price", "rating", "description"]
    )
    search_time = time.time() - start_time
    print(f"   검색 시간: {search_time:.4f}초")
    print(f"   추천 아이템 수: {len(results[0])}")

    for j, hit in enumerate(results[0]):
        similarity = 1 - hit['distance']
        entity = hit['entity']
        print(f"     {j + 1}. {entity.get('title')}")
        print(f"        카테고리: {entity.get('category')}, 브랜드: {entity.get('brand')}")
        print(f"        가격: ${entity.get('price'):.2f}, 평점: {entity.get('rating')}")
        print(f"        설명: {entity.get('description')[:80]}...")
        print(f"        유사도: {similarity:.3f}")

    # 카테고리 필터링 추천
    if interest.get('category_filter'):
        print(f"\n   📁 카테고리 필터링 추천 ({interest['category_filter']})")

        category_filter = interest['category_filter'].replace("'", "\\'")
        expr = f"category == '{category_filter}' and availability == True"

        category_results = client.search(
            collection_name="items",
            data=[query_vector.tolist()],
            anns_field="content_vector",
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 200},
                "expr": expr  # ✅ 여기로 이동
            },
            limit=3,
            output_fields=["title", "brand", "price", "rating", "popularity_score"]
        )

        print(f"   카테고리 내 추천 수: {len(category_results[0])}")
        for j, hit in enumerate(category_results[0]):
            similarity = 1 - hit['distance']
            entity = hit['entity']
            print(f"     {j + 1}. {entity.get('title')}")
            print(f"        브랜드: {entity.get('brand')}, 가격: ${entity.get('price'):.2f}")
            print(f"        평점: {entity.get('rating')}, 인기도: {entity.get('popularity_score'):.2f}")
            print(f"        유사도: {similarity:.3f}")


