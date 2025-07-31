import time
from typing import List

import numpy as np
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from tutorial.common.vector_utils import VectorUtils

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

vectorUtils = VectorUtils()


"""
컬렉션 생성
"""
collection_name = "partition_demo"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="created_date", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="priority", dtype=DataType.INT64),
    FieldSchema(name="score", dtype=DataType.FLOAT)
]

schema = CollectionSchema(
    fields=fields,
    description="파티션 관리 데모 컬렉션"
)


client.create_collection(collection_name=collection_name, schema=schema)

"""
파티션 생성
"""

# 1. 카테고리별 파티션
category_partitions = ["tech", "business", "science", "health"]

# 2. 지역별 파티션
region_partitions = ["asia", "europe", "america"]

# 3. 시간별 파티션
time_partitions = ["2023", "2024", "2025"]

all_partitions = {}

# 카테고리 파티션 생성
for category in category_partitions:
    partition_name = f"category_{category}"
    client.create_partition(collection_name=collection_name, partition_name=partition_name)
    all_partitions[partition_name] = f"카테고리: {category}"
    print(f"  ✅ 파티션 '{partition_name}' 생성됨")

# 지역 파티션 생성
for region in region_partitions:
    partition_name = f"region_{region}"
    client.create_partition(collection_name=collection_name, partition_name=partition_name)
    all_partitions[partition_name] = f"지역: {region}"
    print(f"  ✅ 파티션 '{partition_name}' 생성됨")

# 시간 파티션 생성
for year in time_partitions:
    partition_name = f"year_{year}"
    client.create_partition(collection_name=collection_name, partition_name=partition_name)
    all_partitions[partition_name] = f"년도: {year}"
    print(f"  ✅ 파티션 '{partition_name}' 생성됨")

"""
파티션 조회
"""

partitions = client.list_partitions(collection_name=collection_name)
print(f"파티션 총 수 {len(partitions)}")
print(partitions)
for i, partition in enumerate(partitions):
    print(f"  {i + 1}. {partition}")
    if hasattr(partition, 'description') and partition.description:
        print(f"      설명: {partition.description}")

"""
파티션별 데이터 삽입
"""


def generate_partitioned_data(partition_type: str, partition_value: str, count: int = 1000) -> List[dict]:
    """파티션별 데이터 생성"""
    print(f"  📊 {partition_type}={partition_value} 데이터 {count}개 생성 중...")

    titles = []
    contents = []
    categories = []
    regions = []
    dates = []
    priorities = []
    scores = []

    # 파티션 타입에 따른 데이터 생성
    for i in range(count):
        if partition_type == "category":
            category = partition_value
            region = np.random.choice(["asia", "europe", "america"])
            year = np.random.choice(["2023", "2024", "2025"])
        elif partition_type == "region":
            category = np.random.choice(["tech", "business", "science", "health"])
            region = partition_value
            year = np.random.choice(["2023", "2024", "2025"])
        elif partition_type == "year":
            category = np.random.choice(["tech", "business", "science", "health"])
            region = np.random.choice(["asia", "europe", "america"])
            year = partition_value
        else:
            category = partition_value
            region = "asia"
            year = "2024"

        title = f"{category.title()} Article {i} from {region} in {year}"
        content = f"This is a detailed article about {category} topics, " \
                  f"published in {region} region during {year}. " \
                  f"It contains valuable insights and comprehensive analysis."

        titles.append(title)
        contents.append(content)
        categories.append(category)
        regions.append(region)
        dates.append(f"{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}")
        priorities.append(np.random.randint(1, 6))
        scores.append(np.random.uniform(1.0, 5.0))

    # 벡터 변환
    combined_texts = [f"{title} {content}" for title, content in zip(titles, contents)]
    vectors = vectorUtils.texts_to_vectors(combined_texts)

    # 딕셔너리 리스트로 데이터 구성
    data = []
    for i in range(count):
        data_item = {
            "id": i,
            "title": titles[i],
            "content": contents[i],
            "vector": vectors[i].tolist(),
            "category": categories[i],
            "region": regions[i],
            "created_date": dates[i],
            "priority": priorities[i],
            "score": scores[i]
        }
        data.append(data_item)

    return data

# 카테고리별 데이터 삽입
categories = ["tech", "business", "science", "health"]
for category in categories:
    partition_name = f"category_{category}"
    data = generate_partitioned_data("category", category, 1000)
    insert_result = client.upsert(collection_name=collection_name, partition_name=partition_name, data=data)
    print(f"  ✅ 파티션 '{partition_name}': {insert_result}개 삽입")

# 지역별 데이터 삽입
regions = ["asia", "europe", "america"]
for region in regions:
    partition_name = f"region_{region}"
    data = generate_partitioned_data("region", region, 1500)

    insert_result = client.upsert(collection_name=collection_name, partition_name=partition_name, data=data)
    print(f"  ✅ 파티션 '{partition_name}': {insert_result}개 삽입")

# 시간별 데이터 삽입
years = ["2023", "2024", "2025"]
for year in years:
    partition_name = f"year_{year}"
    data = generate_partitioned_data("year", year, 1000)

    insert_result = client.upsert(collection_name=collection_name, partition_name=partition_name, data=data)
    print(f"  ✅ 파티션 '{partition_name}': {insert_result}개 삽입")

# 데이터 플러시
client.flush(collection_name=collection_name)
print(f"  ✅ 모든 파티션 데이터 삽입 완료")

"""
인덱스 생성
"""
index_params = client.prepare_index_params(collection_name=collection_name)

index_params.add_index(
    field_name="vector", # Name of the vector field to be indexed
    index_type="HNSW", # Type of the index to create
    index_name="vector_index", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "M": 16, # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 200 # Number of candidate neighbors considered for connection during index construction
    } # Index building params
)

client.create_index(collection_name=collection_name, index_params=index_params)

print(client.list_indexes(collection_name=collection_name))

"""
파티션 검색
"""

client.load_collection(collection_name=collection_name)
query_text = "advanced technology artificial intelligence innovation"
query_vectors = vectorUtils.text_to_vector(query_text)
query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors

print(f"검색 쿼리: '{query_text}'")

print("\n1. 전체 컬렉션 검색")
start_time = time.time()

results = client.search(
    collection_name=collection_name,  # Collection name
    anns_field="vector",
    data=[query_vector],  # Query vector
    limit=10,  # TopK results to return
    search_params={"metric_type": "L2", "params": {"ef": 100}},
    output_fields=["title", "category", "region", "created_date"]
)

search_time = time.time() - start_time
print(f"검색 시간: {search_time:.4f}초")
print(f"결과 수: {len(results[0])}")


for j, hit in enumerate(results[0]):
    print(f"    {j + 1}. {hit['entity']['title']}")
    print(f"        카테고리: {hit['entity']['category']}")
    print(f"        지역: {hit['entity']['region']}")
    print(f"        날짜: {hit['entity']['created_date']}")

print("\n2. 특정 파티션 검색 (category_business)")
start_time = time.time()

results = client.search(
    collection_name=collection_name,  # Collection name
    partition_names=["category_business"],
    anns_field="vector",
    data=[query_vector],  # Query vector
    limit=10,  # TopK results to return
    search_params={"metric_type": "L2", "params": {"ef": 100}},
    output_fields=["title", "category", "region", "created_date"]
)

search_time = time.time() - start_time
print(f"검색 시간: {search_time:.4f}초")
print(f"결과 수: {len(results[0])}")

for j, hit in enumerate(results[0]):
    print(f"    {j + 1}. {hit['entity']['title']}")
    print(f"        카테고리: {hit['entity']['category']}")
    print(f"        지역: {hit['entity']['region']}")
    print(f"        날짜: {hit['entity']['created_date']}")

print("\n3. 다중 파티션 검색 (region_asia, region_europe)")
start_time = time.time()

results = client.search(
    collection_name=collection_name,  # Collection name
    partition_names=["region_asia", "region_europe"],
    anns_field="vector",
    data=[query_vector],  # Query vector
    limit=10,  # TopK results to return
    search_params={"metric_type": "L2", "params": {"ef": 100}},
    output_fields=["title", "category", "region", "created_date"]
)

search_time = time.time() - start_time
print(f"검색 시간: {search_time:.4f}초")
print(f"결과 수: {len(results[0])}")

for j, hit in enumerate(results[0]):
    print(f"    {j + 1}. {hit['entity']['title']}")
    print(f"        카테고리: {hit['entity']['category']}")
    print(f"        지역: {hit['entity']['region']}")
    print(f"        날짜: {hit['entity']['created_date']}")






"""
파티션 성능 비교
"""

query_text = "business strategy management"
query_vectors = vectorUtils.text_to_vector(query_text)
query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors

test_cases = [
    {
        "name": "전체 컬렉션",
        "partitions": None,
        "description": "모든 파티션에서 검색"
    },
    {
        "name": "단일 파티션",
        "partitions": ["category_business"],
        "description": "비즈니스 카테고리만"
    },
    {
        "name": "지역 파티션",
        "partitions": ["region_asia"],
        "description": "아시아 지역만"
    },
    {
        "name": "시간 파티션",
        "partitions": ["year_2024"],
        "description": "2024년만"
    },
    {
        "name": "다중 파티션",
        "partitions": ["category_business", "category_tech"],
        "description": "비즈니스 + 기술"
    }
]

print(f"{'테스트 케이스':<15} {'평균시간(초)':<12} {'QPS':<8} {'결과수':<6} {'설명':<20}")
print("-" * 80)

for case in test_cases:
    times = []
    result_count = 0

    # 여러 번 측정하여 평균 계산
    for _ in range(3):
        start_time = time.time()

        if case["partitions"]:
            results = client.search(
                collection_name=collection_name,  # Collection name
                partition_names=case["partitions"],
                anns_field="vector",
                data=[query_vector],  # Query vector
                limit=10,  # TopK results to return
                search_params={"metric_type": "L2", "params": {"ef": 100}},
                output_fields=["title"]
            )
        else:
            results = client.search(
                collection_name=collection_name,  # Collection name
                anns_field="vector",
                data=[query_vector],  # Query vector
                limit=10,  # TopK results to return
                search_params={"metric_type": "L2", "params": {"ef": 100}},
                output_fields=["title"]
            )

        search_time = time.time() - start_time
        times.append(search_time)
        result_count = len(results[0])

    avg_time = np.mean(times)
    qps = 1 / avg_time

    print(f"{case['name']:<15} {avg_time:<12.4f} {qps:<8.2f} {result_count:<6} {case['description']:<20}")



