"""
1. 샘플 컬렉션 생성
2. 샘플 데이터 준비
3. 데이터 삽입
4. 인덱스 생성
5. 기본 검색
6. 필터링 검색
"""
import time

from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection

from tutorial.common.vector_utils import VectorUtils

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 컬렉션 정의
collection_name = "practice_articles"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

# 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="score", dtype=DataType.FLOAT)
]

schema = CollectionSchema(
    fields=fields,
    description="샘플 문서 컬렉션"
)

# 컬렉션 생성
client.create_collection(collection_name=collection_name, schema=schema)

# 샘플 데이터
articles = [
    {
        "title": "인공지능의 미래와 기계학습",
        "content": "인공지능과 기계학습은 현대 기술의 핵심입니다. 딥러닝, 자연어 처리, 컴퓨터 비전 등 다양한 분야에서 혁신을 이끌고 있습니다.",
        "category": "Technology",
        "author": "김AI",
        "score": 4.8
    },
    {
        "title": "클라우드 컴퓨팅과 데이터 분석",
        "content": "클라우드 컴퓨팅은 기업의 디지털 전환을 가속화하고 있습니다. 빅데이터 분석, 실시간 처리, 확장성 등의 장점을 제공합니다.",
        "category": "Technology",
        "author": "박클라우드",
        "score": 4.5
    },
    {
        "title": "지속가능한 비즈니스 전략",
        "content": "ESG 경영과 지속가능성은 현대 기업의 필수 요소입니다. 환경, 사회, 지배구조 측면에서 균형잡힌 성장을 추구해야 합니다.",
        "category": "Business",
        "author": "이비즈",
        "score": 4.2
    },
    {
        "title": "양자컴퓨팅의 원리와 응용",
        "content": "양자컴퓨팅은 기존 컴퓨터의 한계를 뛰어넘는 혁신적인 기술입니다. 암호화, 최적화, 시뮬레이션 분야에서 활용됩니다.",
        "category": "Science",
        "author": "정양자",
        "score": 4.9
    },
    {
        "title": "스타트업 생태계와 투자 트렌드",
        "content": "스타트업 생태계는 혁신의 원동력입니다. 벤처캐피털, 액셀러레이터, 인큐베이터가 생태계를 지원하고 있습니다.",
        "category": "Business",
        "author": "최스타트",
        "score": 4.1
    },
    {
        "title": "생명과학과 바이오테크놀로지",
        "content": "생명과학 기술은 의료, 농업, 환경 분야에 혁신을 가져오고 있습니다. 유전자 편집, 세포 치료 등이 주목받고 있습니다.",
        "category": "Science",
        "author": "김바이오",
        "score": 4.6
    },
    {
        "title": "디지털 마케팅과 고객 경험",
        "content": "디지털 마케팅은 고객과의 접점을 다양화하고 개인화된 경험을 제공합니다. 데이터 분석을 통한 인사이트 도출이 핵심입니다.",
        "category": "Business",
        "author": "오마케팅",
        "score": 4.0
    },
    {
        "title": "로봇공학과 자동화 기술",
        "content": "로봇공학은 제조업부터 서비스업까지 다양한 분야에서 자동화를 실현하고 있습니다. AI와 결합하여 더욱 지능적으로 발전하고 있습니다.",
        "category": "Technology",
        "author": "한로봇",
        "score": 4.4
    }
]

# 벡터 변환을 위한 유틸리티 초기화
vector_utils = VectorUtils()
vector_utils.load_text_model("jhgan/ko-sroberta-multitask")


# 제목과 내용을 결합하여 벡터 변환
combined_texts = [f"{article['title']} {article['content']}" for article in articles]
print(combined_texts)
vectors = vector_utils.texts_to_vectors(combined_texts, normalize=True)
time.sleep(10)

print(vectors)
print(len(vectors))

i = 0
for article in articles:
    article["vector"] = vectors[i]
    i = i + 1

print(f"  ✅ {len(articles)}개 문서 데이터 준비 완료")
print(f"  ✅ {articles[0]['vector']}")
print(f"  📏 벡터 차원: {vectors.shape[1]}")

start_time = time.time()

insert = client.insert(collection_name=collection_name, data=articles)
client.flush(collection_name=collection_name)

print(insert)

"""
인덱스 생성
"""

"""인덱스 생성 데모"""

# Prepare index building params
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    index_type="IVF_FLAT",  # Name of the vector field to be indexed
    field_name="vector",  # Type of the index to create
    index_name="vector_index",  # Name of the index to create
    metric_type="COSINE",  # Metric type used to measure similarity
    params={
        "nlist": 128,  # Number of clusters for the index
    }  # Index building params
)


client.create_index(
    collection_name=collection_name,
    index_params=index_params,
)


print(client.list_indexes(collection_name=collection_name))


"""
기본 검색
"""

client.load_collection(collection_name=collection_name)

# 검색 쿼리들
queries = [
    "인공지능과 기계학습 기술",
    "비즈니스 전략과 경영",
    "과학 기술과 연구",
    "클라우드 컴퓨팅과 데이터"
]

for i, query_text in enumerate(queries):
    print(f"\n{i + 2}. 검색 쿼리: '{query_text}'")

    # 쿼리 벡터 생성
    query_vectors = vector_utils.text_to_vector(query_text)
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors

    search_params = {
        "params": {
            "nprobe": 10,  # Number of clusters to search
        }
    }

    results = client.search(
        collection_name=collection_name,  # Collection name
        anns_field="vector",
        data=[query_vector],  # Query vector
        limit=3,  # TopK results to return
        search_params=search_params,
        output_fields = ["title", "category", "author", "score"]

    )

    # 검색 실행
    start_time = time.time()


    # results = collection.search(
    #     data=[query_vector],
    #     anns_field="vector",
    #     param=search_params,
    #     limit=3,
    #     output_fields=["title", "category", "author", "score"]
    # )

    search_time = time.time() - start_time
    print(f"  검색 시간: {search_time:.4f}초")
    print(f"  검색 결과 수: {len(results[0])}")

    print(results)
    # 결과 출력
    for j, hit in enumerate(results[0]):
        print(f"    {j + 1}. {hit['entity']['title']}")
        print(f"        카테고리: {hit['entity']['category']}")
        print(f"        저자: {hit['entity']['author']}")
        print(f"        점수: {hit['entity']['score']}")
        print(f"        유사도 거리: {hit['distance']:.4f}")

"""
필터링 검색
"""

query_text = "최신 기술 동향"
query_vectors = vector_utils.text_to_vector(query_text)
query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
search_params = {
    "params": {
        "nprobe": 10,  # Number of clusters to search
    }
}
results = client.search(
    collection_name=collection_name,  # Collection name
    anns_field="vector",
    data=[query_vector],  # Query vector
    limit=3,  # TopK results to return
    search_params=search_params,
    filter='category like "Business" ',
    output_fields=["title", "category", "author", "score"]

)
for hits in results:
    print("TopK results:")
    for hit in hits:
        print(hit)
