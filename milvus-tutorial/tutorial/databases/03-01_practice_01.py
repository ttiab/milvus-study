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
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
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

# 제목과 내용을 결합하여 벡터 변환
combined_texts = [f"{article['title']} {article['content']}" for article in articles]
print(combined_texts)
vectors = vector_utils.texts_to_vectors(combined_texts)
time.sleep(10)

print(vectors)
print(len(vectors))

i = 0
for article in articles:
    article["vector"] = vectors[i]
    i = i+1

print(f"  ✅ {len(articles)}개 문서 데이터 준비 완료")
print(f"  ✅ {articles[0]['vector']}")
print(f"  📏 벡터 차원: {vectors.shape[1]}")

start_time = time.time()

insert = client.insert(collection_name=collection_name, data=articles)
client.flush(collection_name=collection_name)

print(insert)

