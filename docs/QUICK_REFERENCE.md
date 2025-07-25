# 🚀 Milvus 빠른 참조 가이드 (Quick Reference)

## 📋 목차

1. [필수 명령어](#필수-명령어)
2. [연결 및 기본 설정](#연결-및-기본-설정)
3. [스키마 및 컬렉션](#스키마-및-컬렉션)
4. [데이터 조작](#데이터-조작)
5. [인덱스 관리](#인덱스-관리)
6. [검색 및 쿼리](#검색-및-쿼리)
7. [성능 최적화](#성능-최적화)
8. [트러블슈팅](#트러블슈팅)

---

## ⚡ 필수 명령어

### 프로젝트 실행
```bash
# 환경 활성화
source venv/bin/activate

# Milvus 서비스 시작
docker-compose up -d

# 서비스 상태 확인
docker-compose ps

# 단계별 실습 실행
python step01_basics/01_environment_setup.py
python step02_core_features/01_index_management.py
python step03_use_cases/01_text_similarity_search.py
python step04_advanced/01_performance_optimization.py
python step05_production/01_kubernetes_deployment.py
```

### 환경 관리
```bash
# 서비스 중지
docker-compose down

# 데이터 초기화
docker-compose down -v

# 로그 확인
docker-compose logs milvus

# 컨테이너 상태 확인
docker ps | grep milvus
```

---

## 🔗 연결 및 기본 설정

### 기본 연결
```python
from pymilvus import connections, Collection

# 연결 설정
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    timeout=60
)

# 연결 상태 확인
print(connections.list_connections())
```

### 연결 풀링 (프로덕션용)
```python
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    pool_size=10,
    timeout=60
)
```

### 컨텍스트 매니저 사용
```python
from common.connection import MilvusConnection

with MilvusConnection() as conn:
    # 여기서 Milvus 작업 수행
    collection = conn.get_collection("my_collection")
```

---

## 📄 스키마 및 컬렉션

### 기본 스키마 정의
```python
from pymilvus import FieldSchema, CollectionSchema, DataType

# 필드 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="score", dtype=DataType.FLOAT),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]

# 스키마 생성
schema = CollectionSchema(
    fields=fields,
    description="Text similarity search collection"
)

# 컬렉션 생성
collection = Collection(name="text_search", schema=schema)
```

### 고급 스키마 (동적 필드)
```python
schema = CollectionSchema(
    fields=fields,
    description="Advanced collection with dynamic fields",
    enable_dynamic_field=True  # 동적 필드 활성화
)
```

### 컬렉션 관리
```python
# 컬렉션 존재 확인
from pymilvus import utility
if utility.has_collection("text_search"):
    print("컬렉션 존재")

# 컬렉션 삭제
utility.drop_collection("text_search")

# 모든 컬렉션 목록
collections = utility.list_collections()
print(f"컬렉션: {collections}")

# 컬렉션 정보
collection_info = collection.describe()
print(collection_info)
```

---

## 💾 데이터 조작

### 데이터 삽입 (List[List] 방식 - 권장)
```python
# 5,000개 데이터 삽입 예시
texts = ["document text 1", "document text 2", ...]
categories = ["tech", "business", ...]
scores = [0.95, 0.87, ...]
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]

# List[List] 형태로 구성 (필드 순서대로)
data = [
    texts,      # text 필드
    categories, # category 필드  
    scores,     # score 필드
    embeddings  # embedding 필드
]

# 삽입 실행
result = collection.insert(data)
collection.flush()  # 메모리 → 디스크
print(f"삽입된 ID: {result.primary_keys}")
```

### 배치 삽입
```python
batch_size = 1000
for i in range(0, len(all_data), batch_size):
    batch_data = all_data[i:i+batch_size]
    collection.insert(batch_data)
    
    if i % (batch_size * 10) == 0:  # 10배치마다 플러시
        collection.flush()

# 최종 플러시
collection.flush()
```

### 파티션 사용
```python
# 파티션 생성
collection.create_partition("2024_Q1")
collection.create_partition("technology")

# 파티션에 데이터 삽입
collection.insert(data, partition_name="technology")

# 파티션 목록 확인
partitions = collection.partitions
for p in partitions:
    print(f"파티션: {p.name}")
```

---

## 🔧 인덱스 관리

### HNSW 인덱스 (고성능)
```python
hnsw_index = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {
        "M": 16,               # 연결 수 (8-64)
        "efConstruction": 256  # 구축 시 탐색 (64-512)
    }
}

collection.create_index(
    field_name="embedding",
    index_params=hnsw_index
)
```

### IVF 계열 인덱스
```python
# IVF_FLAT (균형적)
ivf_flat_index = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}  # 클러스터 수
}

# IVF_SQ8 (메모리 절약)
ivf_sq8_index = {
    "metric_type": "L2", 
    "index_type": "IVF_SQ8",
    "params": {"nlist": 1024}
}

# IVF_PQ (대용량)
ivf_pq_index = {
    "metric_type": "L2",
    "index_type": "IVF_PQ", 
    "params": {
        "nlist": 1024,
        "m": 8,      # 부벡터 수
        "nbits": 8   # 양자화 비트
    }
}
```

### 인덱스 관리
```python
# 기존 인덱스 삭제
collection.drop_index(field_name="embedding")

# 인덱스 정보 확인
indexes = collection.indexes
for idx in indexes:
    print(f"인덱스: {idx.field_name}, 타입: {idx.index_type}")

# 컬렉션 로드 (검색 전 필수)
collection.load()

# 컬렉션 해제 (메모리 절약)
collection.release()
```

---

## 🔍 검색 및 쿼리

### 기본 벡터 검색
```python
# 검색 파라미터
search_params = {
    "metric_type": "L2",
    "params": {
        "ef": 128      # HNSW용 (32-512)
        # "nprobe": 64 # IVF용 (1-nlist)
    }
}

# 단일 검색
query_vector = [[0.1, 0.2, 0.3, ...]]  # 2D 배열
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=["text", "category", "score"]
)

# 결과 출력
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 거리: {hit.distance:.4f}")
        print(f"텍스트: {hit.entity.get('text')}")
```

### 필터링 검색 (하이브리드)
```python
# 스칼라 필터 + 벡터 검색
results = collection.search(
    data=query_vector,
    anns_field="embedding", 
    param=search_params,
    limit=10,
    expr="category == 'technology' and score >= 0.8",  # 필터 조건
    output_fields=["text", "category", "score"]
)
```

### 복잡한 필터 조건
```python
# 여러 조건 조합
filter_expressions = [
    "category in ['tech', 'ai', 'ml']",
    "score >= 0.7 and score <= 1.0", 
    "text like '%artificial%'",
    "id in [1, 2, 3, 4, 5]",
    "category == 'tech' and (score > 0.8 or id < 1000)"
]

for expr in filter_expressions:
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        expr=expr,
        limit=5
    )
```

### 파티션별 검색
```python
# 특정 파티션에서만 검색
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    partition_names=["technology", "ai"],  # 특정 파티션
    limit=10
)
```

### 배치 검색
```python
# 여러 쿼리 동시 검색
query_vectors = [
    [0.1, 0.2, ...],
    [0.3, 0.4, ...],
    [0.5, 0.6, ...]
]

results = collection.search(
    data=query_vectors,
    anns_field="embedding",
    param=search_params,
    limit=10
)

# 각 쿼리별 결과 처리
for i, hits in enumerate(results):
    print(f"Query {i+1} results:")
    for hit in hits:
        print(f"  ID: {hit.id}, Distance: {hit.distance:.4f}")
```

---

## ⚡ 성능 최적화

### 메모리 관리
```python
# 컬렉션 로드 (검색 전)
collection.load()

# 특정 파티션만 로드
collection.load(partition_names=["2024_Q1"])

# 메모리 해제
collection.release()

# 특정 파티션 해제
collection.release(partition_names=["old_partition"])
```

### 배치 처리 최적화
```python
# 최적 배치 크기
OPTIMAL_BATCH_SIZE = 1000

def optimized_insert(collection, all_data):
    """최적화된 배치 삽입"""
    total_inserted = 0
    
    for i in range(0, len(all_data), OPTIMAL_BATCH_SIZE):
        batch = all_data[i:i+OPTIMAL_BATCH_SIZE]
        result = collection.insert(batch)
        total_inserted += len(result.primary_keys)
        
        # 주기적 플러시
        if i % (OPTIMAL_BATCH_SIZE * 10) == 0:
            collection.flush()
            print(f"Progress: {total_inserted} inserted")
    
    # 최종 플러시
    collection.flush()
    return total_inserted
```

### 검색 성능 튜닝
```python
# 인덱스별 최적 파라미터
optimal_params = {
    "HNSW": {
        "build": {"M": 16, "efConstruction": 256},
        "search": {"ef": 128}
    },
    "IVF_FLAT": {
        "build": {"nlist": 1024},
        "search": {"nprobe": 64}
    },
    "IVF_SQ8": {
        "build": {"nlist": 4096},
        "search": {"nprobe": 128}
    }
}

# 동적 파라미터 조정
def auto_tune_search_params(data_size):
    if data_size < 100_000:
        return {"ef": 64}   # 소규모
    elif data_size < 1_000_000:
        return {"ef": 128}  # 중규모
    else:
        return {"ef": 256}  # 대규모
```

---

## 🛠️ 트러블슈팅

### 연결 문제
```python
def check_milvus_connection():
    """Milvus 연결 상태 확인"""
    try:
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus 연결 성공")
        return True
    except Exception as e:
        print(f"❌ Milvus 연결 실패: {e}")
        return False

def diagnose_connection():
    """연결 문제 진단"""
    import subprocess
    
    # Docker 컨테이너 상태 확인
    result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
    if 'milvus' in result.stdout:
        print("✅ Milvus 컨테이너 실행 중")
    else:
        print("❌ Milvus 컨테이너 중지됨")
        print("해결책: docker-compose up -d")
```

### 데이터 삽입 문제
```python
def safe_insert_with_validation(collection, data):
    """검증을 포함한 안전한 데이터 삽입"""
    try:
        # 1. 데이터 형식 검증
        if not isinstance(data, list):
            raise ValueError("데이터는 List 형태여야 합니다")
        
        # 2. 스키마 호환성 확인
        schema_fields = [f.name for f in collection.schema.fields if not f.is_primary or not f.auto_id]
        if len(data) != len(schema_fields):
            raise ValueError(f"필드 수 불일치: 예상 {len(schema_fields)}, 실제 {len(data)}")
        
        # 3. 데이터 길이 확인
        first_field_len = len(data[0])
        for i, field_data in enumerate(data):
            if len(field_data) != first_field_len:
                raise ValueError(f"필드 {i}의 데이터 길이가 다릅니다")
        
        # 4. 삽입 실행
        result = collection.insert(data)
        collection.flush()
        
        print(f"✅ {len(result.primary_keys)}개 데이터 삽입 완료")
        return result
        
    except Exception as e:
        print(f"❌ 삽입 실패: {e}")
        print(f"데이터 형태: {type(data)}")
        if isinstance(data, list) and data:
            print(f"첫 번째 필드 길이: {len(data[0])}")
            print(f"총 필드 수: {len(data)}")
        raise
```

### 인덱스 문제
```python
def fix_index_issues(collection):
    """인덱스 관련 문제 해결"""
    try:
        # 기존 인덱스 확인
        indexes = collection.indexes
        if indexes:
            print("기존 인덱스 발견, 삭제 중...")
            for idx in indexes:
                collection.drop_index(field_name=idx.field_name)
        
        # 새 인덱스 생성
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW", 
            "params": {"M": 16, "efConstruction": 256}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print("✅ 인덱스 재생성 완료")
        
    except Exception as e:
        print(f"❌ 인덱스 문제 해결 실패: {e}")
```

### 성능 문제 진단
```python
def diagnose_performance(collection):
    """성능 문제 진단"""
    import time
    
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    # 1. 컬렉션 통계
    try:
        stats = collection.get_stats()
        report["stats"] = stats
    except:
        report["stats"] = "조회 실패"
    
    # 2. 인덱스 상태
    try:
        indexes = collection.indexes
        report["indexes"] = [{"field": idx.field_name, "type": idx.index_type} for idx in indexes]
    except:
        report["indexes"] = "조회 실패"
    
    # 3. 간단한 성능 테스트
    try:
        dummy_vector = [[0.1] * 384]
        start_time = time.time()
        
        collection.search(
            data=dummy_vector,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": 64}},
            limit=1
        )
        
        latency = (time.time() - start_time) * 1000
        report["search_latency_ms"] = round(latency, 2)
        report["performance_status"] = "정상" if latency < 100 else "느림"
        
    except Exception as e:
        report["search_latency_ms"] = "측정 실패"
        report["error"] = str(e)
    
    return report
```

### 로그 분석
```bash
# Milvus 컨테이너 로그 확인
docker logs milvus-standalone

# 실시간 로그 모니터링
docker logs -f milvus-standalone

# 최근 100줄만 확인
docker logs --tail 100 milvus-standalone

# 특정 시간 이후 로그
docker logs --since "2024-01-01T10:00:00" milvus-standalone
```

---

## 📊 모니터링 및 메트릭

### 컬렉션 정보 확인
```python
def get_collection_info(collection):
    """컬렉션 종합 정보"""
    info = {
        "name": collection.name,
        "description": collection.description,
        "schema": collection.schema,
        "stats": collection.get_stats(),
        "indexes": [(idx.field_name, idx.index_type) for idx in collection.indexes],
        "partitions": [p.name for p in collection.partitions]
    }
    return info

# 사용 예시
info = get_collection_info(collection)
print(f"컬렉션 이름: {info['name']}")
print(f"데이터 수: {info['stats']}")
```

### 시스템 상태 모니터링
```python
def system_health_check():
    """시스템 전체 상태 확인"""
    from pymilvus import utility
    
    health = {
        "connections": len(connections.list_connections()),
        "collections": len(utility.list_collections()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 각 컬렉션 상태 확인
    collection_status = {}
    for coll_name in utility.list_collections():
        try:
            coll = Collection(coll_name)
            collection_status[coll_name] = {
                "loaded": "로드됨" if coll.has_index() else "언로드됨",
                "indexes": len(coll.indexes),
                "partitions": len(coll.partitions)
            }
        except:
            collection_status[coll_name] = "오류"
    
    health["collections_detail"] = collection_status
    return health
```

---

## 🚀 실용적인 코드 스니펫

### 벡터 유사도 검색 시스템
```python
class VectorSearchSystem:
    def __init__(self, collection_name):
        self.collection = Collection(collection_name)
        self.vector_utils = VectorUtils()
    
    def add_documents(self, texts, categories=None):
        """문서 추가"""
        vectors = self.vector_utils.text_to_vector(texts)
        
        if categories is None:
            categories = ["general"] * len(texts)
        
        data = [texts, categories, vectors.tolist()]
        result = self.collection.insert(data)
        self.collection.flush()
        return result.primary_keys
    
    def search(self, query_text, top_k=10, category_filter=None):
        """텍스트 검색"""
        query_vector = self.vector_utils.text_to_vector([query_text])
        
        expr = None
        if category_filter:
            expr = f"category == '{category_filter}'"
        
        results = self.collection.search(
            data=query_vector,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": 128}},
            limit=top_k,
            expr=expr,
            output_fields=["text", "category"]
        )
        
        return [(hit.entity.get('text'), hit.distance) for hit in results[0]]

# 사용 예시
search_system = VectorSearchSystem("my_search")
doc_ids = search_system.add_documents(
    ["AI is amazing", "Machine learning rocks"],
    ["tech", "tech"]
)
results = search_system.search("artificial intelligence", top_k=5)
```

### 배치 처리 유틸리티
```python
def batch_process(collection, data, batch_size=1000, show_progress=True):
    """효율적인 배치 처리"""
    total_items = len(data[0])  # 첫 번째 필드의 길이
    total_inserted = 0
    
    for i in range(0, total_items, batch_size):
        # 배치 데이터 추출
        batch_data = []
        for field_data in data:
            batch_data.append(field_data[i:i+batch_size])
        
        # 삽입
        result = collection.insert(batch_data)
        total_inserted += len(result.primary_keys)
        
        # 진행 상황 출력
        if show_progress and i % (batch_size * 5) == 0:
            progress = (total_inserted / total_items) * 100
            print(f"진행률: {progress:.1f}% ({total_inserted}/{total_items})")
        
        # 주기적 플러시
        if i % (batch_size * 10) == 0:
            collection.flush()
    
    # 최종 플러시
    collection.flush()
    print(f"✅ 배치 처리 완료: {total_inserted}개 항목")
    return total_inserted
```

---

## 🔧 유용한 함수들

### 자동 스키마 생성
```python
def create_schema_from_sample(sample_data, collection_name):
    """샘플 데이터로부터 자동 스키마 생성"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    ]
    
    for field_name, sample_value in sample_data.items():
        if isinstance(sample_value, str):
            fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=512))
        elif isinstance(sample_value, int):
            fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
        elif isinstance(sample_value, float):
            fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT))
        elif isinstance(sample_value, list) and len(sample_value) > 0:
            if isinstance(sample_value[0], float):
                fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=len(sample_value)))
    
    schema = CollectionSchema(fields=fields, description=f"Auto-generated schema for {collection_name}")
    return schema
```

### 성능 벤치마크
```python
def benchmark_search(collection, query_vectors, num_runs=10):
    """검색 성능 벤치마크"""
    import time
    
    latencies = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": 128}},
            limit=10
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"평균 지연시간: {avg_latency:.2f}ms")
    print(f"최소 지연시간: {min_latency:.2f}ms")  
    print(f"최대 지연시간: {max_latency:.2f}ms")
    print(f"QPS (추정): {1000/avg_latency:.1f}")
    
    return avg_latency
```

---

이 빠른 참조 가이드를 통해 **Milvus 개발 시 자주 사용하는 코드들을 빠르게 찾아 활용**하실 수 있습니다! 📚✨ 