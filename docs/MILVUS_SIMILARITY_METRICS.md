# Milvus 유사도 메트릭 (Similarity Metrics) 완전 가이드

## 개요

유사도 메트릭(Similarity Metrics)은 벡터 데이터베이스에서 두 벡터 간의 유사도나 거리를 측정하는 수학적 함수입니다. Milvus에서는 다양한 유사도 메트릭을 제공하여 서로 다른 데이터 유형과 사용 사례에 최적화된 검색을 지원합니다.

올바른 유사도 메트릭 선택은 검색 정확도와 성능에 직접적인 영향을 미치므로, 데이터의 특성과 애플리케이션 요구사항을 고려하여 선택해야 합니다.

## Milvus가 지원하는 유사도 메트릭

### 1. 부동소수점 벡터 메트릭

#### 1.1 유클리드 거리 (Euclidean Distance, L2)

**수식:**
```
d = √(Σ(ai - bi)²)
```

**특징:**
- 가장 직관적인 거리 측정 방법
- 벡터 간의 실제 공간적 거리를 계산
- 값이 작을수록 유사도가 높음
- 범위: [0, ∞)

**사용 사례:**
- 이미지 특징 벡터
- 지리적 좌표 데이터
- 일반적인 수치 데이터 분석

**예제:**
```python
# 벡터 A = [1, 2, 3], 벡터 B = [4, 5, 6]
# L2 거리 = √((1-4)² + (2-5)² + (3-6)²) = √(9 + 9 + 9) = √27 ≈ 5.196
```

#### 1.2 내적 (Inner Product, IP)

**수식:**
```
similarity = Σ(ai × bi)
```

**특징:**
- 벡터의 방향과 크기를 모두 고려
- 값이 클수록 유사도가 높음
- 정규화되지 않은 벡터에 적합
- 범위: (-∞, ∞)

**사용 사례:**
- 추천 시스템에서 사용자-아이템 선호도
- 자연어 처리에서 단어 빈도 벡터
- 협업 필터링

**주의사항:**
- 벡터가 정규화되지 않은 경우에만 의미 있음
- 정규화된 벡터에서는 코사인 유사도와 동일

#### 1.3 코사인 유사도 (Cosine Similarity)

**수식:**
```
similarity = (Σ(ai × bi)) / (√(Σ(ai²)) × √(Σ(bi²)))
```

**특징:**
- 벡터의 방향만 고려, 크기는 무시
- -1부터 1까지의 값 범위
- 1에 가까울수록 유사도가 높음
- 정규화된 벡터에서는 내적과 동일

**사용 사례:**
- 텍스트 문서 유사도 (TF-IDF, Word2Vec, BERT 임베딩)
- 사용자 행동 패턴 분석
- 이미지 특징 벡터 (정규화된 경우)

**예제:**
```python
# 벡터 A = [1, 2, 3], 벡터 B = [2, 4, 6]
# 코사인 유사도 = (1×2 + 2×4 + 3×6) / (√14 × √56) = 28 / 28 = 1.0
# (벡터 B는 벡터 A의 2배이므로 방향이 같아 유사도 1.0)
```

### 2. 이진 벡터 메트릭

#### 2.1 해밍 거리 (Hamming Distance)

**수식:**
```
d = Σ(ai ⊕ bi)  // ⊕는 XOR 연산
```

**특징:**
- 이진 벡터 간의 다른 비트 수를 계산
- 값이 작을수록 유사도가 높음
- 계산이 매우 빠름 (비트 연산)

**사용 사례:**
- 이진화된 특징 벡터
- 해시 기반 검색
- 빠른 근사 검색

#### 2.2 자카드 거리 (Jaccard Distance)

**수식:**
```
Jaccard Index = |A ∩ B| / |A ∪ B|
Jaccard Distance = 1 - Jaccard Index
```

**특징:**
- 집합의 교집합과 합집합 비율로 계산
- 0과 1 사이의 값
- 0에 가까울수록 유사도가 높음

**사용 사례:**
- 문서 또는 사용자의 관심사 집합 비교
- 추천 시스템에서 사용자 유사도
- 카테고리형 데이터 분석

#### 2.3 타니모토 거리 (Tanimoto Distance)

**수식:**
```
Tanimoto = (A ∩ B) / (A ∪ B)
```

**특징:**
- 자카드 거리와 유사하지만 이진 벡터에 특화
- 화학 분야에서 분자 구조 비교에 주로 사용
- 0과 1 사이의 값

### 3. 희소 벡터 메트릭

#### 3.1 IP (Inner Product) for Sparse Vectors

**특징:**
- 희소 벡터(대부분의 값이 0인 벡터)에 최적화
- 0이 아닌 값들만 계산하여 효율적
- 텍스트 검색에서 BM25와 함께 사용

**사용 사례:**
- 전체 텍스트 검색 (BM25)
- 희소 임베딩 (SPLADE, BGE-M3)
- 대규모 텍스트 코퍼스 검색

## 메트릭 선택 가이드

### 데이터 유형별 권장 메트릭

| 데이터 유형 | 권장 메트릭 | 이유 |
|------------|------------|------|
| 텍스트 임베딩 (BERT, Word2Vec) | Cosine | 텍스트의 의미적 유사도는 방향성이 중요 |
| 이미지 특징 벡터 | L2 또는 Cosine | 정규화 여부에 따라 선택 |
| 사용자 선호도 벡터 | Inner Product | 선호도의 강도도 중요한 정보 |
| 이진 특징 | Hamming | 빠른 계산과 직관적 해석 |
| 카테고리 집합 | Jaccard | 집합 간 겹치는 정도가 중요 |
| 희소 텍스트 벡터 | IP (Sparse) | 효율적인 희소 벡터 계산 |

### 성능 고려사항

| 메트릭 | 계산 복잡도 | 메모리 사용량 | 인덱스 효율성 |
|--------|-------------|---------------|---------------|
| L2 | O(d) | 보통 | 높음 |
| Cosine | O(d) | 보통 | 높음 |
| Inner Product | O(d) | 보통 | 높음 |
| Hamming | O(d/w) | 낮음 | 매우 높음 |
| Jaccard | O(d) | 보통 | 보통 |

*d = 벡터 차원, w = 워드 크기 (보통 64비트)*

## 실제 사용 예제

### 1. 컬렉션 생성 시 메트릭 지정

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

# 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, "텍스트 임베딩 컬렉션")

# 컬렉션 생성
collection = Collection("text_embeddings", schema)

# 인덱스 생성 시 메트릭 지정
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",  # 코사인 유사도 사용
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)
```

### 2. 검색 시 메트릭 활용

```python
# 검색 수행
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10
)

# 결과에서 유사도 점수 확인
for result in results[0]:
    print(f"ID: {result.id}, 유사도: {result.distance}")
```

### 3. 다양한 메트릭 비교

```python
import numpy as np

def compare_metrics(vec1, vec2):
    # L2 거리
    l2_distance = np.linalg.norm(vec1 - vec2)
    
    # 코사인 유사도
    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # 내적
    inner_product = np.dot(vec1, vec2)
    
    print(f"L2 거리: {l2_distance:.4f}")
    print(f"코사인 유사도: {cosine_sim:.4f}")
    print(f"내적: {inner_product:.4f}")

# 예제 벡터
vec1 = np.array([1, 2, 3, 4])
vec2 = np.array([2, 4, 6, 8])
compare_metrics(vec1, vec2)
```

## 메트릭별 인덱스 지원

| 메트릭 | FLAT | IVF_FLAT | IVF_SQ8 | IVF_PQ | HNSW | RHNSW_FLAT | RHNSW_SQ | RHNSW_PQ | ANNOY |
|--------|------|----------|---------|--------|------|------------|----------|----------|-------|
| L2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| IP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| COSINE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HAMMING | ✓ | ✓ | - | - | - | - | - | - | - |
| JACCARD | ✓ | ✓ | - | - | - | - | - | - | - |

## 최적화 팁

### 1. 정규화 고려사항
```python
# 벡터 정규화
import numpy as np

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# 정규화된 벡터에서는 IP와 COSINE이 동일
normalized_vec = normalize_vector(original_vec)
```

### 2. 배치 정규화
```python
def batch_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# 대량의 벡터를 한 번에 정규화
normalized_batch = batch_normalize(vector_batch)
```

### 3. 메트릭별 최적 파라미터

```python
# L2 거리에 최적화된 인덱스 설정
l2_index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}

# 코사인 유사도에 최적화된 인덱스 설정
cosine_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}
```

## 주의사항 및 모범 사례

### 1. 메트릭 일관성
- 인덱스 생성과 검색 시 동일한 메트릭 사용 필수
- 메트릭 변경 시 인덱스 재구축 필요

### 2. 데이터 전처리
- 코사인 유사도 사용 시 벡터 정규화 권장
- 내적 사용 시 원본 벡터의 크기 정보 보존

### 3. 성능 최적화
- 이진 벡터는 해밍 거리가 가장 빠름
- 고차원 벡터에서는 코사인 유사도가 안정적
- 희소 벡터에는 전용 메트릭 사용

### 4. 결과 해석
- L2: 값이 작을수록 유사
- Cosine: 값이 클수록 유사 (1에 가까울수록)
- IP: 값이 클수록 유사
- Hamming: 값이 작을수록 유사

이러한 유사도 메트릭들을 올바르게 이해하고 사용하면 Milvus에서 더욱 정확하고 효율적인 벡터 검색을 수행할 수 있습니다. 