# 2단계: 핵심 기능 실습 (Core Feature Practice)

이 단계에서는 Milvus의 핵심 기능인 인덱싱과 검색 최적화를 심화 학습합니다.

## 학습 목표

- ✅ 다양한 인덱스 알고리즘 이해 및 성능 비교
- ✅ 검색 파라미터 최적화와 성능 튜닝
- ✅ 파티션 기반 데이터 관리 전략
- ✅ 하이브리드 검색 및 복합 조건 검색

## 📚 실습 파일

### 1. **01_index_management.py** - 인덱스 관리 실습
```bash
python step02_core_features/01_index_management.py
```
**학습 내용:**
- IVF_FLAT, IVF_SQ8, HNSW 인덱스 비교
- 인덱스 파라미터 튜닝 (M, efConstruction, nlist)
- 성능 벤치마크 및 QPS 측정
- 메모리 사용량 vs 검색 속도 트레이드오프

### 2. **02_search_optimization.py** - 검색 최적화 실습  
```bash
python step02_core_features/02_search_optimization.py
```
**학습 내용:**
- 기본 벡터 검색 vs 필터링 검색
- 복합 조건 검색 (AND, OR, 범위 조건)
- ef 파라미터 튜닝으로 속도/정확도 조절
- 다중 쿼리 배치 검색
- 검색 성능 분석 및 최적화

### 3. **03_partition_management.py** - 파티션 관리 실습
```bash
python step02_core_features/03_partition_management.py
```
**학습 내용:**
- 카테고리/지역/시간 기반 파티션 전략
- 파티션별 데이터 삽입 및 관리
- 파티션 검색 성능 비교
- 파티션 로드/언로드 최적화
- 파티션 통계 및 모니터링

## 🚀 시작하기

### 사전 요구사항
- ✅ 1단계 기본 실습 완료
- ✅ Milvus 서버 실행 중 (`docker-compose ps`로 확인)
- ✅ 가상환경 활성화 (`source venv/bin/activate`)

### 권장 실습 순서
```bash
# 1단계 완료 확인
python step01_basics/04_data_insertion.py

# 2단계 실습 시작
python step02_core_features/01_index_management.py
python step02_core_features/02_search_optimization.py  
python step02_core_features/03_partition_management.py
```

## 🎯 주요 기술 스택

| 기술 영역 | 기술 스택 | 설명 |
|-----------|-----------|------|
| **인덱스** | IVF_FLAT, IVF_SQ8, HNSW | 다양한 인덱스 알고리즘 비교 |
| **검색** | 벡터 검색, 필터링, 하이브리드 | 최적화된 검색 전략 |
| **파티션** | 시간/카테고리/지역 분할 | 대용량 데이터 관리 |
| **메트릭** | L2, IP, 성능 측정 | 거리 계산 및 성능 분석 |
| **최적화** | 파라미터 튜닝, 벤치마킹 | 실제 운영 최적화 |

## 💡 학습 성과

이 단계를 완료하면 다음을 할 수 있게 됩니다:

- 🔍 **인덱스 전문가**: 다양한 인덱스 알고리즘의 특성과 최적 사용 시나리오 이해
- ⚡ **성능 최적화**: 검색 성능을 위한 파라미터 튜닝과 성능 분석
- 📊 **데이터 관리**: 파티션을 활용한 효율적인 대용량 데이터 관리
- 🔧 **실무 적용**: 실제 프로덕션 환경에서 활용 가능한 최적화 전략

## 🛠️ 실습 특징

- **실제 성능 데이터**: 각 실습에서 QPS, 레이턴시 등 실제 성능 지표 측정
- **비교 분석**: 다양한 설정의 성능을 비교하여 최적 조합 도출
- **실무 중심**: 실제 운영 환경에서 바로 적용 가능한 실습 내용
- **자동 정리**: 실습 완료 후 테스트 데이터 자동 정리

## 다음 단계

2단계 완료 후 [3단계: 실제 사용 사례 구현](../step03_use_cases/README.md)으로 진행하세요. 