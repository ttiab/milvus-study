# 1단계: 기초 학습 및 환경 구축

이 단계에서는 Milvus의 기본 개념을 익히고 개발 환경을 구축합니다.

## 학습 목표

- ✅ Milvus 기본 개념 이해
- ✅ 개발 환경 구축 및 연결 테스트
- ✅ 기본 API 사용법 익히기
- ✅ 컬렉션 생성 및 관리
- ✅ 데이터 CRUD 작업

## 실습 내용

### 1.1 환경 구축 ([01_environment_setup.py](./01_environment_setup.py))
- Docker를 이용한 Milvus 설치
- Python 환경 설정
- 연결 테스트

### 1.2 기본 연결 ([02_basic_connection.py](./02_basic_connection.py))
- Milvus 서버 연결
- 연결 상태 확인
- 기본 정보 조회

### 1.3 컬렉션 관리 ([03_collection_management.py](./03_collection_management.py))
- 컬렉션 스키마 설계
- 컬렉션 생성/삭제
- 컬렉션 정보 조회

### 1.4 데이터 삽입 ([04_data_insertion.py](./04_data_insertion.py))
- 벡터 데이터 준비
- 단일/배치 데이터 삽입
- 삽입 성능 측정

### 1.5 기본 검색 ([05_basic_search.py](./05_basic_search.py))
- 벡터 유사도 검색
- 검색 파라미터 설정
- 결과 분석

### 1.6 실습 문제 ([06_exercises.py](./06_exercises.py))
- 종합 실습 문제
- 실전 예제

## 시작하기

### 사전 준비

1. **Docker 환경 확인**
```bash
# Docker가 실행 중인지 확인
docker --version
docker-compose --version
```

2. **Milvus 서버 시작**
```bash
# 프로젝트 루트에서 실행
docker-compose up -d

# 서비스 상태 확인
docker-compose ps
```

3. **Python 환경 설정**
```bash
# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 실습 순서

1. **환경 테스트**
```bash
python common/test_connection.py
```

2. **단계별 실습**
```bash
# 1.1 환경 구축
python step01_basics/01_environment_setup.py

# 1.2 기본 연결
python step01_basics/02_basic_connection.py

# 1.3 컬렉션 관리
python step01_basics/03_collection_management.py

# 1.4 데이터 삽입
python step01_basics/04_data_insertion.py

# 1.5 기본 검색
python step01_basics/05_basic_search.py

# 1.6 실습 문제
python step01_basics/06_exercises.py
```

## 주요 개념

### 벡터 데이터베이스란?
- 고차원 벡터 데이터를 저장하고 검색하는 특수한 데이터베이스
- 유사도 기반 검색에 최적화
- AI/ML 애플리케이션에서 핵심 역할

### Milvus 아키텍처
- **Proxy**: 클라이언트 요청 처리
- **Query Node**: 검색 쿼리 실행
- **Data Node**: 데이터 저장 및 관리
- **Index Node**: 인덱스 구축
- **Coord**: 전체 조정 및 관리

### 기본 용어
- **Collection**: 테이블과 유사한 개념
- **Field**: 컬럼과 유사한 개념
- **Entity**: 레코드와 유사한 개념
- **Vector Field**: 벡터 데이터를 저장하는 필드
- **Index**: 검색 성능 향상을 위한 데이터 구조

## 문제 해결

### 자주 발생하는 문제

1. **연결 실패**
```bash
# Milvus 서비스 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs milvus-standalone
```

2. **포트 충돌 (MinIO 9000, 9001 포트)**
```bash
# 포트 사용 현황 확인
lsof -i :9000
lsof -i :9001

# 해결: docker-compose.yml에서 MinIO 포트를 9010, 9011로 변경됨
```

3. **marshmallow 호환성 오류**
```bash
# 해결: 호환되는 버전으로 설치
pip install "marshmallow<4.0.0"
```

4. **메모리 부족**
```bash
# Docker 메모리 설정 확인
docker stats
```

### 도움말
- [Milvus 공식 문서](https://milvus.io/docs)
- [PyMilvus API 참조](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- [Milvus 예제](https://github.com/milvus-io/milvus/tree/master/examples)

## 다음 단계

1단계를 완료한 후, [2단계: 핵심 기능 실습](../step02_core_features/README.md)으로 진행하세요. 