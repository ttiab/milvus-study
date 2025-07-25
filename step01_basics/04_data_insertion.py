#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 1단계: 데이터 삽입 및 기본 검색

이 스크립트는 Milvus에 데이터를 삽입하고 기본 검색을 수행하는 방법을 학습합니다.
- 데이터 삽입 방법
- 벡터 변환
- 기본 검색
- 검색 결과 분석
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType


def create_sample_collection(conn: MilvusConnection) -> Collection:
    """샘플 컬렉션 생성"""
    print("\n📁 샘플 컬렉션 'sample_articles' 생성 중...")
    
    collection_name = "sample_articles"
    
    # 기존 컬렉션 삭제
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"  기존 컬렉션 '{collection_name}' 삭제됨")
    
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
    
    collection = Collection(name=collection_name, schema=schema)
    print(f"  ✅ 컬렉션 '{collection_name}' 생성 완료")
    
    return collection


def prepare_sample_data() -> Dict[str, List]:
    """샘플 데이터 준비"""
    print("\n📊 샘플 데이터 준비 중...")
    
    # 샘플 문서 데이터
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
    print("  벡터 변환 중...")
    vectors = vector_utils.texts_to_vectors(combined_texts)
    
    # 삽입용 데이터 형식으로 변환
    data = {
        'titles': [article['title'] for article in articles],
        'contents': [article['content'] for article in articles],
        'vectors': vectors.tolist(),
        'categories': [article['category'] for article in articles],
        'authors': [article['author'] for article in articles],
        'scores': [article['score'] for article in articles]
    }
    
    print(f"  ✅ {len(articles)}개 문서 데이터 준비 완료")
    print(f"  📏 벡터 차원: {vectors.shape[1]}")
    
    return data


def insert_data_demo(collection: Collection, data: Dict[str, List]) -> None:
    """데이터 삽입 데모"""
    print("\n" + "="*60)
    print(" 💾 데이터 삽입 데모")
    print("="*60)
    
    # 1. 기본 삽입
    print("\n1. 기본 데이터 삽입")
    start_time = time.time()
    
    insert_result = collection.insert([
        data['titles'],
        data['contents'],
        data['vectors'],
        data['categories'],
        data['authors'],
        data['scores']
    ])
    
    insert_time = time.time() - start_time
    print(f"  삽입된 엔티티 수: {insert_result.insert_count}")
    print(f"  삽입 시간: {insert_time:.4f}초")
    print(f"  자동 생성된 ID 범위: {insert_result.primary_keys[0]} ~ {insert_result.primary_keys[-1]}")
    
    # 2. 메모리에 플러시
    print("\n2. 메모리 플러시")
    start_time = time.time()
    collection.flush()
    flush_time = time.time() - start_time
    print(f"  플러시 시간: {flush_time:.4f}초")
    
    # 3. 컬렉션 통계 확인
    print("\n3. 컬렉션 통계")
    print(f"  총 엔티티 수: {collection.num_entities}")
    
    print("  ✅ 데이터 삽입 완료!")


def create_index_demo(collection: Collection) -> None:
    """인덱스 생성 데모"""
    print("\n" + "="*60)
    print(" 🔍 인덱스 생성 데모")
    print("="*60)
    
    print("\n1. 벡터 인덱스 생성 중...")
    start_time = time.time()
    
    # IVF_FLAT 인덱스 생성
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    
    index_time = time.time() - start_time
    print(f"  ✅ 인덱스 생성 완료 ({index_time:.2f}초)")
    print(f"  인덱스 타입: IVF_FLAT")
    print(f"  메트릭: L2 (유클리드 거리)")
    print(f"  nlist 파라미터: 128")


def basic_search_demo(collection: Collection) -> None:
    """기본 검색 데모"""
    print("\n" + "="*60)
    print(" 🔍 기본 검색 데모")
    print("="*60)
    
    # 컬렉션 로드
    print("\n1. 컬렉션 로드")
    collection.load()
    print("  ✅ 컬렉션 로드 완료")
    
    # 벡터 유틸리티 초기화
    vector_utils = VectorUtils()
    
    # 검색 쿼리들
    queries = [
        "인공지능과 기계학습 기술",
        "비즈니스 전략과 경영",
        "과학 기술과 연구",
        "클라우드 컴퓨팅과 데이터"
    ]
    
    for i, query_text in enumerate(queries):
        print(f"\n{i+2}. 검색 쿼리: '{query_text}'")
        
        # 쿼리 벡터 생성
        query_vectors = vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 검색 실행
        start_time = time.time()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=["title", "category", "author", "score"]
        )
        
        search_time = time.time() - start_time
        print(f"  검색 시간: {search_time:.4f}초")
        print(f"  검색 결과 수: {len(results[0])}")
        
        # 결과 출력
        for j, hit in enumerate(results[0]):
            print(f"    {j+1}. {hit.entity.get('title')}")
            print(f"        카테고리: {hit.entity.get('category')}")
            print(f"        저자: {hit.entity.get('author')}")
            print(f"        점수: {hit.entity.get('score')}")
            print(f"        유사도 거리: {hit.distance:.4f}")


def filtered_search_demo(collection: Collection) -> None:
    """필터링 검색 데모"""
    print("\n" + "="*60)
    print(" 🎯 필터링 검색 데모")
    print("="*60)
    
    vector_utils = VectorUtils()
    
    # 1. 카테고리 필터링
    print("\n1. 카테고리 필터링 (Technology)")
    query_text = "최신 기술 동향"
    query_vectors = vector_utils.text_to_vector(query_text)
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        expr='category == "Technology"',
        output_fields=["title", "category", "score"]
    )
    
    print(f"  검색 결과 수: {len(results[0])}")
    for i, hit in enumerate(results[0]):
        print(f"    {i+1}. {hit.entity.get('title')}")
        print(f"        카테고리: {hit.entity.get('category')}")
        print(f"        점수: {hit.entity.get('score')}")
    
    # 2. 점수 필터링
    print("\n2. 점수 필터링 (score >= 4.5)")
    query_text = "고품질 연구 논문"
    query_vectors = vector_utils.text_to_vector(query_text)
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        expr='score >= 4.5',
        output_fields=["title", "author", "score"]
    )
    
    print(f"  검색 결과 수: {len(results[0])}")
    for i, hit in enumerate(results[0]):
        print(f"    {i+1}. {hit.entity.get('title')}")
        print(f"        저자: {hit.entity.get('author')}")
        print(f"        점수: {hit.entity.get('score')}")
    
    # 3. 복합 필터링
    print("\n3. 복합 필터링 (Business 카테고리 + score > 4.0)")
    query_text = "비즈니스 인사이트"
    query_vectors = vector_utils.text_to_vector(query_text)
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        expr='category == "Business" and score > 4.0',
        output_fields=["title", "category", "score"]
    )
    
    print(f"  검색 결과 수: {len(results[0])}")
    for i, hit in enumerate(results[0]):
        print(f"    {i+1}. {hit.entity.get('title')}")
        print(f"        카테고리: {hit.entity.get('category')}")
        print(f"        점수: {hit.entity.get('score')}")


def main():
    """메인 실행 함수"""
    print("💾 Milvus 데이터 삽입 및 기본 검색 실습")
    print("실행 시간:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 1. 샘플 컬렉션 생성
            collection = create_sample_collection(conn)
            
            # 2. 샘플 데이터 준비
            data = prepare_sample_data()
            
            # 3. 데이터 삽입
            insert_data_demo(collection, data)
            
            # 4. 인덱스 생성
            create_index_demo(collection)
            
            # 5. 기본 검색
            basic_search_demo(collection)
            
            # 6. 필터링 검색
            filtered_search_demo(collection)
            
            # 정리
            collection.drop()
            print("\n🧹 테스트 컬렉션 정리 완료")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 데이터 삽입 및 검색 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 텍스트 데이터를 벡터로 변환하여 삽입")
    print("  • 인덱스 생성으로 검색 성능 향상")
    print("  • 벡터 유사도 검색의 기본 원리")
    print("  • 스칼라 필드를 활용한 필터링 검색")
    print("  • 검색 결과 분석 및 해석")
    
    print("\n🚀 다음 단계:")
    print("  1단계 완료! 이제 2단계 핵심 기능 실습으로 진행하세요:")
    print("  python step02_core_features/01_index_management.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 