#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 3단계: 텍스트 유사도 검색 시스템 (간단 버전)

2단계에서 검증된 안정적인 패턴을 사용합니다.
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
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType


def create_text_collection(collection_name: str = "text_search_demo") -> Collection:
    """텍스트 검색용 컬렉션 생성"""
    print(f"\n📁 텍스트 검색 컬렉션 '{collection_name}' 생성 중...")
    
    # 기존 컬렉션 삭제
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"  기존 컬렉션 삭제됨")
    
    # 간단한 스키마 (2단계 패턴)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="텍스트 검색 데모 컬렉션"
    )
    
    collection = Collection(name=collection_name, schema=schema)
    print(f"  ✅ 컬렉션 생성 완료")
    return collection


def generate_sample_documents(count: int = 100) -> List[Dict[str, Any]]:
    """샘플 문서 생성"""
    print(f"\n📊 샘플 문서 {count}개 생성 중...")
    
    categories = ["Technology", "Science", "Business", "Health", "Education"]
    authors = [f"Author_{i}" for i in range(1, 21)]
    
    documents = []
    
    for i in range(count):
        category = np.random.choice(categories)
        author = np.random.choice(authors)
        
        title = f"{category} Article {i+1}: Advanced concepts and applications"
        content = f"This {category.lower()} article discusses advanced concepts in {category.lower()}. "
        content += f"Written by {author}, this comprehensive guide covers key principles and practical applications. "
        content += f"The content is designed for professionals and students interested in {category.lower()} research."
        
        doc = {
            "title": title,
            "content": content,
            "category": category,
            "author": author,
            "score": round(np.random.uniform(1.0, 5.0), 2)
        }
        
        documents.append(doc)
    
    print(f"  ✅ {count}개 문서 생성 완료")
    return documents


def insert_documents(collection: Collection, documents: List[Dict[str, Any]], vector_utils: VectorUtils) -> None:
    """문서 데이터 삽입"""
    print(f"\n💾 문서 데이터 삽입 중...")
    
    # 텍스트 벡터화
    combined_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    vectors = vector_utils.texts_to_vectors(combined_texts)
    
    # 데이터를 2단계 패턴으로 구성 (List[List])
    data = [
        [doc["title"] for doc in documents],        # 제목들
        [doc["content"] for doc in documents],      # 내용들
        [doc["category"] for doc in documents],     # 카테고리들
        [doc["author"] for doc in documents],       # 저자들
        [doc["score"] for doc in documents],        # 점수들
        vectors.tolist()                            # 벡터들
    ]
    
    print(f"  첫 번째 데이터 샘플:")
    print(f"    제목: {data[0][0][:50]}...")
    print(f"    벡터 길이: {len(data[5][0])}")
    
    # 삽입 (2단계 패턴)
    result = collection.insert(data)
    print(f"  삽입된 엔티티 수: {len(result.primary_keys)}")
    
    # 플러시
    collection.flush()
    print(f"  ✅ 데이터 삽입 완료")


def create_index(collection: Collection) -> None:
    """인덱스 생성"""
    print(f"\n🔍 인덱스 생성 중...")
    
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
    
    collection.create_index("vector", index_params)
    print(f"  ✅ 인덱스 생성 완료")


def document_search_demo(collection: Collection, vector_utils: VectorUtils) -> None:
    """문서 검색 데모"""
    print("\n" + "="*70)
    print(" 📖 문서 검색 데모")
    print("="*70)
    
    collection.load()
    
    search_queries = [
        "artificial intelligence machine learning technology",
        "scientific research methodology analysis",
        "business strategy management innovation",
        "health medical treatment care"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n{i}. 검색 쿼리: '{query}'")
        
        # 쿼리 벡터화
        query_vectors = vector_utils.text_to_vector(query)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 검색
        start_time = time.time()
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=3,
            output_fields=["title", "category", "author", "score"]
        )
        search_time = time.time() - start_time
        
        print(f"   검색 시간: {search_time:.4f}초")
        print(f"   결과 수: {len(results[0])}")
        
        for j, hit in enumerate(results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     {j+1}. {entity.get('title')[:60]}...")
            print(f"        카테고리: {entity.get('category')}, 저자: {entity.get('author')}")
            print(f"        점수: {entity.get('score')}, 유사도: {similarity:.3f}")


def text_to_text_search_demo(collection: Collection, vector_utils: VectorUtils) -> None:
    """텍스트-텍스트 검색 데모"""
    print("\n" + "="*70)
    print(" 💬 시맨틱 검색 데모")
    print("="*70)
    
    semantic_queries = [
        {
            "query": "기계가 사람처럼 생각하는 방법",
            "description": "AI 관련 문서 검색"
        },
        {
            "query": "회사 수익을 늘리는 전략",
            "description": "비즈니스 관련 문서 검색"
        },
        {
            "query": "병을 치료하는 새로운 방법",
            "description": "의료 관련 문서 검색"
        }
    ]
    
    for i, case in enumerate(semantic_queries, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   검색어: '{case['query']}'")
        
        # 쿼리 벡터화
        query_vectors = vector_utils.text_to_vector(case['query'])
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 검색
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=3,
            output_fields=["title", "category", "score"]
        )
        
        print(f"   검색 결과:")
        for j, hit in enumerate(results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     {j+1}. {entity.get('title')[:50]}...")
            print(f"        카테고리: {entity.get('category')}, 유사도: {similarity:.3f}")


def filtered_search_demo(collection: Collection, vector_utils: VectorUtils) -> None:
    """필터링 검색 데모"""
    print("\n" + "="*70)
    print(" 🎯 필터링 검색 데모")
    print("="*70)
    
    query_text = "advanced research technology innovation"
    query_vectors = vector_utils.text_to_vector(query_text)
    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
    
    # 1. 카테고리 필터링
    print(f"\n1. 카테고리 필터링 (Technology)")
    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 200}},
        limit=3,
        expr="category == 'Technology'",
        output_fields=["title", "author", "score"]
    )
    
    print(f"   Technology 카테고리 결과:")
    for j, hit in enumerate(results[0]):
        similarity = 1 - hit.distance
        entity = hit.entity
        print(f"     {j+1}. {entity.get('title')[:50]}...")
        print(f"        저자: {entity.get('author')}, 점수: {entity.get('score')}")
        print(f"        유사도: {similarity:.3f}")
    
    # 2. 점수 필터링
    print(f"\n2. 고품질 문서 필터링 (score >= 4.0)")
    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="vector", 
        param={"metric_type": "COSINE", "params": {"ef": 200}},
        limit=3,
        expr="score >= 4.0",
        output_fields=["title", "category", "score"]
    )
    
    print(f"   고품질 문서 결과:")
    for j, hit in enumerate(results[0]):
        similarity = 1 - hit.distance
        entity = hit.entity
        print(f"     {j+1}. {entity.get('title')[:50]}...")
        print(f"        카테고리: {entity.get('category')}, 점수: {entity.get('score')}")
        print(f"        유사도: {similarity:.3f}")


def main():
    """메인 실행 함수"""
    print("🚀 텍스트 유사도 검색 시스템 실습 (간단 버전)")
    print(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 벡터 유틸리티 초기화
            vector_utils = VectorUtils()
            
            # 컬렉션 생성
            collection = create_text_collection()
            
            # 샘플 데이터 생성 및 삽입
            documents = generate_sample_documents(100)
            insert_documents(collection, documents, vector_utils)
            
            # 인덱스 생성
            create_index(collection)
            
            # 검색 데모
            document_search_demo(collection, vector_utils)
            text_to_text_search_demo(collection, vector_utils)
            filtered_search_demo(collection, vector_utils)
            
            # 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            utility.drop_collection("text_search_demo")
            print("✅ 정리 완료")
            
        print("\n🎉 텍스트 검색 시스템 실습 완료!")
        
        print("\n💡 학습 포인트:")
        print("  • 시맨틱 검색으로 의미 기반 문서 찾기")
        print("  • 필터링과 벡터 검색의 효과적 조합")
        print("  • 다양한 도메인에서의 텍스트 유사도 활용")
        print("  • 실시간 검색 성능 최적화")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 