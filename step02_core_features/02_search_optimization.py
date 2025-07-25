#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 2단계: 검색 최적화 실습

이 스크립트는 Milvus의 검색 최적화 기법을 학습합니다.
- 검색 파라미터 최적화
- 필터링과 검색 조합
- 하이브리드 검색
- 검색 성능 분석
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType


class SearchOptimizer:
    """Milvus 검색 최적화 클래스"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        self.collection_name = "search_optimization_demo"
        
    def create_demo_collection(self) -> Collection:
        """데모용 컬렉션 생성"""
        print(f"\n📁 컬렉션 '{self.collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 (더 다양한 필드)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="publish_year", dtype=DataType.INT64),
            FieldSchema(name="rating", dtype=DataType.FLOAT),
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="is_featured", dtype=DataType.BOOL)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="검색 최적화 데모 컬렉션"
        )
        
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def generate_demo_data(self, count: int = 50000) -> List[List]:
        """데모 데이터 생성"""
        print(f"\n📊 데모 데이터 {count}개 생성 중...")
        
        categories = ["Technology", "Science", "Business", "Health", "Education", "Entertainment"]
        authors = [f"Author_{i}" for i in range(1, 101)]  # 100명의 저자
        
        titles = []
        contents = []
        category_list = []
        author_list = []
        years = []
        ratings = []
        view_counts = []
        featured_flags = []
        
        for i in range(count):
            category = np.random.choice(categories)
            author = np.random.choice(authors)
            year = np.random.randint(2020, 2025)
            
            title = f"{category} Article {i}: Advanced concepts and applications"
            content = f"This is a comprehensive article about {category.lower()} " \
                     f"written by {author} in {year}. It covers various topics " \
                     f"and provides detailed insights into the subject matter."
            
            titles.append(title)
            contents.append(content)
            category_list.append(category)
            author_list.append(author)
            years.append(year)
            ratings.append(np.random.uniform(1.0, 5.0))
            view_counts.append(np.random.randint(100, 100000))
            featured_flags.append(np.random.choice([True, False], p=[0.1, 0.9]))
        
        # 벡터 변환 (제목과 내용 결합)
        print("  벡터 변환 중...")
        combined_texts = [f"{title} {content}" for title, content in zip(titles, contents)]
        vectors = self.vector_utils.texts_to_vectors(combined_texts)
        
        data = [
            titles,
            contents,
            vectors.tolist(),
            category_list,
            author_list,
            years,
            ratings,
            view_counts,
            featured_flags
        ]
        
        print(f"  ✅ 데모 데이터 생성 완료")
        return data
    
    def insert_and_index_data(self, collection: Collection, data: List[List]) -> None:
        """데이터 삽입 및 인덱스 생성"""
        print(f"\n💾 데이터 삽입 중...")
        start_time = time.time()
        
        # 배치 단위로 데이터 삽입
        batch_size = 10000
        total_count = len(data[0])
        
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            batch_data = [field[i:end_idx] for field in data]
            collection.insert(batch_data)
            print(f"  배치 {i//batch_size + 1} 삽입 완료 ({end_idx - i}개)")
        
        collection.flush()
        print(f"  ✅ 전체 데이터 삽입 완료 ({time.time() - start_time:.2f}초)")
        
        # 인덱스 생성
        print(f"\n🔍 인덱스 생성 중...")
        start_time = time.time()
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"  ✅ 인덱스 생성 완료 ({time.time() - start_time:.2f}초)")
    
    def basic_search_demo(self, collection: Collection) -> None:
        """기본 검색 데모"""
        print("\n" + "="*60)
        print(" 🔍 기본 검색 데모")
        print("="*60)
        
        collection.load()
        
        # 검색 쿼리
        query_text = "advanced technology artificial intelligence machine learning"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        print(f"검색 쿼리: '{query_text}'")
        
        # 기본 검색
        print("\n1. 기본 유사도 검색")
        start_time = time.time()
        
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=10,
            output_fields=["title", "category", "author", "rating"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:50]}...")
            print(f"      카테고리: {hit.entity.get('category')}, "
                  f"저자: {hit.entity.get('author')}, "
                  f"평점: {hit.entity.get('rating'):.2f}, "
                  f"거리: {hit.distance:.4f}")
    
    def filtered_search_demo(self, collection: Collection) -> None:
        """필터링 검색 데모"""
        print("\n" + "="*60)
        print(" 🎯 필터링 검색 데모")
        print("="*60)
        
        query_text = "business strategy and innovation"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        print(f"검색 쿼리: '{query_text}'")
        
        # 1. 카테고리 필터링
        print("\n1. 카테고리 필터: Business")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            expr='category == "Business"',
            output_fields=["title", "category", "rating"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:50]}...")
            print(f"      카테고리: {hit.entity.get('category')}, "
                  f"평점: {hit.entity.get('rating'):.2f}")
        
        # 2. 복합 필터링
        print("\n2. 복합 필터: Technology + 높은 평점 + 최근 년도")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            expr='category == "Technology" and rating > 4.0 and publish_year >= 2023',
            output_fields=["title", "category", "rating", "publish_year"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:50]}...")
            print(f"      카테고리: {hit.entity.get('category')}, "
                  f"평점: {hit.entity.get('rating'):.2f}, "
                  f"년도: {hit.entity.get('publish_year')}")
        
        # 3. 범위 필터링
        print("\n3. 범위 필터: 조회수 10,000 이상")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            expr='view_count >= 10000',
            output_fields=["title", "view_count", "is_featured"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:50]}...")
            print(f"      조회수: {hit.entity.get('view_count')}, "
                  f"추천: {hit.entity.get('is_featured')}")
    
    def search_parameter_tuning(self, collection: Collection) -> None:
        """검색 파라미터 튜닝"""
        print("\n" + "="*60)
        print(" 🎛️ 검색 파라미터 튜닝")
        print("="*60)
        
        query_text = "scientific research methodology"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # ef 파라미터 튜닝
        ef_values = [50, 100, 200, 400]
        
        print("HNSW ef 파라미터 영향 분석:")
        print(f"{'ef':<5} {'검색시간(초)':<12} {'QPS':<8}")
        print("-" * 30)
        
        for ef in ef_values:
            times = []
            
            # 여러 번 측정하여 평균 계산
            for _ in range(5):
                start_time = time.time()
                
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "L2", "params": {"ef": ef}},
                    limit=10,
                    output_fields=["title"]
                )
                
                search_time = time.time() - start_time
                times.append(search_time)
            
            avg_time = np.mean(times)
            qps = 1 / avg_time
            
            print(f"{ef:<5} {avg_time:<12.4f} {qps:<8.2f}")
    
    def hybrid_search_demo(self, collection: Collection) -> None:
        """하이브리드 검색 데모"""
        print("\n" + "="*60)
        print(" 🔄 하이브리드 검색 데모")
        print("="*60)
        
        # 여러 검색 쿼리 조합
        queries = [
            "artificial intelligence machine learning",
            "business strategy management",
            "scientific research methodology"
        ]
        
        query_vectors = []
        for q in queries:
            qv = self.vector_utils.text_to_vector(q)
            query_vectors.append(qv[0] if len(qv.shape) > 1 else qv)
        
        print("다중 쿼리 검색:")
        start_time = time.time()
        
        # 여러 벡터로 동시 검색
        results = collection.search(
            data=query_vectors,
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=3,
            output_fields=["title", "category"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        
        for i, (query, result) in enumerate(zip(queries, results)):
            print(f"\n쿼리 {i+1}: '{query}'")
            for j, hit in enumerate(result):
                print(f"  {j+1}. {hit.entity.get('title')[:40]}... "
                      f"({hit.entity.get('category')})")
    
    def performance_analysis(self, collection: Collection) -> None:
        """성능 분석"""
        print("\n" + "="*60)
        print(" 📊 성능 분석")
        print("="*60)
        
        query_text = "technology innovation development"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 다양한 조건에서 성능 측정
        test_cases = [
            {"name": "기본 검색", "limit": 10, "expr": None},
            {"name": "필터링 검색", "limit": 10, "expr": 'category == "Technology"'},
            {"name": "복합 필터링", "limit": 10, "expr": 'category == "Technology" and rating > 3.0'},
            {"name": "대량 결과", "limit": 100, "expr": None},
        ]
        
        print(f"{'테스트 케이스':<15} {'평균시간(초)':<12} {'QPS':<8} {'결과수':<6}")
        print("-" * 50)
        
        for case in test_cases:
            times = []
            result_count = 0
            
            for _ in range(3):
                start_time = time.time()
                
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "L2", "params": {"ef": 100}},
                    limit=case["limit"],
                    expr=case["expr"],
                    output_fields=["title"]
                )
                
                search_time = time.time() - start_time
                times.append(search_time)
                result_count = len(results[0])
            
            avg_time = np.mean(times)
            qps = 1 / avg_time
            
            print(f"{case['name']:<15} {avg_time:<12.4f} {qps:<8.2f} {result_count:<6}")


def main():
    """메인 실행 함수"""
    print("🚀 Milvus 검색 최적화 실습 시작")
    print("실행 시간:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 검색 최적화 매니저 생성
            optimizer = SearchOptimizer(conn)
            
            # 1. 데모 컬렉션 생성 및 데이터 준비
            collection = optimizer.create_demo_collection()
            demo_data = optimizer.generate_demo_data(20000)  # 20K 데이터
            optimizer.insert_and_index_data(collection, demo_data)
            
            # 2. 기본 검색 데모
            optimizer.basic_search_demo(collection)
            
            # 3. 필터링 검색 데모
            optimizer.filtered_search_demo(collection)
            
            # 4. 검색 파라미터 튜닝
            optimizer.search_parameter_tuning(collection)
            
            # 5. 하이브리드 검색 데모
            optimizer.hybrid_search_demo(collection)
            
            # 6. 성능 분석
            optimizer.performance_analysis(collection)
            
            # 정리
            collection.drop()
            print("\n🧹 테스트 컬렉션 정리 완료")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 검색 최적화 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 필터링과 벡터 검색을 조합하여 정확한 결과 얻기")
    print("  • ef 파라미터로 검색 정확도와 속도 조절")
    print("  • 복합 조건으로 세밀한 검색 조건 설정")
    print("  • 다중 쿼리로 효율적인 배치 검색")
    print("  • 성능 분석으로 최적 설정 찾기")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 