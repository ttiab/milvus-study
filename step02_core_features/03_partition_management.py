#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 2단계: 파티션 관리 실습

이 스크립트는 Milvus의 파티션 관리를 학습합니다.
- 파티션 생성 및 관리
- 파티션별 데이터 삽입
- 파티션 기반 검색
- 파티션 성능 최적화
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType


class PartitionManager:
    """Milvus 파티션 관리 클래스"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        self.collection_name = "partition_demo"
        
    def create_partitioned_collection(self) -> Collection:
        """파티션을 위한 컬렉션 생성"""
        print(f"\n📁 파티션 데모 컬렉션 '{self.collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의
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
        
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def create_partitions(self, collection: Collection) -> Dict[str, str]:
        """다양한 기준으로 파티션 생성"""
        print(f"\n🗂️ 파티션 생성 중...")
        
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
            collection.create_partition(partition_name)
            all_partitions[partition_name] = f"카테고리: {category}"
            print(f"  ✅ 파티션 '{partition_name}' 생성됨")
        
        # 지역 파티션 생성
        for region in region_partitions:
            partition_name = f"region_{region}"
            collection.create_partition(partition_name)
            all_partitions[partition_name] = f"지역: {region}"
            print(f"  ✅ 파티션 '{partition_name}' 생성됨")
        
        # 시간 파티션 생성
        for year in time_partitions:
            partition_name = f"year_{year}"
            collection.create_partition(partition_name)
            all_partitions[partition_name] = f"년도: {year}"
            print(f"  ✅ 파티션 '{partition_name}' 생성됨")
        
        return all_partitions
    
    def list_partitions(self, collection: Collection) -> None:
        """파티션 목록 조회"""
        print(f"\n📋 파티션 목록 조회")
        
        partitions = collection.partitions
        print(f"총 파티션 수: {len(partitions)}")
        
        for i, partition in enumerate(partitions):
            print(f"  {i+1}. {partition.name}")
            if hasattr(partition, 'description') and partition.description:
                print(f"      설명: {partition.description}")
    
    def generate_partitioned_data(self, partition_type: str, partition_value: str, count: int = 1000) -> List[List]:
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
            dates.append(f"{year}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}")
            priorities.append(np.random.randint(1, 6))
            scores.append(np.random.uniform(1.0, 5.0))
        
        # 벡터 변환
        combined_texts = [f"{title} {content}" for title, content in zip(titles, contents)]
        vectors = self.vector_utils.texts_to_vectors(combined_texts)
        
        data = [
            titles,
            contents,
            vectors.tolist(),
            categories,
            regions,
            dates,
            priorities,
            scores
        ]
        
        return data
    
    def insert_partitioned_data(self, collection: Collection) -> None:
        """파티션별 데이터 삽입"""
        print(f"\n💾 파티션별 데이터 삽입 중...")
        
        # 카테고리별 데이터 삽입
        categories = ["tech", "business", "science", "health"]
        for category in categories:
            partition_name = f"category_{category}"
            data = self.generate_partitioned_data("category", category, 2000)
            
            insert_result = collection.insert(data, partition_name=partition_name)
            print(f"  ✅ 파티션 '{partition_name}': {insert_result.insert_count}개 삽입")
        
        # 지역별 데이터 삽입
        regions = ["asia", "europe", "america"]
        for region in regions:
            partition_name = f"region_{region}"
            data = self.generate_partitioned_data("region", region, 1500)
            
            insert_result = collection.insert(data, partition_name=partition_name)
            print(f"  ✅ 파티션 '{partition_name}': {insert_result.insert_count}개 삽입")
        
        # 시간별 데이터 삽입
        years = ["2023", "2024", "2025"]
        for year in years:
            partition_name = f"year_{year}"
            data = self.generate_partitioned_data("year", year, 1000)
            
            insert_result = collection.insert(data, partition_name=partition_name)
            print(f"  ✅ 파티션 '{partition_name}': {insert_result.insert_count}개 삽입")
        
        # 데이터 플러시
        collection.flush()
        print(f"  ✅ 모든 파티션 데이터 삽입 완료")
    
    def create_indexes(self, collection: Collection) -> None:
        """파티션별 인덱스 생성"""
        print(f"\n🔍 인덱스 생성 중...")
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"  ✅ 벡터 인덱스 생성 완료")
    
    def partition_search_demo(self, collection: Collection) -> None:
        """파티션 검색 데모"""
        print("\n" + "="*60)
        print(" 🔍 파티션 검색 데모")
        print("="*60)
        
        collection.load()
        
        query_text = "advanced technology artificial intelligence innovation"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        print(f"검색 쿼리: '{query_text}'")
        
        # 1. 전체 컬렉션 검색
        print("\n1. 전체 컬렉션 검색")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            output_fields=["title", "category", "region", "created_date"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        print(f"결과 수: {len(results[0])}")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:40]}...")
            print(f"      카테고리: {hit.entity.get('category')}, "
                  f"지역: {hit.entity.get('region')}, "
                  f"날짜: {hit.entity.get('created_date')}")
        
        # 2. 특정 파티션 검색
        print("\n2. 특정 파티션 검색 (category_tech)")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            partition_names=["category_tech"],
            output_fields=["title", "category", "region"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        print(f"결과 수: {len(results[0])}")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:40]}...")
            print(f"      카테고리: {hit.entity.get('category')}, "
                  f"지역: {hit.entity.get('region')}")
        
        # 3. 다중 파티션 검색
        print("\n3. 다중 파티션 검색 (region_asia, region_europe)")
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=5,
            partition_names=["region_asia", "region_europe"],
            output_fields=["title", "region", "created_date"]
        )
        
        search_time = time.time() - start_time
        print(f"검색 시간: {search_time:.4f}초")
        print(f"결과 수: {len(results[0])}")
        
        for i, hit in enumerate(results[0]):
            print(f"  {i+1}. {hit.entity.get('title')[:40]}...")
            print(f"      지역: {hit.entity.get('region')}, "
                  f"날짜: {hit.entity.get('created_date')}")
    
    def partition_performance_comparison(self, collection: Collection) -> None:
        """파티션 성능 비교"""
        print("\n" + "="*60)
        print(" ⚡ 파티션 성능 비교")
        print("="*60)
        
        query_text = "business strategy management"
        query_vectors = self.vector_utils.text_to_vector(query_text)
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
                    results = collection.search(
                        data=[query_vector],
                        anns_field="vector",
                        param={"metric_type": "L2", "params": {"ef": 100}},
                        limit=10,
                        partition_names=case["partitions"],
                        output_fields=["title"]
                    )
                else:
                    results = collection.search(
                        data=[query_vector],
                        anns_field="vector",
                        param={"metric_type": "L2", "params": {"ef": 100}},
                        limit=10,
                        output_fields=["title"]
                    )
                
                search_time = time.time() - start_time
                times.append(search_time)
                result_count = len(results[0])
            
            avg_time = np.mean(times)
            qps = 1 / avg_time
            
            print(f"{case['name']:<15} {avg_time:<12.4f} {qps:<8.2f} {result_count:<6} {case['description']:<20}")
    
    def partition_statistics(self, collection: Collection) -> None:
        """파티션 통계 정보"""
        print("\n" + "="*60)
        print(" 📊 파티션 통계 정보")
        print("="*60)
        
        partitions = collection.partitions
        
        print(f"{'파티션 이름':<20} {'엔티티 수':<10} {'상태':<10}")
        print("-" * 45)
        
        total_entities = 0
        for partition in partitions:
            try:
                # 파티션 통계 조회
                stats = partition.num_entities
                status = "로드됨" if partition.is_loaded else "언로드됨"
                
                print(f"{partition.name:<20} {stats:<10} {status:<10}")
                total_entities += stats
                
            except Exception as e:
                print(f"{partition.name:<20} {'오류':<10} {'N/A':<10}")
        
        print("-" * 45)
        print(f"{'총 엔티티 수':<20} {total_entities:<10}")
    
    def partition_management_demo(self, collection: Collection) -> None:
        """파티션 관리 데모"""
        print("\n" + "="*60)
        print(" 🛠️ 파티션 관리 데모")
        print("="*60)
        
        # 1. 새 파티션 생성
        print("\n1. 새 파티션 생성")
        new_partition_name = "category_entertainment"
        collection.create_partition(new_partition_name)
        print(f"  ✅ 파티션 '{new_partition_name}' 생성됨")
        
        # 2. 새 파티션에 데이터 삽입
        print("\n2. 새 파티션에 데이터 삽입")
        data = self.generate_partitioned_data("category", "entertainment", 500)
        insert_result = collection.insert(data, partition_name=new_partition_name)
        collection.flush()
        print(f"  ✅ {insert_result.insert_count}개 엔티티 삽입됨")
        
        # 3. 파티션 로드/언로드
        print("\n3. 파티션 로드/언로드 테스트")
        partition = collection.partition(new_partition_name)
        
        # 파티션 로드
        partition.load()
        print(f"  ✅ 파티션 '{new_partition_name}' 로드됨")
        
        # 파티션 정보 확인
        print(f"  엔티티 수: {partition.num_entities}")
        print(f"  로드 상태: 로드됨")
        
        # 4. 특정 파티션에서 검색
        print("\n4. 새 파티션에서 검색 테스트")
        query_text = "entertainment music movie"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 100}},
            limit=3,
            partition_names=[new_partition_name],
            output_fields=["title", "category"]
        )
        
        print(f"  검색 결과 수: {len(results[0])}")
        for i, hit in enumerate(results[0]):
            print(f"    {i+1}. {hit.entity.get('title')[:40]}...")
        
        # 5. 파티션 삭제
        print(f"\n5. 파티션 '{new_partition_name}' 삭제")
        # 먼저 파티션 언로드
        collection.release(partition_names=[new_partition_name])
        print(f"  파티션 언로드됨")
        collection.drop_partition(new_partition_name)
        print(f"  ✅ 파티션 삭제 완료")


def main():
    """메인 실행 함수"""
    print("🚀 Milvus 파티션 관리 실습 시작")
    print("실행 시간:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 파티션 매니저 생성
            partition_manager = PartitionManager(conn)
            
            # 1. 파티션 컬렉션 생성
            collection = partition_manager.create_partitioned_collection()
            
            # 2. 파티션 생성
            partitions = partition_manager.create_partitions(collection)
            
            # 3. 파티션 목록 조회
            partition_manager.list_partitions(collection)
            
            # 4. 파티션별 데이터 삽입
            partition_manager.insert_partitioned_data(collection)
            
            # 5. 인덱스 생성
            partition_manager.create_indexes(collection)
            
            # 6. 파티션 검색 데모
            partition_manager.partition_search_demo(collection)
            
            # 7. 파티션 성능 비교
            partition_manager.partition_performance_comparison(collection)
            
            # 8. 파티션 통계 정보
            partition_manager.partition_statistics(collection)
            
            # 9. 파티션 관리 데모
            partition_manager.partition_management_demo(collection)
            
            # 정리
            collection.drop()
            print("\n🧹 테스트 컬렉션 정리 완료")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 파티션 관리 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 파티션으로 데이터를 논리적으로 분리하여 검색 성능 향상")
    print("  • 카테고리, 지역, 시간 등 다양한 기준으로 파티션 구성")
    print("  • 특정 파티션만 검색하여 검색 범위 제한")
    print("  • 파티션별 독립적인 로드/언로드로 메모리 효율성 증대")
    print("  • 파티션 기반 데이터 관리로 운영 편의성 향상")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 