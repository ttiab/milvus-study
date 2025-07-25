#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔧 Milvus 고급 인덱싱 실습

이 스크립트는 Milvus의 고급 인덱싱 기법들을 실습합니다:
- 다양한 인덱스 타입 비교 (HNSW, IVF_PQ, IVF_SQ8, FLAT)
- GPU 인덱스 활용 (GPU_IVF_FLAT, GPU_IVF_PQ) 
- 복합 인덱스 및 하이브리드 검색
- 동적 인덱스 관리 (빌드, 드롭, 재빌드)
- 인덱스 성능 및 메모리 사용량 분석
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedIndexingManager:
    """고급 인덱싱 관리 클래스"""
    
    def __init__(self):
        self.milvus_conn = MilvusConnection()
        self.vector_utils = VectorUtils()
        self.index_performance_stats = {}
        
    def create_test_collection(self, collection_name: str, dim: int = 384) -> Collection:
        """테스트용 컬렉션 생성"""
        print(f"📁 테스트 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 고급 스키마 정의 (스칼라 필드 + 단일 벡터 필드)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="rating", dtype=DataType.FLOAT),
            FieldSchema(name="year", dtype=DataType.INT32),
            FieldSchema(name="is_premium", dtype=DataType.BOOL),
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Advanced indexing test collection with single vector field"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료 (단일 벡터 필드)")
        
        return collection
    
    def generate_test_data(self, size: int = 5000) -> List[List]:
        """테스트 데이터 생성"""
        print(f"📊 테스트 데이터 {size:,}개 생성 중...")
        
        # 텍스트 데이터 생성
        categories = ['technology', 'science', 'business', 'health', 'education', 'entertainment']
        titles = []
        contents = []
        
        for i in range(size):
            category = np.random.choice(categories)
            titles.append(f"{category.title()} Article {i}: Advanced concepts and applications")
            contents.append(f"This is a comprehensive {category} article about advanced concepts, "
                          f"methodologies, and practical applications in the field. "
                          f"Document ID: {i}, Category: {category}")
        
        # 벡터 생성
        print("  🔤 텍스트 벡터 생성 중...")
        text_vectors = self.vector_utils.texts_to_vectors(titles)
        
        # 메타데이터 생성
        category_list = [np.random.choice(categories) for _ in range(size)]
        prices = np.random.uniform(10.0, 1000.0, size)
        ratings = np.random.uniform(1.0, 5.0, size)
        years = np.random.randint(2020, 2025, size)
        is_premium = np.random.choice([True, False], size)
        view_counts = np.random.randint(100, 100000, size)
        
        # 데이터 구조화 (List[List] 형식)
        data = [
            titles,
            contents,
            category_list,
            prices.tolist(),
            ratings.tolist(),
            years.tolist(),
            is_premium.tolist(),
            view_counts.tolist(),
            text_vectors.tolist()
        ]
        
        print(f"  ✅ 데이터 생성 완료")
        return data
    
    def insert_test_data(self, collection: Collection, data: List[List]):
        """테스트 데이터 삽입"""
        print("💾 테스트 데이터 삽입 중...")
        
        start_time = time.time()
        collection.insert(data)
        collection.flush()
        insert_time = time.time() - start_time
        
        print(f"  ✅ 데이터 삽입 완료: {insert_time:.2f}초")
        return insert_time
    
    def create_index_with_timing(self, collection: Collection, field_name: str, 
                               index_params: Dict[str, Any], index_name: str) -> float:
        """인덱스 생성 및 시간 측정"""
        print(f"  🔍 {index_name} 인덱스 생성 중...")
        
        start_time = time.time()
        collection.create_index(
            field_name=field_name,
            index_params=index_params
        )
        build_time = time.time() - start_time
        
        print(f"    ✅ {index_name} 완료: {build_time:.2f}초")
        return build_time
    
    def compare_index_types(self, base_collection: Collection, test_data: List[List]) -> Dict[str, Any]:
        """다양한 인덱스 타입 비교"""
        print("\n🔧 다양한 인덱스 타입 비교...")
        
        index_configs = {
            "FLAT": {
                "metric_type": "COSINE",
                "index_type": "FLAT",
                "params": {}
            },
            "IVF_FLAT": {
                "metric_type": "COSINE", 
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            },
            "IVF_SQ8": {
                "metric_type": "COSINE",
                "index_type": "IVF_SQ8", 
                "params": {"nlist": 128}
            },
            "IVF_PQ": {
                "metric_type": "COSINE",
                "index_type": "IVF_PQ",
                "params": {
                    "nlist": 128,
                    "m": 16,      # PQ 코드북 수
                    "nbits": 8    # 각 코드북의 비트 수
                }
            },
            "HNSW": {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,              # 연결 수
                    "efConstruction": 200  # 구축 시 탐색 깊이
                }
            }
        }
        
        index_results = {}
        
        for index_name, params in index_configs.items():
            print(f"\n📋 {index_name} 인덱스 테스트:")
            
            # 각 인덱스 테스트용 새로운 컬렉션 생성
            test_collection_name = f"index_test_{index_name.lower()}"
            test_collection = self.create_test_collection(test_collection_name, 384)
            
            # 데이터 삽입
            test_collection.insert(test_data)
            test_collection.flush()
            
            # 인덱스 생성
            build_time = self.create_index_with_timing(
                test_collection, "text_vector", params, index_name
            )
            
            # 컬렉션 로드
            test_collection.load()
            
            # 검색 성능 테스트
            search_results = self.benchmark_index_search(test_collection, index_name, params)
            
            # 메모리 사용량 추정 (인덱스 타입별 특성 기반)
            memory_usage = self.estimate_index_memory(index_name, 5000, 384)
            
            index_results[index_name] = {
                "build_time": build_time,
                "search_performance": search_results,
                "estimated_memory_mb": memory_usage,
                "accuracy_vs_speed": self.get_index_characteristics(index_name)
            }
            
            test_collection.release()
            
            # 테스트 컬렉션 삭제
            utility.drop_collection(test_collection_name)
        
        return index_results
    
    def benchmark_index_search(self, collection: Collection, index_name: str, 
                             index_params: Dict[str, Any]) -> Dict[str, float]:
        """인덱스별 검색 성능 벤치마킹"""
        test_queries = [
            "machine learning artificial intelligence",
            "data science analytics research",
            "cloud computing distributed systems",
            "mobile application development framework",
            "cybersecurity network protection"
        ]
        
        search_times = []
        
        # 검색 파라미터 설정
        search_params = {"metric_type": "COSINE", "params": {}}
        
        if index_name == "IVF_FLAT" or index_name == "IVF_SQ8" or index_name == "IVF_PQ":
            search_params["params"]["nprobe"] = 16
        elif index_name == "HNSW":
            search_params["params"]["ef"] = 100
        
        # 검색 테스트
        for query in test_queries:
            query_vectors = self.vector_utils.text_to_vector(query)
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            start_time = time.time()
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="text_vector",
                param=search_params,
                limit=10,
                output_fields=["title", "category"]
            )
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        return {
            "avg_search_time": np.mean(search_times),
            "p95_search_time": np.percentile(search_times, 95),
            "qps": 1.0 / np.mean(search_times)
        }
    
    def estimate_index_memory(self, index_type: str, num_vectors: int, dim: int) -> float:
        """인덱스 메모리 사용량 추정 (MB)"""
        vector_size_mb = num_vectors * dim * 4 / (1024 * 1024)  # float32 기준
        
        if index_type == "FLAT":
            return vector_size_mb  # 원본 벡터만 저장
        elif index_type == "IVF_FLAT":
            return vector_size_mb * 1.1  # 10% 오버헤드
        elif index_type == "IVF_SQ8":
            return vector_size_mb * 0.3  # 압축으로 70% 절약
        elif index_type == "IVF_PQ":
            return vector_size_mb * 0.1  # PQ 압축으로 90% 절약
        elif index_type == "HNSW":
            return vector_size_mb * 1.5  # 그래프 구조로 50% 증가
        else:
            return vector_size_mb
    
    def get_index_characteristics(self, index_type: str) -> Dict[str, str]:
        """인덱스 특성 설명"""
        characteristics = {
            "FLAT": {"accuracy": "100%", "speed": "느림", "memory": "높음", "use_case": "정확도 최우선"},
            "IVF_FLAT": {"accuracy": "높음", "speed": "보통", "memory": "높음", "use_case": "균형적"},
            "IVF_SQ8": {"accuracy": "높음", "speed": "빠름", "memory": "보통", "use_case": "메모리 절약"},
            "IVF_PQ": {"accuracy": "보통", "speed": "매우빠름", "memory": "낮음", "use_case": "대용량 데이터"},
            "HNSW": {"accuracy": "높음", "speed": "매우빠름", "memory": "높음", "use_case": "실시간 검색"}
        }
        return characteristics.get(index_type, {})
    
    def gpu_index_demo(self, collection: Collection):
        """GPU 인덱스 데모 (GPU 사용 가능시)"""
        print("\n🚀 GPU 인덱스 데모...")
        
        # GPU 인덱스 설정들
        gpu_index_configs = {
            "GPU_IVF_FLAT": {
                "metric_type": "COSINE",
                "index_type": "GPU_IVF_FLAT",
                "params": {"nlist": 128}
            },
            "GPU_IVF_PQ": {
                "metric_type": "COSINE", 
                "index_type": "GPU_IVF_PQ",
                "params": {
                    "nlist": 128,
                    "m": 16,
                    "nbits": 8
                }
            }
        }
        
        print("  ⚠️  참고: GPU 인덱스는 GPU가 있는 환경에서만 사용 가능합니다.")
        print("  💡 GPU 인덱스의 장점:")
        print("    - 대용량 데이터셋에서 빠른 인덱스 구축")
        print("    - 병렬 처리로 검색 성능 향상")
        print("    - 메모리 대역폭 활용 최적화")
        
        # GPU 사용 가능 여부 확인 (시뮬레이션)
        gpu_available = False  # 실제 환경에서는 GPU 확인 로직 필요
        
        if gpu_available:
            print("\n  🎮 GPU 감지됨 - GPU 인덱스 생성 중...")
            for index_name, params in gpu_index_configs.items():
                try:
                    collection.drop_index()
                    build_time = self.create_index_with_timing(
                        collection, "text_vector", params, index_name
                    )
                    print(f"    ✅ {index_name} 구축 완료: {build_time:.2f}초")
                except Exception as e:
                    print(f"    ❌ {index_name} 실패: {e}")
        else:
            print("  ℹ️  GPU가 감지되지 않아 CPU 인덱스를 사용합니다.")
            print("  🔧 GPU 인덱스 사용 방법:")
            print("    1. NVIDIA GPU (CUDA 지원) 필요")
            print("    2. Milvus GPU 버전 설치")
            print("    3. GPU 메모리 충분히 확보")
    
    def hybrid_search_demo(self, collection: Collection):
        """하이브리드 검색 데모 (벡터 + 스칼라)"""
        print("\n🔍 하이브리드 검색 데모 (벡터 + 스칼라 필터링)...")
        
        # 인덱스 생성 (HNSW 사용)
        try:
            collection.drop_index(field_name="text_vector")
        except:
            pass
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW", 
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("text_vector", index_params)
        collection.load()
        
        # 하이브리드 검색 시나리오들
        search_scenarios = [
            {
                "name": "고급 기술 문서 (평점 4.0 이상)",
                "query": "advanced technology artificial intelligence",
                "filter": "category == 'technology' and rating >= 4.0"
            },
            {
                "name": "프리미엄 비즈니스 콘텐츠 (2023년 이후)",
                "query": "business strategy management innovation",
                "filter": "is_premium == True and year >= 2023"
            },
            {
                "name": "인기 건강 정보 (조회수 5만 이상)",
                "query": "health medical research treatment",
                "filter": "category == 'health' and view_count >= 50000"
            },
            {
                "name": "저가 교육 자료 (100달러 미만)",
                "query": "education learning online course",
                "filter": "category == 'education' and price < 100.0"
            }
        ]
        
        for scenario in search_scenarios:
            print(f"\n  📋 시나리오: {scenario['name']}")
            print(f"    쿼리: '{scenario['query']}'")
            print(f"    필터: {scenario['filter']}")
            
            # 벡터 생성
            query_vectors = self.vector_utils.text_to_vector(scenario['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 하이브리드 검색 실행
            start_time = time.time()
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 100}},
                limit=5,
                expr=scenario['filter'],
                output_fields=["title", "category", "rating", "price", "year", "is_premium", "view_count"]
            )
            search_time = time.time() - start_time
            
            print(f"    검색 시간: {search_time*1000:.2f}ms")
            print(f"    결과 수: {len(results[0]) if results and results[0] else 0}")
            
            if results and len(results[0]) > 0:
                print("    상위 결과:")
                for i, hit in enumerate(results[0][:3], 1):
                    entity = hit.entity
                    print(f"      {i}. {entity.get('title')[:50]}...")
                    print(f"         카테고리: {entity.get('category')}, 평점: {entity.get('rating'):.1f}")
                    print(f"         가격: ${entity.get('price'):.2f}, 유사도: {hit.distance:.3f}")
            else:
                print("    ⚠️  조건에 맞는 결과가 없습니다.")
    
    def dynamic_index_management(self, collection: Collection):
        """동적 인덱스 관리 데모"""
        print("\n🔄 동적 인덱스 관리 데모...")
        
        print("  📋 인덱스 생성 → 성능 측정 → 최적화 → 재구축 과정")
        
        # 1. 초기 인덱스 (IVF_FLAT)
        print("\n  1️⃣ 초기 인덱스: IVF_FLAT")
        try:
            collection.drop_index(field_name="text_vector")
            import time
            while collection.has_index():
                print("  ⏳ 인덱스 삭제 대기 중...")
                time.sleep(10)
            print("  ✅ 기존 인덱스 삭제 완료")
        except:
            pass
        
        initial_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 64}  # 작은 nlist로 시작
        }
        
        build_time_1 = self.create_index_with_timing(
            collection, "text_vector", initial_params, "IVF_FLAT (nlist=64)"
        )
        

        
        collection.load()
        perf_1 = self.benchmark_index_search(collection, "IVF_FLAT", initial_params)
        print(f"    성능: 평균 {perf_1['avg_search_time']*1000:.2f}ms, QPS: {perf_1['qps']:.1f}")
        collection.release()
        
        # 2. 최적화된 인덱스 (nlist 증가)
        print("\n  2️⃣ 최적화된 인덱스: IVF_FLAT (nlist 증가)")
        try:
            collection.drop_index(field_name="text_vector")
        except:
            pass
        
        optimized_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 128}  # nlist 증가
        }
        
        build_time_2 = self.create_index_with_timing(
            collection, "text_vector", optimized_params, "IVF_FLAT (nlist=128)"
        )
        

        
        collection.load()
        perf_2 = self.benchmark_index_search(collection, "IVF_FLAT", optimized_params)
        print(f"    성능: 평균 {perf_2['avg_search_time']*1000:.2f}ms, QPS: {perf_2['qps']:.1f}")
        collection.release()
        
        # 3. 고성능 인덱스 (HNSW)
        print("\n  3️⃣ 고성능 인덱스: HNSW")
        try:
            collection.drop_index(field_name="text_vector")
        except:
            pass
        
        hnsw_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        build_time_3 = self.create_index_with_timing(
            collection, "text_vector", hnsw_params, "HNSW"
        )
        

        
        collection.load()
        perf_3 = self.benchmark_index_search(collection, "HNSW", hnsw_params)
        print(f"    성능: 평균 {perf_3['avg_search_time']*1000:.2f}ms, QPS: {perf_3['qps']:.1f}")
        
        # 성능 비교 결과
        print("\n  📊 성능 비교 결과:")
        print(f"    IVF_FLAT (nlist=64):  구축 {build_time_1:.2f}s, 검색 {perf_1['avg_search_time']*1000:.2f}ms")
        print(f"    IVF_FLAT (nlist=128): 구축 {build_time_2:.2f}s, 검색 {perf_2['avg_search_time']*1000:.2f}ms") 
        print(f"    HNSW:                구축 {build_time_3:.2f}s, 검색 {perf_3['avg_search_time']*1000:.2f}ms")
        
        improvement = perf_1['avg_search_time'] / perf_3['avg_search_time']
        print(f"    💡 HNSW는 초기 대비 {improvement:.1f}x 성능 향상!")
        
        collection.release()
    
    def vector_search_demo(self, collection: Collection):
        """벡터 검색 최적화 데모"""
        print("\n🔍 벡터 검색 최적화 데모...")
        
        # 현재 HNSW 인덱스로 다양한 검색 패턴 테스트
        print("  📋 다양한 검색 매개변수로 성능 테스트")
        
        # 컬렉션 로드
        collection.load()
        
        # 검색 테스트 시나리오
        test_scenarios = [
            {"name": "정확도 우선 (ef=200)", "ef": 200},
            {"name": "균형 (ef=100)", "ef": 100}, 
            {"name": "속도 우선 (ef=50)", "ef": 50}
        ]
        
        query_text = "artificial intelligence machine learning"
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        print(f"\n  🔍 검색 쿼리: '{query_text}'")
        
        for scenario in test_scenarios:
            print(f"\n  📊 {scenario['name']}:")
            
            # 5회 측정하여 평균 계산
            times = []
            for _ in range(5):
                start_time = time.time()
                results = collection.search(
                    data=[query_vector.tolist()],
                    anns_field="text_vector",
                    param={"metric_type": "COSINE", "params": {"ef": scenario['ef']}},
                    limit=10,
                    output_fields=["title", "category"]
                )
                search_time = time.time() - start_time
                times.append(search_time)
            
            avg_time = np.mean(times)
            print(f"    평균 검색 시간: {avg_time*1000:.2f}ms")
            print(f"    QPS: {1/avg_time:.1f}")
            
            if results and len(results[0]) > 0:
                print(f"    상위 결과 유사도: {results[0][0].distance:.3f}")
        
        collection.release()
    
    def run_advanced_indexing_demo(self):
        """고급 인덱싱 종합 데모"""
        print("🔧 Milvus 고급 인덱싱 실습")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Milvus 연결
            self.milvus_conn.connect()
            print("✅ Milvus 연결 성공\n")
            
            print("=" * 80)
            print(" 🏗️ 테스트 환경 구축")
            print("=" * 80)
            
            # 테스트 컬렉션 생성
            collection = self.create_test_collection("advanced_indexing_test")
            
            # 테스트 데이터 생성 및 삽입
            test_data = self.generate_test_data(5000)
            self.insert_test_data(collection, test_data)
            
            print("\n" + "=" * 80)
            print(" 🔧 인덱스 타입 비교 분석")
            print("=" * 80)
            
            # 다양한 인덱스 타입 비교
            index_comparison = self.compare_index_types(collection, test_data)
            
            print("\n📊 인덱스 성능 비교 결과:")
            print(f"{'인덱스':<12} {'구축시간(s)':<12} {'검색시간(ms)':<12} {'QPS':<8} {'메모리(MB)':<10} {'특징'}")
            print("-" * 80)
            
            for index_name, stats in index_comparison.items():
                build_time = stats['build_time']
                search_time = stats['search_performance']['avg_search_time'] * 1000
                qps = stats['search_performance']['qps']
                memory = stats['estimated_memory_mb']
                characteristics = stats['accuracy_vs_speed']
                use_case = characteristics.get('use_case', 'N/A')
                
                print(f"{index_name:<12} {build_time:<12.2f} {search_time:<12.2f} {qps:<8.1f} {memory:<10.1f} {use_case}")
            
            print("\n" + "=" * 80)
            print(" 🚀 GPU 인덱스 데모")
            print("=" * 80)
            
            # GPU 인덱스 데모
            self.gpu_index_demo(collection)
            
            print("\n" + "=" * 80)
            print(" 🔍 하이브리드 검색 (벡터 + 스칼라)")
            print("=" * 80)
            
            # 하이브리드 검색 데모
            self.hybrid_search_demo(collection)
            
            print("\n" + "=" * 80)
            print(" 🔄 동적 인덱스 관리")
            print("=" * 80)
            
            # 동적 인덱스 관리 데모
            self.dynamic_index_management(collection)
            
            print("\n" + "=" * 80)
            print(" 🔍 벡터 검색 최적화")
            print("=" * 80)
            
            # 벡터 검색 최적화 데모
            self.vector_search_demo(collection)
            
            print("\n" + "=" * 80)
            print(" 💡 고급 인덱싱 권장사항")
            print("=" * 80)
            
            print("\n🎯 인덱스 선택 가이드:")
            print("  📊 데이터 크기별:")
            print("    • < 1M 벡터: HNSW (최고 성능)")
            print("    • 1M - 10M 벡터: IVF_FLAT 또는 IVF_SQ8")
            print("    • > 10M 벡터: IVF_PQ (메모리 효율)")
            
            print("\n  🎯 용도별:")
            print("    • 실시간 검색: HNSW")
            print("    • 배치 처리: IVF_FLAT")
            print("    • 메모리 제약: IVF_PQ 또는 IVF_SQ8")
            print("    • 정확도 최우선: FLAT")
            
            print("\n  ⚡ 성능 튜닝:")
            print("    • IVF 계열: nlist = sqrt(num_vectors)")
            print("    • HNSW: M=16-32, efConstruction=200-400")
            print("    • 검색 시: nprobe=nlist/8, ef=limit*2")
            
            print("\n  🔧 하이브리드 검색:")
            print("    • 벡터 유사도 + 스칼라 필터링 조합")
            print("    • 필터링 먼저 vs 벡터 검색 먼저 선택")
            print("    • 인덱스 설계 시 필터링 패턴 고려")
            
            # 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            utility.drop_collection("advanced_indexing_test")
            print("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")
        
        finally:
            self.milvus_conn.disconnect()
            
        print(f"\n🎉 고급 인덱싱 실습 완료!")
        print("\n💡 학습 포인트:")
        print("  • 다양한 인덱스 타입의 특성과 적합한 사용 사례")
        print("  • 성능과 메모리 사용량의 트레이드오프 이해")
        print("  • 하이브리드 검색으로 정확도 향상")
        print("  • 동적 인덱스 관리로 최적화 전략 수립")
        print("\n🚀 다음 단계:")
        print("  python step04_advanced/03_distributed_scaling.py")

def main():
    """메인 실행 함수"""
    indexing_manager = AdvancedIndexingManager()
    indexing_manager.run_advanced_indexing_demo()

if __name__ == "__main__":
    main() 