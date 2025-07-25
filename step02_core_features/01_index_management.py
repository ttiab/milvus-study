#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 2단계: 인덱스 관리 실습

이 스크립트는 Milvus의 다양한 인덱스 알고리즘과 최적화를 학습합니다.
- IVF_FLAT, IVF_SQ8, IVF_PQ 인덱스
- HNSW 인덱스
- 인덱스 성능 비교
- 파라미터 튜닝
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


class IndexManager:
    """Milvus 인덱스 관리 클래스"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        
    def create_test_collection(self, name: str, dimension: int = 384) -> Collection:
        """테스트용 컬렉션 생성"""
        print(f"\n📁 컬렉션 '{name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(name):
            utility.drop_collection(name)
            print(f"  기존 컬렉션 '{name}' 삭제됨")
        
        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="score", dtype=DataType.FLOAT)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Index 테스트용 컬렉션: {name}"
        )
        
        collection = Collection(name=name, schema=schema)
        print(f"  ✅ 컬렉션 '{name}' 생성 완료")
        return collection
    
    def generate_test_data(self, count: int = 10000) -> List[List]:
        """테스트 데이터 생성"""
        print(f"\n📊 테스트 데이터 {count}개 생성 중...")
        
        # 텍스트 데이터 생성
        texts = [
            f"Sample document {i} about various topics like technology, science, and business"
            for i in range(count)
        ]
        
        # 벡터 변환
        print("  벡터 변환 중...")
        vectors = self.vector_utils.texts_to_vectors(texts)
        
        # 카테고리와 점수 생성
        categories = ["tech", "science", "business", "health", "education"]
        data = [
            texts,
            vectors.tolist(),
            [categories[i % len(categories)] for i in range(count)],
            [np.random.uniform(0, 10) for _ in range(count)]
        ]
        
        print(f"  ✅ 테스트 데이터 생성 완료 (벡터 차원: {vectors.shape[1]})")
        return data
    
    def insert_data(self, collection: Collection, data: List[List]) -> None:
        """데이터 삽입"""
        print(f"\n💾 데이터 삽입 중...")
        start_time = time.time()
        
        insert_result = collection.insert(data)
        print(f"  삽입된 엔티티 수: {insert_result.insert_count}")
        
        # 메모리에 flush
        collection.flush()
        print(f"  ✅ 데이터 삽입 완료 ({time.time() - start_time:.2f}초)")
    
    def create_index_ivf_flat(self, collection: Collection, nlist: int = 1024) -> None:
        """IVF_FLAT 인덱스 생성"""
        print(f"\n🔍 IVF_FLAT 인덱스 생성 중 (nlist={nlist})...")
        start_time = time.time()
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": nlist}
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        print(f"  ✅ IVF_FLAT 인덱스 생성 완료 ({time.time() - start_time:.2f}초)")
    
    def create_index_ivf_sq8(self, collection: Collection, nlist: int = 1024) -> None:
        """IVF_SQ8 인덱스 생성"""
        print(f"\n🔍 IVF_SQ8 인덱스 생성 중 (nlist={nlist})...")
        start_time = time.time()
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_SQ8",
            "params": {"nlist": nlist}
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        print(f"  ✅ IVF_SQ8 인덱스 생성 완료 ({time.time() - start_time:.2f}초)")
    
    def create_index_hnsw(self, collection: Collection, M: int = 16, efConstruction: int = 200) -> None:
        """HNSW 인덱스 생성"""
        print(f"\n🔍 HNSW 인덱스 생성 중 (M={M}, efConstruction={efConstruction})...")
        start_time = time.time()
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": M,
                "efConstruction": efConstruction
            }
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        print(f"  ✅ HNSW 인덱스 생성 완료 ({time.time() - start_time:.2f}초)")
    
    def benchmark_search(self, collection: Collection, query_vectors: np.ndarray, 
                        top_k: int = 10, nprobe: int = 10) -> Dict[str, float]:
        """검색 성능 벤치마크"""
        print(f"\n⚡ 검색 성능 테스트 (top_k={top_k}, nprobe={nprobe})...")
        
        # 컬렉션 로드
        collection.load()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": nprobe}
        }
        
        # 여러 번 검색하여 평균 시간 측정
        times = []
        for i in range(5):
            start_time = time.time()
            
            results = collection.search(
                data=query_vectors[:10],  # 10개 쿼리로 테스트
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "category", "score"]
            )
            
            search_time = time.time() - start_time
            times.append(search_time)
            print(f"  검색 {i+1}: {search_time:.4f}초")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  📊 평균 검색 시간: {avg_time:.4f}초 (±{std_time:.4f})")
        print(f"  📊 QPS: {10/avg_time:.2f}")
        
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "qps": 10/avg_time
        }
    
    def compare_indexes(self) -> None:
        """다양한 인덱스 알고리즘 비교"""
        print("\n" + "="*60)
        print(" 🔬 인덱스 알고리즘 성능 비교")
        print("="*60)
        
        # 테스트 데이터 준비
        test_data = self.generate_test_data(10000)
        query_texts = [
            "technology innovation and development",
            "scientific research methodology",
            "business strategy and planning"
        ]
        query_vectors = self.vector_utils.texts_to_vectors(query_texts)
        
        results = {}
        
        # 1. IVF_FLAT 테스트
        print("\n🧪 IVF_FLAT 인덱스 테스트")
        collection1 = self.create_test_collection("test_ivf_flat")
        self.insert_data(collection1, test_data)
        self.create_index_ivf_flat(collection1, nlist=128)
        results["IVF_FLAT"] = self.benchmark_search(collection1, query_vectors)
        
        # 2. IVF_SQ8 테스트
        print("\n🧪 IVF_SQ8 인덱스 테스트")
        collection2 = self.create_test_collection("test_ivf_sq8")
        self.insert_data(collection2, test_data)
        self.create_index_ivf_sq8(collection2, nlist=128)
        results["IVF_SQ8"] = self.benchmark_search(collection2, query_vectors)
        
        # 3. HNSW 테스트
        print("\n🧪 HNSW 인덱스 테스트")
        collection3 = self.create_test_collection("test_hnsw")
        self.insert_data(collection3, test_data)
        self.create_index_hnsw(collection3, M=16, efConstruction=200)
        results["HNSW"] = self.benchmark_search(collection3, query_vectors)
        
        # 결과 요약
        print("\n" + "="*60)
        print(" 📊 성능 비교 결과")
        print("="*60)
        print(f"{'인덱스 타입':<15} {'평균시간(초)':<12} {'QPS':<10} {'표준편차':<10}")
        print("-" * 60)
        
        for index_type, metrics in results.items():
            print(f"{index_type:<15} {metrics['avg_time']:<12.4f} {metrics['qps']:<10.2f} {metrics['std_time']:<10.4f}")
        
        # 정리
        collection1.drop()
        collection2.drop()
        collection3.drop()
        print("\n🧹 테스트 컬렉션 정리 완료")
    
    def tune_parameters(self) -> None:
        """인덱스 파라미터 튜닝 실습"""
        print("\n" + "="*60)
        print(" 🎛️ 인덱스 파라미터 튜닝")
        print("="*60)
        
        # 테스트 데이터 준비
        test_data = self.generate_test_data(5000)
        query_texts = ["technology and innovation"]
        query_vectors = self.vector_utils.texts_to_vectors(query_texts)
        
        # HNSW 파라미터 튜닝
        print("\n🔧 HNSW 파라미터 튜닝")
        hnsw_params = [
            {"M": 8, "efConstruction": 100},
            {"M": 16, "efConstruction": 200},
            {"M": 32, "efConstruction": 400}
        ]
        
        for i, params in enumerate(hnsw_params):
            print(f"\n  테스트 {i+1}: M={params['M']}, efConstruction={params['efConstruction']}")
            
            collection = self.create_test_collection(f"tune_hnsw_{i}")
            self.insert_data(collection, test_data)
            self.create_index_hnsw(collection, **params)
            
            # 빠른 성능 테스트
            collection.load()
            start_time = time.time()
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param={"metric_type": "L2", "params": {"ef": 100}},
                limit=10
            )
            search_time = time.time() - start_time
            
            print(f"    검색 시간: {search_time:.4f}초")
            collection.drop()
        
        print("\n✅ 파라미터 튜닝 완료")


def main():
    """메인 실행 함수"""
    print("🚀 Milvus 인덱스 관리 실습 시작")
    print("실행 시간:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 인덱스 매니저 생성
            index_manager = IndexManager(conn)
            
            # 1. 인덱스 알고리즘 비교
            index_manager.compare_indexes()
            
            # 2. 파라미터 튜닝
            index_manager.tune_parameters()
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 인덱스 관리 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • IVF_FLAT: 높은 정확도, 많은 메모리 사용")
    print("  • IVF_SQ8: 메모리 효율적, 약간의 정확도 손실")
    print("  • HNSW: 빠른 검색, 그래프 기반 알고리즘")
    print("  • 파라미터 튜닝으로 성능과 정확도의 균형 조절")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 