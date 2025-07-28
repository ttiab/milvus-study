#!/usr/bin/env python3
"""
Milvus 인덱스 타입 비교 실습

이 스크립트는 다양한 인덱스 타입의 성능, 정확도, 메모리 사용량을 실제로 비교합니다.
"""

import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility
)
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndexPerformance:
    """인덱스 성능 메트릭 클래스"""
    index_type: str
    build_time: float
    search_time: float
    memory_usage: float
    accuracy: float
    index_size: int

class IndexComparator:
    """인덱스 타입 비교 클래스"""
    
    def __init__(self, dim: int = 128, num_vectors: int = 10000):
        self.dim = dim
        self.num_vectors = num_vectors
        self.collection_name = "index_comparison"
        self.test_data = None
        self.query_vectors = None
        self.ground_truth = None
        
        # 테스트할 인덱스 설정
        self.index_configs = {
            "FLAT": {
                "index_type": "FLAT",
                "metric_type": "L2",
                "params": {}
            },
            "IVF_FLAT": {
                "index_type": "IVF_FLAT", 
                "metric_type": "L2",
                "params": {"nlist": 128}
            },
            "IVF_SQ8": {
                "index_type": "IVF_SQ8",
                "metric_type": "L2", 
                "params": {"nlist": 128}
            },
            "HNSW": {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {"M": 16, "efConstruction": 200}
            }
        }
        
        # IVF_PQ는 IP 메트릭만 지원하므로 별도 처리
        self.ivf_pq_config = {
            "IVF_PQ": {
                "index_type": "IVF_PQ",
                "metric_type": "IP",
                "params": {"nlist": 128, "m": 8, "nbits": 8}
            }
        }
        
        self.search_configs = {
            "FLAT": {"metric_type": "L2", "params": {}},
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 16}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 16}}, 
            "HNSW": {"metric_type": "L2", "params": {"ef": 64}},
            "IVF_PQ": {"metric_type": "IP", "params": {"nprobe": 16}}
        }
    
    def connect_to_milvus(self) -> bool:
        """Milvus 연결"""
        try:
            connections.connect("default", host="localhost", port="19530")
            logger.info("Milvus에 성공적으로 연결되었습니다.")
            return True
        except Exception as e:
            logger.error(f"Milvus 연결 실패: {e}")
            return False
    
    def generate_test_data(self):
        """테스트 데이터 생성"""
        logger.info(f"{self.num_vectors}개의 {self.dim}차원 벡터 생성 중...")
        
        # 정규분포 기반 랜덤 벡터 생성
        np.random.seed(42)  # 재현 가능한 결과를 위해
        self.test_data = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        
        # 정규화 (코사인 유사도나 내적을 위해)
        norms = np.linalg.norm(self.test_data, axis=1, keepdims=True)
        self.test_data = self.test_data / norms
        
        # 쿼리 벡터 생성 (테스트 데이터의 부분집합 + 새로운 벡터)
        num_queries = 100
        query_indices = np.random.choice(self.num_vectors, num_queries // 2, replace=False)
        known_queries = self.test_data[query_indices]
        
        new_queries = np.random.randn(num_queries // 2, self.dim).astype(np.float32)
        new_queries = new_queries / np.linalg.norm(new_queries, axis=1, keepdims=True)
        
        self.query_vectors = np.vstack([known_queries, new_queries])
        
        logger.info(f"테스트 데이터 생성 완료: {self.test_data.shape}, 쿼리: {self.query_vectors.shape}")
    
    def compute_ground_truth(self):
        """정확도 평가를 위한 정답 계산 (FLAT 인덱스 사용)"""
        logger.info("정답 계산 중...")
        
        # 브루트 포스로 정확한 결과 계산
        distances = []
        indices = []
        
        for query in self.query_vectors:
            # L2 거리 계산
            dists = np.linalg.norm(self.test_data - query, axis=1)
            sorted_indices = np.argsort(dists)[:10]  # top 10
            
            distances.append(dists[sorted_indices])
            indices.append(sorted_indices)
        
        self.ground_truth = {
            'distances': distances,
            'indices': indices
        }
        
        logger.info("정답 계산 완료")
    
    def create_collection(self, metric_type: str = "L2"):
        """컬렉션 생성"""
        # 기존 컬렉션 삭제
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "인덱스 비교 테스트 컬렉션")
        
        # 컬렉션 생성
        collection = Collection(self.collection_name, schema)
        
        # 데이터 삽입
        logger.info("데이터 삽입 중...")
        entities = [self.test_data.tolist()]
        collection.insert(entities)
        collection.flush()
        
        logger.info("데이터 삽입 완료")
        return collection
    
    def measure_memory_usage(self) -> float:
        """현재 메모리 사용량 측정 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def measure_index_performance(self, collection: Collection, index_config: Dict, 
                                search_config: Dict, index_name: str) -> IndexPerformance:
        """특정 인덱스의 성능 측정"""
        logger.info(f"{index_name} 인덱스 성능 측정 시작")
        
        # 메모리 사용량 측정 시작
        memory_before = self.measure_memory_usage()
        
        # 인덱스 구축 시간 측정
        start_time = time.time()
        collection.create_index("embedding", index_config)
        build_time = time.time() - start_time
        
        # 인덱스 구축 후 메모리 사용량
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before
        
        # 컬렉션 로드
        collection.load()
        
        # 인덱스 크기 정보 (추정)
        index_info = collection.index()
        
        # 검색 시간 측정
        start_time = time.time()
        results = collection.search(
            data=self.query_vectors.tolist(),
            anns_field="embedding",
            param=search_config,
            limit=10
        )
        search_time = time.time() - start_time
        
        # 정확도 계산 (Recall@10)
        accuracy = self.calculate_recall(results, index_name)
        
        # 컬렉션 해제
        collection.release()
        
        # 인덱스 삭제
        collection.drop_index()
        
        performance = IndexPerformance(
            index_type=index_name,
            build_time=build_time,
            search_time=search_time,
            memory_usage=memory_usage,
            accuracy=accuracy,
            index_size=0  # 정확한 크기 측정이 어려워 0으로 설정
        )
        
        logger.info(f"{index_name} 성능 측정 완료")
        return performance
    
    def calculate_recall(self, search_results: List, index_name: str) -> float:
        """Recall@10 계산"""
        if not self.ground_truth:
            return 0.0
        
        total_recall = 0.0
        
        for i, result in enumerate(search_results):
            if i >= len(self.ground_truth['indices']):
                break
                
            true_indices = set(self.ground_truth['indices'][i])
            pred_indices = set(hit.id for hit in result)
            
            if len(true_indices) > 0:
                recall = len(true_indices.intersection(pred_indices)) / len(true_indices)
                total_recall += recall
        
        avg_recall = total_recall / len(search_results) if search_results else 0.0
        return avg_recall
    
    def run_comparison(self) -> List[IndexPerformance]:
        """모든 인덱스 타입 비교 실행"""
        if not self.connect_to_milvus():
            logger.error("Milvus 연결 실패로 비교를 중단합니다.")
            return []
        
        # 테스트 데이터 준비
        self.generate_test_data()
        self.compute_ground_truth()
        
        results = []
        
        # L2 메트릭 인덱스들 테스트
        collection = self.create_collection("L2")
        
        for index_name, index_config in self.index_configs.items():
            try:
                performance = self.measure_index_performance(
                    collection, index_config, 
                    self.search_configs[index_name], index_name
                )
                results.append(performance)
                
                # 결과 출력
                print(f"\n{index_name} 결과:")
                print(f"  구축 시간: {performance.build_time:.2f}초")
                print(f"  검색 시간: {performance.search_time:.4f}초")
                print(f"  메모리 사용량: {performance.memory_usage:.2f}MB")
                print(f"  정확도 (Recall@10): {performance.accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"{index_name} 테스트 중 오류: {e}")
        
        # IVF_PQ 테스트 (IP 메트릭 필요)
        try:
            # IP 메트릭용 컬렉션 생성
            utility.drop_collection(self.collection_name)
            collection = self.create_collection("IP")
            
            performance = self.measure_index_performance(
                collection, self.ivf_pq_config["IVF_PQ"],
                self.search_configs["IVF_PQ"], "IVF_PQ"
            )
            results.append(performance)
            
            print(f"\nIVF_PQ 결과:")
            print(f"  구축 시간: {performance.build_time:.2f}초")
            print(f"  검색 시간: {performance.search_time:.4f}초") 
            print(f"  메모리 사용량: {performance.memory_usage:.2f}MB")
            print(f"  정확도 (Recall@10): {performance.accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"IVF_PQ 테스트 중 오류: {e}")
        
        # 정리
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        return results
    
    def visualize_results(self, results: List[IndexPerformance]):
        """결과 시각화"""
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 결과 데이터 준비
            index_names = [r.index_type for r in results]
            build_times = [r.build_time for r in results]
            search_times = [r.search_time * 1000 for r in results]  # ms로 변환
            memory_usage = [r.memory_usage for r in results]
            accuracies = [r.accuracy for r in results]
            
            # 4개 서브플롯 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 구축 시간
            axes[0, 0].bar(index_names, build_times, color='skyblue')
            axes[0, 0].set_title('인덱스 구축 시간')
            axes[0, 0].set_ylabel('시간 (초)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 검색 시간  
            axes[0, 1].bar(index_names, search_times, color='lightgreen')
            axes[0, 1].set_title('검색 시간')
            axes[0, 1].set_ylabel('시간 (ms)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 메모리 사용량
            axes[1, 0].bar(index_names, memory_usage, color='lightcoral')
            axes[1, 0].set_title('메모리 사용량')
            axes[1, 0].set_ylabel('메모리 (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. 정확도
            axes[1, 1].bar(index_names, accuracies, color='gold')
            axes[1, 1].set_title('정확도 (Recall@10)')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('index_comparison_results.png', dpi=300, bbox_inches='tight')
            print("\n그래프가 'index_comparison_results.png'로 저장되었습니다.")
            plt.show()
            
        except ImportError:
            print("matplotlib이 설치되지 않아 시각화를 건너뜁니다.")
    
    def print_summary_table(self, results: List[IndexPerformance]):
        """결과 요약 테이블 출력"""
        if not results:
            return
        
        print("\n" + "="*80)
        print("인덱스 성능 비교 요약")
        print("="*80)
        
        # 헤더
        header = f"{'인덱스':<12} {'구축시간(s)':<12} {'검색시간(ms)':<14} {'메모리(MB)':<12} {'정확도':<10}"
        print(header)
        print("-" * 80)
        
        # 데이터 행
        for result in results:
            row = f"{result.index_type:<12} {result.build_time:<12.2f} {result.search_time*1000:<14.2f} {result.memory_usage:<12.2f} {result.accuracy:<10.4f}"
            print(row)
        
        print("-" * 80)
        
        # 순위 정보
        print("\n📊 성능 순위:")
        
        # 속도 순위 (검색 시간 기준)
        speed_ranking = sorted(results, key=lambda x: x.search_time)
        print("🚀 검색 속도 순위:")
        for i, result in enumerate(speed_ranking, 1):
            print(f"  {i}. {result.index_type}: {result.search_time*1000:.2f}ms")
        
        # 정확도 순위
        accuracy_ranking = sorted(results, key=lambda x: x.accuracy, reverse=True)
        print("\n🎯 정확도 순위:")
        for i, result in enumerate(accuracy_ranking, 1):
            print(f"  {i}. {result.index_type}: {result.accuracy:.4f}")
        
        # 메모리 효율성 순위
        memory_ranking = sorted(results, key=lambda x: x.memory_usage)
        print("\n💾 메모리 효율성 순위:")
        for i, result in enumerate(memory_ranking, 1):
            print(f"  {i}. {result.index_type}: {result.memory_usage:.2f}MB")

def run_parameter_tuning_demo():
    """파라미터 튜닝 데모"""
    print("\n" + "="*60)
    print("IVF_FLAT 파라미터 튜닝 데모")
    print("="*60)
    
    # 다양한 nlist, nprobe 조합 테스트
    nlist_values = [64, 128, 256, 512]
    nprobe_values = [1, 4, 8, 16, 32]
    
    print("\nnlist와 nprobe의 영향:")
    print("nlist ↑ → 구축시간 ↑, 메모리 ↑")
    print("nprobe ↑ → 검색시간 ↑, 정확도 ↑")
    
    print(f"\n{'nlist':<8} {'nprobe':<8} {'예상 구축시간':<15} {'예상 검색시간':<15} {'예상 정확도':<12}")
    print("-" * 70)
    
    for nlist in nlist_values:
        for nprobe in [min(nprobe_values), max(nprobe_values)]:
            # 경험적 공식으로 성능 추정
            build_time_factor = nlist / 128  # 기준: nlist=128
            search_time_factor = nprobe / 16  # 기준: nprobe=16
            accuracy_factor = min(1.0, 0.7 + (nprobe / nlist) * 0.3)
            
            print(f"{nlist:<8} {nprobe:<8} {build_time_factor:<15.2f} {search_time_factor:<15.2f} {accuracy_factor:<12.3f}")

def print_selection_guide():
    """인덱스 선택 가이드 출력"""
    print("\n" + "="*60)
    print("🎯 인덱스 선택 가이드")
    print("="*60)
    
    scenarios = {
        "소규모 프로토타입 (< 1만 벡터)": {
            "추천": "FLAT",
            "이유": "간단하고 정확하며 설정이 필요 없음"
        },
        "일반적인 웹 서비스 (10만~100만 벡터)": {
            "추천": "IVF_FLAT",
            "이유": "균형잡힌 성능과 정확도"
        },
        "실시간 검색 서비스": {
            "추천": "HNSW", 
            "이유": "가장 빠른 검색 속도"
        },
        "메모리 제약 환경": {
            "추천": "IVF_SQ8",
            "이유": "75% 메모리 절약"
        },
        "대용량 데이터 (수억 벡터)": {
            "추천": "IVF_PQ",
            "이유": "최대 압축률과 확장성"
        },
        "최고 정확도 요구": {
            "추천": "FLAT",
            "이유": "100% 정확도 보장"
        }
    }
    
    for scenario, info in scenarios.items():
        print(f"\n📌 {scenario}")
        print(f"   권장 인덱스: {info['추천']}")
        print(f"   선택 이유: {info['이유']}")

def main():
    """메인 함수"""
    print("Milvus 인덱스 타입 비교 실습 시작")
    print("="*60)
    
    print("\n이 실습에서는 다음을 비교합니다:")
    print("- FLAT: 브루트 포스 (100% 정확)")
    print("- IVF_FLAT: 균형잡힌 성능")
    print("- IVF_SQ8: 메모리 효율적")
    print("- HNSW: 빠른 검색")
    print("- IVF_PQ: 최대 압축")
    
    # 사용자 입력
    try:
        num_vectors = int(input(f"\n테스트할 벡터 개수를 입력하세요 (기본값: 10000): ") or "10000")
        dim = int(input(f"벡터 차원을 입력하세요 (기본값: 128): ") or "128")
    except ValueError:
        num_vectors, dim = 10000, 128
        print("기본값을 사용합니다.")
    
    # 비교 실행
    comparator = IndexComparator(dim=dim, num_vectors=num_vectors)
    results = comparator.run_comparison()
    
    if results:
        # 결과 출력
        comparator.print_summary_table(results)
        
        # 시각화
        comparator.visualize_results(results)
        
        # 파라미터 튜닝 데모
        run_parameter_tuning_demo()
        
        # 선택 가이드
        print_selection_guide()
        
        print("\n" + "="*60)
        print("실습 완료!")
        print("="*60)
        print("\n주요 학습 포인트:")
        print("1. 각 인덱스는 서로 다른 트레이드오프를 가짐")
        print("2. 데이터 규모와 요구사항에 따른 선택이 중요")
        print("3. 파라미터 튜닝으로 성능 최적화 가능")
        print("4. 실제 환경에서는 A/B 테스트 권장")
        
    else:
        print("\n⚠️  Milvus 서버가 실행되지 않아 실습을 완료할 수 없습니다.")
        print("docker-compose up -d 명령으로 Milvus를 시작한 후 다시 시도해주세요.")
        
        # 이론적 비교라도 제공
        print_selection_guide()

if __name__ == "__main__":
    main() 