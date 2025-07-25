#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Milvus 성능 최적화 실습

이 스크립트는 Milvus의 성능을 최적화하는 다양한 기법들을 실습합니다:
- 쿼리 최적화 (Query Optimization)
- 캐싱 전략 (Caching Strategy)  
- 배치 처리 (Batch Processing)
- 연결 풀링 (Connection Pooling)
- 메모리 최적화 (Memory Optimization)
- 성능 벤치마킹 (Performance Benchmarking)
"""

import os
import sys
import time
import logging
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import gc

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryCache:
    """쿼리 결과 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, query_vector: np.ndarray, params: dict) -> str:
        """쿼리 키 생성"""
        vector_str = np.array_str(query_vector)
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{vector_str}_{params_str}".encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """캐시 만료 확인"""
        if key not in self.creation_times:
            return True
        return (time.time() - self.creation_times[key]) > self.ttl
    
    def _evict_expired(self):
        """만료된 캐시 제거"""
        current_time = time.time()
        expired_keys = [
            key for key, creation_time in self.creation_times.items()
            if (current_time - creation_time) > self.ttl
        ]
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self):
        """LRU 정책으로 캐시 제거"""
        if len(self.cache) >= self.max_size:
            # 가장 오래된 접근 시간의 키 찾기
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """키 제거"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get(self, query_vector: np.ndarray, params: dict):
        """캐시에서 결과 조회"""
        with self._lock:
            key = self._generate_key(query_vector, params)
            
            if key in self.cache and not self._is_expired(key):
                self.access_times[key] = time.time()
                return self.cache[key]
            
            return None
    
    def put(self, query_vector: np.ndarray, params: dict, result):
        """캐시에 결과 저장"""
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            
            key = self._generate_key(query_vector, params)
            current_time = time.time()
            
            self.cache[key] = result
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def clear(self):
        """캐시 초기화"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hit_ratio": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_count', 1), 1)
            }

class ConnectionPool:
    """Milvus 연결 풀"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 pool_size: int = 10, alias_prefix: str = "pool"):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.alias_prefix = alias_prefix
        self.available_connections = deque()
        self.in_use_connections = set()
        self._lock = threading.RLock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """연결 풀 초기화"""
        for i in range(self.pool_size):
            alias = f"{self.alias_prefix}_{i}"
            try:
                connections.connect(
                    alias=alias,
                    host=self.host,
                    port=self.port
                )
                self.available_connections.append(alias)
                logger.info(f"연결 풀에 연결 추가: {alias}")
            except Exception as e:
                logger.error(f"연결 생성 실패 {alias}: {e}")
    
    def get_connection(self, timeout: float = 30.0):
        """연결 획득"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if self.available_connections:
                    alias = self.available_connections.popleft()
                    self.in_use_connections.add(alias)
                    return alias
            
            time.sleep(0.1)  # 잠시 대기
        
        raise TimeoutError("연결 풀에서 연결을 획득할 수 없습니다")
    
    def return_connection(self, alias: str):
        """연결 반환"""
        with self._lock:
            if alias in self.in_use_connections:
                self.in_use_connections.remove(alias)
                self.available_connections.append(alias)
    
    def close_all(self):
        """모든 연결 종료"""
        with self._lock:
            all_aliases = list(self.available_connections) + list(self.in_use_connections)
            for alias in all_aliases:
                try:
                    connections.remove_connection(alias)
                except:
                    pass
            self.available_connections.clear()
            self.in_use_connections.clear()
    
    def stats(self) -> Dict[str, Any]:
        """연결 풀 통계"""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "available": len(self.available_connections),
                "in_use": len(self.in_use_connections),
                "utilization": len(self.in_use_connections) / self.pool_size
            }

class PerformanceOptimizer:
    """Milvus 성능 최적화 클래스"""
    
    def __init__(self):
        self.milvus_conn = MilvusConnection()
        self.vector_utils = VectorUtils()
        self.query_cache = QueryCache(max_size=500, ttl=300)
        self.connection_pool = None
        self.performance_stats = defaultdict(list)
        
    def setup_connection_pool(self, pool_size: int = 5):
        """연결 풀 설정"""
        print("🔗 연결 풀 설정 중...")
        self.connection_pool = ConnectionPool(pool_size=pool_size)
        print(f"  ✅ {pool_size}개 연결 풀 생성 완료")
        
    def create_optimized_collection(self, collection_name: str, dim: int = 384) -> Collection:
        """최적화된 컬렉션 생성"""
        print(f"📁 최적화된 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 최적화된 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Performance optimized collection"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        
        return collection
    
    def batch_insert_optimized(self, collection: Collection, data_size: int = 10000, 
                             batch_size: int = 1000) -> float:
        """최적화된 배치 삽입"""
        print(f"💾 최적화된 배치 삽입 ({data_size:,}개 데이터, 배치크기: {batch_size:,})...")
        
        start_time = time.time()
        
        # 배치별로 데이터 생성 및 삽입
        for i in range(0, data_size, batch_size):
            batch_end = min(i + batch_size, data_size)
            actual_batch_size = batch_end - i
            
            # 배치 데이터 생성
            texts = [f"Optimized document {j} for performance testing" for j in range(i, batch_end)]
            categories = np.random.choice(['tech', 'science', 'business', 'health'], actual_batch_size)
            scores = np.random.uniform(1.0, 5.0, actual_batch_size)
            timestamps = [int(time.time()) + j for j in range(actual_batch_size)]
            
            # 벡터 생성 (배치 처리)
            vectors = self.vector_utils.texts_to_vectors(texts)
            
            # 데이터 구조화 (List[List] 형식)
            batch_data = [
                texts,
                categories.tolist(),
                scores.tolist(),
                timestamps,
                vectors.tolist()
            ]
            
            # 배치 삽입
            collection.insert(batch_data)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  진행률: {i + actual_batch_size:,}/{data_size:,} ({(i + actual_batch_size)/data_size*100:.1f}%)")
        
        # 데이터 플러시
        collection.flush()
        
        insert_time = time.time() - start_time
        print(f"  ✅ 삽입 완료: {insert_time:.2f}초")
        print(f"  📊 처리량: {data_size/insert_time:.0f} docs/sec")
        
        return insert_time
    
    def create_optimized_index(self, collection: Collection) -> float:
        """최적화된 인덱스 생성"""
        print("🔍 최적화된 인덱스 생성 중...")
        
        start_time = time.time()
        
        # HNSW 인덱스 (가장 빠른 검색 성능)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,        # 연결 수 (높을수록 정확도 증가, 메모리 사용량 증가)
                "efConstruction": 200  # 구축 시 탐색 깊이
            }
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        index_time = time.time() - start_time
        print(f"  ✅ HNSW 인덱스 생성 완료: {index_time:.2f}초")
        
        return index_time
    
    def optimized_search(self, collection: Collection, query_text: str, 
                        use_cache: bool = True) -> Tuple[Any, float, bool]:
        """최적화된 검색 (캐싱 포함)"""
        # 쿼리 벡터 생성
        query_vectors = self.vector_utils.text_to_vector(query_text)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 검색 파라미터
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 100  # 검색 시 탐색 깊이 (높을수록 정확도 증가)
            }
        }
        
        # 캐시 확인
        cache_hit = False
        if use_cache:
            cached_result = self.query_cache.get(query_vector, search_params)
            if cached_result is not None:
                return cached_result, 0.0, True  # 캐시 히트
        
        # 실제 검색 수행
        start_time = time.time()
        
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=10,
            output_fields=["text", "category", "score", "timestamp"]
        )
        
        search_time = time.time() - start_time
        
        # 결과 캐싱
        if use_cache:
            self.query_cache.put(query_vector, search_params, results)
        
        return results, search_time, cache_hit
    
    def benchmark_search_performance(self, collection: Collection, 
                                   num_queries: int = 100) -> Dict[str, Any]:
        """검색 성능 벤치마킹"""
        print(f"📊 검색 성능 벤치마킹 ({num_queries}개 쿼리)...")
        
        # 테스트 쿼리 생성
        test_queries = [
            "machine learning artificial intelligence",
            "data science analytics",
            "cloud computing technology",
            "mobile app development",
            "cybersecurity network protection",
            "blockchain cryptocurrency",
            "artificial neural networks",
            "big data processing",
            "internet of things IoT",
            "quantum computing research"
        ]
        
        # 성능 측정 변수
        total_time = 0
        cache_hits = 0
        search_times = []
        
        print("  🔍 검색 시작...")
        
        # 첫 번째 라운드: 캐시 없이
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            
            # 캐시 비활성화 검색
            _, search_time, _ = self.optimized_search(collection, query, use_cache=False)
            search_times.append(search_time)
            total_time += search_time
            
            if (i + 1) % 20 == 0:
                print(f"    진행률: {i + 1}/{num_queries}")
        
        # 캐시 통계 수집
        print("  🗄️ 캐시 성능 테스트...")
        cache_test_times = []
        
        # 두 번째 라운드: 캐시 활성화 (같은 쿼리 반복)
        for i in range(50):
            query = test_queries[i % len(test_queries)]
            _, search_time, is_cache_hit = self.optimized_search(collection, query, use_cache=True)
            cache_test_times.append(search_time)
            if is_cache_hit:
                cache_hits += 1
        
        # 통계 계산
        avg_search_time = np.mean(search_times)
        avg_cache_time = np.mean(cache_test_times)
        p95_time = np.percentile(search_times, 95)
        p99_time = np.percentile(search_times, 99)
        qps = num_queries / total_time
        
        benchmark_results = {
            "총_쿼리수": num_queries,
            "총_시간": f"{total_time:.3f}초",
            "평균_응답시간": f"{avg_search_time*1000:.2f}ms",
            "P95_응답시간": f"{p95_time*1000:.2f}ms", 
            "P99_응답시간": f"{p99_time*1000:.2f}ms",
            "QPS": f"{qps:.1f}",
            "캐시_히트율": f"{cache_hits/50*100:.1f}%",
            "캐시_평균시간": f"{avg_cache_time*1000:.2f}ms",
            "성능_향상": f"{(avg_search_time/max(avg_cache_time, 0.001)):.1f}x"
        }
        
        return benchmark_results
    
    def memory_optimization_demo(self, collection: Collection):
        """메모리 최적화 데모"""
        print("💾 메모리 최적화 데모...")
        
        # 메모리 사용량 측정 (파이썬 프로세스)
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  초기 메모리 사용량: {initial_memory:.1f}MB")
        
        # 대량 데이터 로드
        print("  📊 대량 데이터 로드 중...")
        collection.load()
        
        loaded_memory = process.memory_info().rss / 1024 / 1024
        print(f"  로드 후 메모리 사용량: {loaded_memory:.1f}MB (+{loaded_memory-initial_memory:.1f}MB)")
        
        # 메모리 최적화 팁 출력
        print("\n  💡 메모리 최적화 팁:")
        print("    1. 필요한 필드만 output_fields에 지정")
        print("    2. 컬렉션 사용 후 release() 호출")
        print("    3. 인덱스 파라미터 조정 (M, efConstruction)")
        print("    4. 파티션 활용으로 메모리 사용량 분산")
        
        # 메모리 해제
        collection.release()
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        released_memory = process.memory_info().rss / 1024 / 1024
        print(f"  해제 후 메모리 사용량: {released_memory:.1f}MB (-{loaded_memory-released_memory:.1f}MB)")
    
    def concurrent_search_test(self, collection: Collection, num_threads: int = 5, 
                             queries_per_thread: int = 20) -> Dict[str, Any]:
        """동시 검색 성능 테스트"""
        print(f"🔀 동시 검색 성능 테스트 ({num_threads}개 스레드, 스레드당 {queries_per_thread}개 쿼리)...")
        
        collection.load()
        
        test_queries = [
            "artificial intelligence machine learning",
            "data science big data analytics", 
            "cloud computing distributed systems",
            "mobile application development",
            "cybersecurity information security"
        ]
        
        def worker_function(thread_id: int) -> Dict[str, Any]:
            """워커 스레드 함수"""
            thread_times = []
            
            for i in range(queries_per_thread):
                query = test_queries[i % len(test_queries)]
                
                start_time = time.time()
                _, search_time, _ = self.optimized_search(collection, query, use_cache=False)
                total_time = time.time() - start_time
                thread_times.append(total_time)
            
            return {
                "thread_id": thread_id,
                "queries": queries_per_thread,
                "total_time": sum(thread_times),
                "avg_time": np.mean(thread_times),
                "times": thread_times
            }
        
        # 동시 실행
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        total_concurrent_time = time.time() - start_time
        
        # 결과 분석
        total_queries = num_threads * queries_per_thread
        all_times = []
        for result in results:
            all_times.extend(result["times"])
        
        concurrent_stats = {
            "스레드수": num_threads,
            "총_쿼리수": total_queries,
            "총_시간": f"{total_concurrent_time:.3f}초",
            "동시_QPS": f"{total_queries/total_concurrent_time:.1f}",
            "평균_응답시간": f"{np.mean(all_times)*1000:.2f}ms",
            "P95_응답시간": f"{np.percentile(all_times, 95)*1000:.2f}ms"
        }
        
        return concurrent_stats
    
    def run_performance_optimization_demo(self):
        """성능 최적화 종합 데모"""
        print("🚀 Milvus 성능 최적화 실습")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Milvus 연결
            self.milvus_conn.connect()
            print("✅ Milvus 연결 성공\n")
            
            # 연결 풀 설정
            self.setup_connection_pool(pool_size=3)
            
            print("=" * 80)
            print(" 🏗️ 최적화된 컬렉션 및 데이터 구축")
            print("=" * 80)
            
            # 최적화된 컬렉션 생성
            collection = self.create_optimized_collection("performance_test")
            
            # 최적화된 배치 삽입
            insert_time = self.batch_insert_optimized(collection, data_size=5000, batch_size=500)
            
            # 최적화된 인덱스 생성
            index_time = self.create_optimized_index(collection)
            
            # 컬렉션 로드
            print("\n🔄 컬렉션 로드 중...")
            collection.load()
            print("  ✅ 컬렉션 로드 완료")
            
            print("\n" + "=" * 80)
            print(" 📊 성능 벤치마킹")
            print("=" * 80)
            
            # 검색 성능 벤치마킹
            benchmark_results = self.benchmark_search_performance(collection, num_queries=50)
            
            print("\n🎯 검색 성능 결과:")
            for key, value in benchmark_results.items():
                print(f"  {key}: {value}")
            
            print("\n" + "=" * 80)
            print(" 🔀 동시성 테스트")
            print("=" * 80)
            
            # 동시 검색 테스트
            concurrent_results = self.concurrent_search_test(collection, num_threads=3, queries_per_thread=15)
            
            print("\n⚡ 동시 검색 성능 결과:")
            for key, value in concurrent_results.items():
                print(f"  {key}: {value}")
            
            print("\n" + "=" * 80)
            print(" 💾 메모리 최적화")
            print("=" * 80)
            
            # 메모리 최적화 데모
            self.memory_optimization_demo(collection)
            
            print("\n" + "=" * 80)
            print(" 📈 캐시 및 연결 풀 통계")
            print("=" * 80)
            
            # 캐시 통계
            cache_stats = self.query_cache.stats()
            print("\n🗄️ 쿼리 캐시 통계:")
            for key, value in cache_stats.items():
                print(f"  {key}: {value}")
            
            # 연결 풀 통계
            if self.connection_pool:
                pool_stats = self.connection_pool.stats()
                print("\n🔗 연결 풀 통계:")
                for key, value in pool_stats.items():
                    print(f"  {key}: {value}")
            
            print("\n" + "=" * 80)
            print(" 💡 성능 최적화 권장사항")
            print("=" * 80)
            
            print("\n🚀 성능 최적화 팁:")
            print("  1. 📊 배치 삽입: 대용량 데이터는 배치로 처리")
            print("  2. 🔍 인덱스 선택: HNSW (속도) vs IVF_PQ (메모리)")
            print("  3. 🗄️ 쿼리 캐싱: 반복 쿼리의 응답 시간 단축")
            print("  4. 🔗 연결 풀링: 연결 오버헤드 감소")
            print("  5. 💾 메모리 관리: 사용 후 release() 호출")
            print("  6. ⚡ 동시성: 적절한 스레드 수로 처리량 증대")
            print("  7. 🎯 필드 선택: 필요한 output_fields만 조회")
            print("  8. 📈 모니터링: 성능 메트릭 지속 관찰")
            
            # 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            utility.drop_collection("performance_test")
            print("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")
        
        finally:
            # 연결 정리
            if self.connection_pool:
                self.connection_pool.close_all()
            self.milvus_conn.disconnect()
            
        print(f"\n🎉 성능 최적화 실습 완료!")
        print("\n💡 학습 포인트:")
        print("  • 배치 처리와 인덱스 최적화로 성능 향상")
        print("  • 쿼리 캐싱으로 응답 시간 단축")
        print("  • 연결 풀링으로 동시성 처리 개선")
        print("  • 메모리 최적화로 시스템 안정성 확보")
        print("\n🚀 다음 단계:")
        print("  python step04_advanced/02_advanced_indexing.py")

def main():
    """메인 실행 함수"""
    optimizer = PerformanceOptimizer()
    optimizer.run_performance_optimization_demo()

if __name__ == "__main__":
    main() 