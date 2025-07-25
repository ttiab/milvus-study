#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📈 Milvus 분산 처리 및 확장성 실습

이 스크립트는 Milvus의 분산 처리 및 확장성 기법들을 실습합니다:
- 클러스터 관리 및 노드 상태 모니터링
- 데이터 파티셔닝 및 샤딩 전략
- 로드 밸런싱 및 트래픽 분산
- 복제본 관리 및 고가용성
- 스케일 아웃 시뮬레이션
- 성능 확장성 벤치마킹
"""

import os
import sys
import time
import logging
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedScalingManager:
    """분산 처리 및 확장성 관리 클래스"""
    
    def __init__(self):
        self.milvus_conn = MilvusConnection()
        self.vector_utils = VectorUtils()
        self.scaling_stats = defaultdict(list)
        self.partition_info = {}
        
    def check_cluster_status(self):
        """클러스터 상태 확인"""
        print("🌐 클러스터 상태 확인...")
        
        try:
            # Milvus 서버 정보 확인
            print("  📊 서버 정보:")
            print(f"    연결 상태: {'연결됨' if connections.has_connection('default') else '연결 끊김'}")
            
            # 컬렉션 목록 확인
            collections = utility.list_collections()
            print(f"    활성 컬렉션 수: {len(collections)}")
            
            # 메모리 사용량 (시뮬레이션)
            memory_usage = {
                "total_memory": "8.0GB",
                "used_memory": "2.3GB",
                "available_memory": "5.7GB",
                "memory_usage_percent": 28.8
            }
            
            print(f"  💾 메모리 사용량:")
            for key, value in memory_usage.items():
                print(f"    {key}: {value}")
            
            # CPU 사용량 (시뮬레이션)
            cpu_usage = {
                "cpu_cores": 8,
                "avg_cpu_usage": 45.2,
                "peak_cpu_usage": 78.9,
                "idle_cpu": 54.8
            }
            
            print(f"  🖥️  CPU 사용량:")
            for key, value in cpu_usage.items():
                print(f"    {key}: {value}")
                
            return True
            
        except Exception as e:
            print(f"  ❌ 클러스터 상태 확인 실패: {e}")
            return False
    
    def create_partitioned_collection(self, collection_name: str, num_partitions: int = 8) -> Collection:
        """파티션 기반 컬렉션 생성"""
        print(f"📂 파티션 기반 컬렉션 '{collection_name}' 생성 중...")
        print(f"  목표 파티션 수: {num_partitions}개")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="priority", dtype=DataType.INT32),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Distributed collection with partitioning strategy"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        
        # 파티션 생성 (지역별, 카테고리별, 시간별)
        partition_strategies = [
            # 지역별 파티션
            "region_us", "region_eu", "region_asia",
            # 카테고리별 파티션
            "category_tech", "category_business", "category_health",
            # 우선순위별 파티션
            "priority_high", "priority_normal"
        ]
        
        for partition_name in partition_strategies[:num_partitions]:
            collection.create_partition(partition_name)
            print(f"    ✅ 파티션 '{partition_name}' 생성됨")
            
        self.partition_info[collection_name] = partition_strategies[:num_partitions]
        print(f"  ✅ 컬렉션 생성 완료 ({num_partitions}개 파티션)")
        
        return collection
    
    def generate_distributed_data(self, total_size: int = 10000) -> Dict[str, List[List]]:
        """분산 처리용 데이터 생성"""
        print(f"📊 분산 처리용 데이터 {total_size:,}개 생성 중...")
        
        # 데이터 분포 정의
        regions = ["us", "eu", "asia"]
        categories = ["tech", "business", "health"] 
        priorities = [1, 2, 3]  # 1: high, 2: normal, 3: low
        
        # 각 파티션별 데이터 생성
        partition_data = {}
        
        for i, partition_name in enumerate(self.partition_info["distributed_collection"]):
            # 파티션별 데이터 크기 계산
            partition_size = total_size // len(self.partition_info["distributed_collection"])
            if i == 0:  # 첫 번째 파티션에 나머지 데이터 할당
                partition_size += total_size % len(self.partition_info["distributed_collection"])
            
            print(f"  📂 '{partition_name}' 파티션: {partition_size:,}개 데이터")
            
            # 파티션 특성에 맞는 데이터 생성
            if "region_" in partition_name:
                region = partition_name.split("_")[1]
                region_filter = region
                category_filter = None
                priority_filter = None
            elif "category_" in partition_name:
                category_filter = partition_name.split("_")[1]
                region_filter = None
                priority_filter = None
            elif "priority_" in partition_name:
                priority_filter = 1 if "high" in partition_name else 2
                region_filter = None
                category_filter = None
            else:
                region_filter = None
                category_filter = None
                priority_filter = None
            
            # 데이터 생성
            titles = []
            contents = []
            categories_list = []
            regions_list = []
            timestamps = []
            priorities_list = []
            scores = []
            
            for j in range(partition_size):
                # 파티션 특성 반영
                if region_filter:
                    region = region_filter
                else:
                    region = np.random.choice(regions)
                    
                if category_filter:
                    category = category_filter
                else:
                    category = np.random.choice(categories)
                    
                if priority_filter:
                    priority = priority_filter
                else:
                    priority = np.random.choice(priorities)
                
                # 문서 생성
                titles.append(f"{category.title()} Document {j} in {region.upper()}")
                contents.append(f"This is a {category} document from {region} region with priority {priority}. "
                              f"Content includes relevant information for distributed processing.")
                categories_list.append(category)
                regions_list.append(region)
                timestamps.append(int(time.time()) + j)
                priorities_list.append(priority)
                scores.append(np.random.uniform(1.0, 10.0))
            
            # 벡터 생성
            vectors = self.vector_utils.texts_to_vectors(titles)
            
            # 파티션 데이터 구조화
            partition_data[partition_name] = [
                titles,
                contents,
                categories_list,
                regions_list,
                timestamps,
                priorities_list,
                scores,
                vectors.tolist()
            ]
        
        print(f"  ✅ 분산 데이터 생성 완료")
        return partition_data
    
    def distributed_data_insertion(self, collection: Collection, partition_data: Dict[str, List[List]]) -> Dict[str, float]:
        """분산 데이터 삽입"""
        print("💾 분산 데이터 삽입 중...")
        
        insertion_stats = {}
        
        # 각 파티션에 병렬로 데이터 삽입
        def insert_to_partition(partition_name: str, data: List[List]) -> float:
            start_time = time.time()
            collection.insert(data, partition_name=partition_name)
            collection.flush()
            insertion_time = time.time() - start_time
            
            data_count = len(data[0])  # 첫 번째 필드의 길이가 데이터 개수
            print(f"    ✅ '{partition_name}': {data_count:,}개 삽입 완료 ({insertion_time:.2f}초)")
            return insertion_time
        
        # 병렬 삽입 실행
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(insert_to_partition, partition_name, data): partition_name
                for partition_name, data in partition_data.items()
            }
            
            for future in as_completed(futures):
                partition_name = futures[future]
                try:
                    insertion_time = future.result()
                    insertion_stats[partition_name] = insertion_time
                except Exception as e:
                    print(f"    ❌ '{partition_name}' 삽입 실패: {e}")
                    insertion_stats[partition_name] = -1
        
        total_time = sum(t for t in insertion_stats.values() if t > 0)
        print(f"  ✅ 전체 삽입 완료: {total_time:.2f}초")
        
        return insertion_stats
    
    def create_distributed_indexes(self, collection: Collection) -> Dict[str, float]:
        """분산 인덱스 생성"""
        print("🔍 분산 인덱스 생성 중...")
        
        # 인덱스 파라미터 (대용량 데이터에 적합한 설정)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 256}  # 파티션 수에 맞춰 조정
        }
        
        start_time = time.time()
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        build_time = time.time() - start_time
        
        print(f"  ✅ 분산 인덱스 생성 완료: {build_time:.2f}초")
        
        return {"vector_index": build_time}
    
    def partition_specific_search(self, collection: Collection) -> Dict[str, Any]:
        """파티션별 검색 성능 테스트"""
        print("🔍 파티션별 검색 성능 테스트...")
        
        collection.load()
        
        # 테스트 쿼리들
        test_queries = [
            {"query": "technology innovation artificial intelligence", "partitions": ["region_us", "category_tech"]},
            {"query": "business strategy market analysis", "partitions": ["region_eu", "category_business"]},
            {"query": "healthcare medical research", "partitions": ["region_asia", "category_health"]},
            {"query": "high priority urgent task", "partitions": ["priority_high"]}
        ]
        
        search_results = {}
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n  📋 테스트 {i}: '{test['query']}'")
            
            # 쿼리 벡터 생성
            query_vectors = self.vector_utils.text_to_vector(test['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 전체 컬렉션 검색
            start_time = time.time()
            all_results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 32}},
                limit=10,
                output_fields=["title", "category", "region", "priority"]
            )
            all_search_time = time.time() - start_time
            
            # 특정 파티션 검색
            start_time = time.time()
            partition_results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 32}},
                limit=10,
                partition_names=test['partitions'],
                output_fields=["title", "category", "region", "priority"]
            )
            partition_search_time = time.time() - start_time
            
            # 결과 분석
            speedup = all_search_time / partition_search_time if partition_search_time > 0 else 0
            
            print(f"    전체 검색: {all_search_time*1000:.2f}ms, 결과: {len(all_results[0])}개")
            print(f"    파티션 검색: {partition_search_time*1000:.2f}ms, 결과: {len(partition_results[0])}개")
            print(f"    성능 향상: {speedup:.1f}x")
            
            search_results[f"test_{i}"] = {
                "query": test['query'],
                "partitions": test['partitions'],
                "all_search_time": all_search_time,
                "partition_search_time": partition_search_time,
                "speedup": speedup,
                "all_results_count": len(all_results[0]),
                "partition_results_count": len(partition_results[0])
            }
        
        collection.release()
        return search_results
    
    def load_balancing_simulation(self, collection: Collection) -> Dict[str, Any]:
        """로드 밸런싱 시뮬레이션"""
        print("\n⚖️ 로드 밸런싱 시뮬레이션...")
        
        collection.load()
        
        # 다양한 쿼리 패턴 시뮬레이션
        query_patterns = [
            {"type": "regional", "weight": 0.4, "queries": ["news from america", "european markets", "asian technology"]},
            {"type": "categorical", "weight": 0.35, "queries": ["tech innovation", "business analysis", "health research"]},
            {"type": "priority", "weight": 0.25, "queries": ["urgent task", "high priority", "critical update"]}
        ]
        
        # 동시 요청 시뮬레이션
        def worker_simulation(worker_id: int, num_requests: int) -> Dict[str, Any]:
            worker_stats = {
                "worker_id": worker_id,
                "total_requests": num_requests,
                "total_time": 0,
                "avg_response_time": 0,
                "requests_per_pattern": defaultdict(int)
            }
            
            start_time = time.time()
            
            for i in range(num_requests):
                # 쿼리 패턴 선택 (가중치 기반)
                pattern = np.random.choice(
                    [p["type"] for p in query_patterns],
                    p=[p["weight"] for p in query_patterns]
                )
                
                # 패턴에 맞는 쿼리 선택
                pattern_info = next(p for p in query_patterns if p["type"] == pattern)
                query = np.random.choice(pattern_info["queries"])
                
                # 벡터 변환 및 검색
                query_vectors = self.vector_utils.text_to_vector(query)
                query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
                
                # 패턴에 따른 파티션 선택
                if pattern == "regional":
                    partition_names = ["region_us", "region_eu", "region_asia"]
                elif pattern == "categorical":
                    partition_names = ["category_tech", "category_business", "category_health"]
                else:  # priority
                    partition_names = ["priority_high", "priority_normal"]
                
                # 검색 실행
                selected_partitions = [np.random.choice(partition_names)]
                collection.search(
                    data=[query_vector.tolist()],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    limit=5,
                    partition_names=selected_partitions,
                    output_fields=["title"]
                )
                
                worker_stats["requests_per_pattern"][pattern] += 1
            
            worker_stats["total_time"] = time.time() - start_time
            worker_stats["avg_response_time"] = worker_stats["total_time"] / num_requests
            
            return worker_stats
        
        # 다중 워커로 부하 분산 테스트
        print("  🔄 다중 워커 부하 분산 테스트...")
        
        num_workers = 5
        requests_per_worker = 20
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_simulation, i, requests_per_worker)
                for i in range(num_workers)
            ]
            
            worker_results = [future.result() for future in as_completed(futures)]
        
        # 결과 분석
        total_requests = sum(w["total_requests"] for w in worker_results)
        total_time = max(w["total_time"] for w in worker_results)
        avg_response_time = np.mean([w["avg_response_time"] for w in worker_results])
        throughput = total_requests / total_time
        
        print(f"    총 요청 수: {total_requests}")
        print(f"    실행 시간: {total_time:.2f}초")
        print(f"    평균 응답 시간: {avg_response_time*1000:.2f}ms")
        print(f"    처리량: {throughput:.1f} requests/sec")
        
        # 패턴별 분포 분석
        pattern_distribution = defaultdict(int)
        for worker in worker_results:
            for pattern, count in worker["requests_per_pattern"].items():
                pattern_distribution[pattern] += count
        
        print(f"    패턴 분포:")
        for pattern, count in pattern_distribution.items():
            percentage = (count / total_requests) * 100
            print(f"      {pattern}: {count}회 ({percentage:.1f}%)")
        
        collection.release()
        
        return {
            "total_requests": total_requests,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "throughput": throughput,
            "pattern_distribution": dict(pattern_distribution),
            "worker_results": worker_results
        }
    
    def scalability_benchmarking(self, collection: Collection) -> Dict[str, Any]:
        """확장성 벤치마킹"""
        print("\n📈 확장성 벤치마킹...")
        
        collection.load()
        
        # 다양한 부하 레벨 테스트
        load_levels = [
            {"name": "저부하", "concurrent_users": 2, "requests_per_user": 10},
            {"name": "중부하", "concurrent_users": 5, "requests_per_user": 15},
            {"name": "고부하", "concurrent_users": 10, "requests_per_user": 20}
        ]
        
        benchmark_results = {}
        
        for load_test in load_levels:
            print(f"\n  📊 {load_test['name']} 테스트:")
            print(f"    동시 사용자: {load_test['concurrent_users']}명")
            print(f"    사용자당 요청: {load_test['requests_per_user']}회")
            
            def benchmark_user(user_id: int, requests: int) -> Dict[str, Any]:
                user_times = []
                
                for i in range(requests):
                    query = f"benchmark query {i} from user {user_id}"
                    query_vectors = self.vector_utils.text_to_vector(query)
                    query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
                    
                    start_time = time.time()
                    collection.search(
                        data=[query_vector.tolist()],
                        anns_field="vector",
                        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                        limit=10,
                        output_fields=["title"]
                    )
                    response_time = time.time() - start_time
                    user_times.append(response_time)
                
                return {
                    "user_id": user_id,
                    "requests": requests,
                    "times": user_times,
                    "avg_time": np.mean(user_times),
                    "p95_time": np.percentile(user_times, 95)
                }
            
            # 동시 실행
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=load_test['concurrent_users']) as executor:
                futures = [
                    executor.submit(benchmark_user, i, load_test['requests_per_user'])
                    for i in range(load_test['concurrent_users'])
                ]
                user_results = [future.result() for future in as_completed(futures)]
            
            total_test_time = time.time() - start_time
            
            # 결과 분석
            total_requests = sum(r["requests"] for r in user_results)
            all_times = []
            for r in user_results:
                all_times.extend(r["times"])
            
            avg_response_time = np.mean(all_times)
            p95_response_time = np.percentile(all_times, 95)
            p99_response_time = np.percentile(all_times, 99)
            throughput = total_requests / total_test_time
            
            print(f"    총 요청 수: {total_requests}")
            print(f"    평균 응답 시간: {avg_response_time*1000:.2f}ms")
            print(f"    P95 응답 시간: {p95_response_time*1000:.2f}ms")
            print(f"    P99 응답 시간: {p99_response_time*1000:.2f}ms")
            print(f"    처리량: {throughput:.1f} req/sec")
            
            benchmark_results[load_test['name']] = {
                "concurrent_users": load_test['concurrent_users'],
                "requests_per_user": load_test['requests_per_user'],
                "total_requests": total_requests,
                "total_time": total_test_time,
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "throughput": throughput
            }
        
        collection.release()
        return benchmark_results
    
    def resource_utilization_analysis(self) -> Dict[str, Any]:
        """리소스 사용률 분석"""
        print("\n💻 리소스 사용률 분석...")
        
        # 시뮬레이션된 리소스 메트릭스
        resource_metrics = {
            "cpu_utilization": {
                "idle": np.random.uniform(15, 25),
                "user": np.random.uniform(40, 60),
                "system": np.random.uniform(10, 20),
                "iowait": np.random.uniform(2, 8)
            },
            "memory_usage": {
                "total_gb": 16.0,
                "used_gb": np.random.uniform(8, 12),
                "cached_gb": np.random.uniform(2, 4),
                "buffer_gb": np.random.uniform(0.5, 1.5)
            },
            "disk_io": {
                "read_mbps": np.random.uniform(50, 150),
                "write_mbps": np.random.uniform(30, 100),
                "io_utilization": np.random.uniform(20, 60)
            },
            "network": {
                "rx_mbps": np.random.uniform(10, 50),
                "tx_mbps": np.random.uniform(5, 30),
                "connections": np.random.randint(100, 500)
            }
        }
        
        print("  📊 현재 리소스 사용률:")
        
        cpu = resource_metrics["cpu_utilization"]
        cpu_total_used = cpu["user"] + cpu["system"] + cpu["iowait"]
        print(f"    CPU: {cpu_total_used:.1f}% (사용자: {cpu['user']:.1f}%, 시스템: {cpu['system']:.1f}%)")
        
        mem = resource_metrics["memory_usage"]
        mem_usage_percent = (mem["used_gb"] / mem["total_gb"]) * 100
        print(f"    메모리: {mem_usage_percent:.1f}% ({mem['used_gb']:.1f}GB / {mem['total_gb']}GB)")
        
        disk = resource_metrics["disk_io"]
        print(f"    디스크 I/O: 읽기 {disk['read_mbps']:.1f}MB/s, 쓰기 {disk['write_mbps']:.1f}MB/s")
        
        net = resource_metrics["network"]
        print(f"    네트워크: 수신 {net['rx_mbps']:.1f}MB/s, 송신 {net['tx_mbps']:.1f}MB/s")
        
        # 최적화 권장사항
        recommendations = []
        
        if cpu_total_used > 80:
            recommendations.append("CPU 사용률이 높습니다. 워커 노드 추가를 고려하세요.")
        
        if mem_usage_percent > 85:
            recommendations.append("메모리 사용률이 높습니다. 메모리 증설 또는 캐시 최적화가 필요합니다.")
        
        if disk["io_utilization"] > 70:
            recommendations.append("디스크 I/O가 포화 상태입니다. SSD 사용 또는 I/O 최적화를 검토하세요.")
        
        if net["connections"] > 400:
            recommendations.append("동시 연결 수가 많습니다. 연결 풀 크기 조정을 고려하세요.")
        
        if not recommendations:
            recommendations.append("현재 리소스 사용률이 적정 수준입니다.")
        
        print("\n  💡 최적화 권장사항:")
        for rec in recommendations:
            print(f"    • {rec}")
        
        return {
            "metrics": resource_metrics,
            "recommendations": recommendations
        }
    
    def run_distributed_scaling_demo(self):
        """분산 처리 및 확장성 종합 데모"""
        print("📈 Milvus 분산 처리 및 확장성 실습")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Milvus 연결
            self.milvus_conn.connect()
            print("✅ Milvus 연결 성공\n")
            
            print("=" * 80)
            print(" 🌐 클러스터 상태 및 환경 확인")
            print("=" * 80)
            
            # 클러스터 상태 확인
            cluster_ok = self.check_cluster_status()
            
            if not cluster_ok:
                print("⚠️  클러스터 상태가 불안정합니다. 단일 노드 시뮬레이션으로 진행합니다.")
            
            print("\n" + "=" * 80)
            print(" 📂 분산 데이터 아키텍처 구축")
            print("=" * 80)
            
            # 파티션 기반 컬렉션 생성
            collection = self.create_partitioned_collection("distributed_collection", num_partitions=8)
            
            # 분산 데이터 생성
            partition_data = self.generate_distributed_data(total_size=8000)
            
            # 분산 데이터 삽입
            insertion_stats = self.distributed_data_insertion(collection, partition_data)
            
            # 분산 인덱스 생성
            index_stats = self.create_distributed_indexes(collection)
            
            print("\n" + "=" * 80)
            print(" 🔍 파티션 기반 검색 최적화")
            print("=" * 80)
            
            # 파티션별 검색 성능 테스트
            search_results = self.partition_specific_search(collection)
            
            print(f"\n📊 파티션 검색 성능 요약:")
            total_speedup = 0
            valid_tests = 0
            
            for test_name, result in search_results.items():
                if result['speedup'] > 0:
                    print(f"  {test_name}: {result['speedup']:.1f}x 성능 향상")
                    total_speedup += result['speedup']
                    valid_tests += 1
            
            if valid_tests > 0:
                avg_speedup = total_speedup / valid_tests
                print(f"  평균 성능 향상: {avg_speedup:.1f}x")
            
            print("\n" + "=" * 80)
            print(" ⚖️ 로드 밸런싱 및 동시성")
            print("=" * 80)
            
            # 로드 밸런싱 시뮬레이션
            load_balancing_results = self.load_balancing_simulation(collection)
            
            print("\n" + "=" * 80)
            print(" 📈 확장성 벤치마킹")
            print("=" * 80)
            
            # 확장성 벤치마킹
            scalability_results = self.scalability_benchmarking(collection)
            
            print(f"\n📊 확장성 벤치마킹 요약:")
            for load_name, result in scalability_results.items():
                print(f"  {load_name}: {result['throughput']:.1f} req/sec, "
                      f"평균 {result['avg_response_time']*1000:.2f}ms")
            
            print("\n" + "=" * 80)
            print(" 💻 리소스 사용률 및 최적화")
            print("=" * 80)
            
            # 리소스 사용률 분석
            resource_analysis = self.resource_utilization_analysis()
            
            print("\n" + "=" * 80)
            print(" 🎯 분산 처리 권장사항")
            print("=" * 80)
            
            print("\n🏗️ 아키텍처 설계 가이드:")
            print("  📊 데이터 분산 전략:")
            print("    • 지역별 파티셔닝: 지리적 분산 및 지연 시간 최소화")
            print("    • 카테고리별 파티셔닝: 도메인 특화 검색 최적화")
            print("    • 시간별 파티셔닝: 최신 데이터 우선 검색")
            print("    • 우선순위별 파티셔닝: 중요도 기반 리소스 할당")
            
            print("\n  ⚖️ 로드 밸런싱:")
            print("    • 라운드 로빈: 균등 분산")
            print("    • 가중치 기반: 노드 성능에 따른 분산")
            print("    • 지역 기반: 지연 시간 최소화")
            print("    • 동적 분산: 실시간 부하 모니터링")
            
            print("\n  📈 확장성 최적화:")
            print("    • 수평 확장: 노드 추가로 처리량 증대")
            print("    • 수직 확장: 하드웨어 성능 향상")
            print("    • 읽기 복제본: 읽기 성능 분산")
            print("    • 캐싱 계층: 반복 쿼리 성능 향상")
            
            print("\n  🔧 운영 모범 사례:")
            print("    • 모니터링: 실시간 성능 지표 추적")
            print("    • 오토 스케일링: 부하에 따른 자동 확장")
            print("    • 장애 복구: 고가용성 설계")
            print("    • 백업 전략: 데이터 보호 및 복구")
            
            # 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            utility.drop_collection("distributed_collection")
            print("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")
        
        finally:
            self.milvus_conn.disconnect()
            
        print(f"\n🎉 분산 처리 및 확장성 실습 완료!")
        print("\n💡 학습 포인트:")
        print("  • 파티셔닝을 통한 데이터 분산 및 검색 최적화")
        print("  • 로드 밸런싱으로 시스템 부하 분산")
        print("  • 확장성 벤치마킹을 통한 성능 한계 파악")
        print("  • 리소스 모니터링 및 최적화 전략 수립")
        print("\n🚀 다음 단계:")
        print("  python step04_advanced/04_realtime_streaming.py")

def main():
    """메인 실행 함수"""
    scaling_manager = DistributedScalingManager()
    scaling_manager.run_distributed_scaling_demo()

if __name__ == "__main__":
    main() 