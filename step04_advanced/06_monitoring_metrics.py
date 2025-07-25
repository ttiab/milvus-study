#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 Milvus 모니터링 및 메트릭스 실습

이 스크립트는 Milvus의 모니터링 및 메트릭스 수집을 실습합니다:
- 실시간 성능 메트릭 수집
- 시스템 리소스 모니터링
- 쿼리 성능 분석 및 추적
- 알림 시스템 구축
- 대시보드 데이터 생성
- 성능 이상 탐지
"""

import os
import sys
import time
import logging
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
import json
import statistics
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

class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self, collection_window: int = 300):
        self.collection_window = collection_window  # 5분 윈도우
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules = []
        self.is_collecting = False
        self.collection_thread = None
        
    def start_collection(self):
        """메트릭 수집 시작"""
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.start()
        print("📊 메트릭 수집 시작됨")
    
    def stop_collection(self):
        """메트릭 수집 중지"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        print("📊 메트릭 수집 중지됨")
    
    def _collect_metrics_loop(self):
        """메트릭 수집 루프"""
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # 시스템 메트릭 수집
                system_metrics = self._collect_system_metrics()
                
                # 메트릭 저장
                for metric_name, value in system_metrics.items():
                    self.metrics_history[metric_name].append({
                        "timestamp": timestamp,
                        "value": value
                    })
                
                # 알림 확인
                self._check_alerts(system_metrics, timestamp)
                
                time.sleep(10)  # 10초마다 수집
                
            except Exception as e:
                logger.error(f"메트릭 수집 오류: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)
            
            # 프로세스 수
            process_count = len(psutil.pids())
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "memory_available_gb": memory_available_gb,
                "disk_percent": disk_percent,
                "disk_used_gb": disk_used_gb,
                "disk_free_gb": disk_free_gb,
                "network_sent_mb": network_sent_mb,
                "network_recv_mb": network_recv_mb,
                "process_count": process_count
            }
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      operator: str = "greater", duration: int = 60):
        """알림 규칙 추가"""
        rule = {
            "metric_name": metric_name,
            "threshold": threshold,
            "operator": operator,  # greater, less, equal
            "duration": duration,  # 초
            "last_triggered": 0,
            "consecutive_violations": 0
        }
        self.alert_rules.append(rule)
    
    def _check_alerts(self, current_metrics: Dict[str, float], timestamp: float):
        """알림 확인"""
        for rule in self.alert_rules:
            metric_name = rule["metric_name"]
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            threshold = rule["threshold"]
            
            # 조건 확인
            violation = False
            if rule["operator"] == "greater" and current_value > threshold:
                violation = True
            elif rule["operator"] == "less" and current_value < threshold:
                violation = True
            elif rule["operator"] == "equal" and abs(current_value - threshold) < 0.01:
                violation = True
            
            if violation:
                rule["consecutive_violations"] += 1
                
                # 지속 시간 확인
                if (rule["consecutive_violations"] * 10 >= rule["duration"] and 
                    timestamp - rule["last_triggered"] > 300):  # 5분 간격
                    
                    self._trigger_alert(rule, current_value, timestamp)
                    rule["last_triggered"] = timestamp
                    rule["consecutive_violations"] = 0
            else:
                rule["consecutive_violations"] = 0
    
    def _trigger_alert(self, rule: Dict, current_value: float, timestamp: float):
        """알림 발생"""
        alert_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"🚨 알림: {rule['metric_name']} = {current_value:.2f} "
              f"({rule['operator']} {rule['threshold']}) at {alert_time}")
    
    def get_metrics_summary(self, metric_name: str, 
                           time_range: int = 300) -> Dict[str, Any]:
        """메트릭 요약 통계"""
        if metric_name not in self.metrics_history:
            return {"error": f"Metric {metric_name} not found"}
        
        current_time = time.time()
        recent_data = [
            point for point in self.metrics_history[metric_name]
            if current_time - point["timestamp"] <= time_range
        ]
        
        if not recent_data:
            return {"error": "No recent data available"}
        
        values = [point["value"] for point in recent_data]
        
        return {
            "metric_name": metric_name,
            "time_range_seconds": time_range,
            "data_points": len(values),
            "current": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }

class QueryPerformanceTracker:
    """쿼리 성능 추적기"""
    
    def __init__(self, max_history: int = 10000):
        self.query_history = deque(maxlen=max_history)
        self.performance_stats = defaultdict(list)
        self.slow_query_threshold = 1.0  # 1초 이상은 느린 쿼리
        
    def track_query(self, query_type: str, execution_time: float, 
                   result_count: int, parameters: Dict[str, Any] = None):
        """쿼리 추적"""
        query_record = {
            "timestamp": time.time(),
            "query_type": query_type,
            "execution_time": execution_time,
            "result_count": result_count,
            "parameters": parameters or {},
            "is_slow": execution_time > self.slow_query_threshold
        }
        
        self.query_history.append(query_record)
        self.performance_stats[query_type].append(execution_time)
    
    def get_performance_metrics(self, time_range: int = 300) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        current_time = time.time()
        recent_queries = [
            q for q in self.query_history
            if current_time - q["timestamp"] <= time_range
        ]
        
        if not recent_queries:
            return {"message": "No recent queries"}
        
        # 전체 통계
        execution_times = [q["execution_time"] for q in recent_queries]
        result_counts = [q["result_count"] for q in recent_queries]
        slow_queries = [q for q in recent_queries if q["is_slow"]]
        
        # 쿼리 타입별 통계
        query_type_stats = defaultdict(list)
        for query in recent_queries:
            query_type_stats[query["query_type"]].append(query["execution_time"])
        
        type_metrics = {}
        for query_type, times in query_type_stats.items():
            type_metrics[query_type] = {
                "count": len(times),
                "avg_time": statistics.mean(times),
                "max_time": max(times),
                "min_time": min(times)
            }
        
        return {
            "time_range_seconds": time_range,
            "total_queries": len(recent_queries),
            "avg_execution_time": statistics.mean(execution_times),
            "p95_execution_time": np.percentile(execution_times, 95),
            "p99_execution_time": np.percentile(execution_times, 99),
            "avg_result_count": statistics.mean(result_counts),
            "slow_query_count": len(slow_queries),
            "slow_query_percentage": (len(slow_queries) / len(recent_queries)) * 100,
            "qps": len(recent_queries) / time_range,
            "query_type_metrics": type_metrics
        }
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """느린 쿼리 조회"""
        slow_queries = [q for q in self.query_history if q["is_slow"]]
        slow_queries.sort(key=lambda x: x["execution_time"], reverse=True)
        return slow_queries[:limit]

class MilvusMonitor:
    """Milvus 모니터링 시스템"""
    
    def __init__(self, collection: Collection):
        self.collection = collection
        self.metrics_collector = MetricsCollector()
        self.query_tracker = QueryPerformanceTracker()
        self.vector_utils = VectorUtils()
        self.monitoring_active = False
        
    def start_monitoring(self):
        """모니터링 시작"""
        print("🔍 Milvus 모니터링 시작...")
        
        # 메트릭 수집 시작
        self.metrics_collector.start_collection()
        
        # 알림 규칙 설정
        self.setup_alert_rules()
        
        self.monitoring_active = True
        print("✅ 모니터링 활성화됨")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        self.metrics_collector.stop_collection()
        print("✅ 모니터링 중지됨")
    
    def setup_alert_rules(self):
        """알림 규칙 설정"""
        # CPU 사용률 알림
        self.metrics_collector.add_alert_rule("cpu_percent", 80.0, "greater", 60)
        
        # 메모리 사용률 알림
        self.metrics_collector.add_alert_rule("memory_percent", 85.0, "greater", 60)
        
        # 디스크 사용률 알림
        self.metrics_collector.add_alert_rule("disk_percent", 90.0, "greater", 120)
        
        print("📋 알림 규칙 설정 완료")
    
    def execute_monitored_search(self, query_text: str, limit: int = 10, 
                                filters: str = None) -> Tuple[List, float]:
        """모니터링되는 검색 실행"""
        start_time = time.time()
        
        try:
            # 벡터화
            query_vectors = self.vector_utils.text_to_vector(query_text)
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 검색 실행
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filters,
                output_fields=["content", "source", "priority"]
            )
            
            execution_time = time.time() - start_time
            result_count = len(results[0]) if results and len(results) > 0 else 0
            
            # 성능 추적
            self.query_tracker.track_query(
                query_type="vector_search",
                execution_time=execution_time,
                result_count=result_count,
                parameters={"limit": limit, "filters": filters}
            )
            
            return results, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"검색 실행 오류: {e}")
            
            # 오류도 추적
            self.query_tracker.track_query(
                query_type="vector_search_error",
                execution_time=execution_time,
                result_count=0,
                parameters={"error": str(e)}
            )
            
            return [], execution_time
    
    def execute_monitored_query(self, expr: str, limit: int = 100) -> Tuple[List, float]:
        """모니터링되는 쿼리 실행"""
        start_time = time.time()
        
        try:
            results = self.collection.query(
                expr=expr,
                limit=limit,
                output_fields=["content", "source", "priority"]
            )
            
            execution_time = time.time() - start_time
            
            # 성능 추적
            self.query_tracker.track_query(
                query_type="scalar_query",
                execution_time=execution_time,
                result_count=len(results),
                parameters={"expr": expr, "limit": limit}
            )
            
            return results, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"쿼리 실행 오류: {e}")
            
            # 오류도 추적
            self.query_tracker.track_query(
                query_type="scalar_query_error",
                execution_time=execution_time,
                result_count=0,
                parameters={"error": str(e)}
            )
            
            return [], execution_time
    
    def run_performance_stress_test(self, duration: int = 60):
        """성능 스트레스 테스트"""
        print(f"⚡ 성능 스트레스 테스트 시작 ({duration}초)...")
        
        self.collection.load()
        
        # 테스트 쿼리들
        test_queries = [
            "artificial intelligence machine learning",
            "data science analytics",
            "cloud computing technology",
            "mobile app development",
            "cybersecurity network security",
            "blockchain cryptocurrency",
            "software engineering",
            "database management"
        ]
        
        test_filters = [
            None,
            "priority >= 3",
            "source == 'web'",
            "priority <= 2 and source != 'api'"
        ]
        
        def worker_function(worker_id: int):
            """워커 함수"""
            end_time = time.time() + duration
            worker_queries = 0
            
            while time.time() < end_time:
                try:
                    # 랜덤 쿼리 선택
                    query = np.random.choice(test_queries)
                    filter_expr = np.random.choice(test_filters)
                    limit = np.random.randint(5, 20)
                    
                    # 검색 실행
                    _, exec_time = self.execute_monitored_search(query, limit, filter_expr)
                    worker_queries += 1
                    
                    # 스칼라 쿼리도 일부 실행
                    if np.random.random() < 0.3:  # 30% 확률
                        expr = np.random.choice(["priority >= 1", "source == 'web'", "priority <= 3"])
                        _, exec_time = self.execute_monitored_query(expr, 50)
                        worker_queries += 1
                    
                    # 간격 조정
                    time.sleep(np.random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    logger.error(f"워커 {worker_id} 오류: {e}")
            
            return worker_queries
        
        # 동시 워커 실행
        num_workers = 5
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_workers)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        total_queries = sum(worker_results)
        
        self.collection.release()
        
        print(f"  ✅ 스트레스 테스트 완료")
        print(f"  📊 총 쿼리 수: {total_queries}")
        print(f"  ⚡ 평균 QPS: {total_queries/duration:.1f}")
        
        return {
            "duration": duration,
            "total_queries": total_queries,
            "average_qps": total_queries / duration,
            "worker_count": num_workers
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """모니터링 보고서 생성"""
        print("📊 모니터링 보고서 생성 중...")
        
        current_time = datetime.now()
        
        # 시스템 메트릭 요약
        system_metrics = {}
        key_metrics = ["cpu_percent", "memory_percent", "disk_percent"]
        
        for metric in key_metrics:
            summary = self.metrics_collector.get_metrics_summary(metric, 300)
            system_metrics[metric] = summary
        
        # 쿼리 성능 메트릭
        query_metrics = self.query_tracker.get_performance_metrics(300)
        
        # 느린 쿼리 목록
        slow_queries = self.query_tracker.get_slow_queries(5)
        
        # 컬렉션 상태
        collection_stats = {
            "name": self.collection.name,
            "num_entities": self.collection.num_entities,
            "description": self.collection.description
        }
        
        # 보고서 구성
        report = {
            "generated_at": current_time.isoformat(),
            "monitoring_period": "Last 5 minutes",
            "system_metrics": system_metrics,
            "query_performance": query_metrics,
            "slow_queries": slow_queries,
            "collection_status": collection_stats,
            "alert_rules_count": len(self.metrics_collector.alert_rules),
            "recommendations": self._generate_recommendations(system_metrics, query_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, system_metrics: Dict, query_metrics: Dict) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # CPU 사용률 확인
        if system_metrics.get("cpu_percent", {}).get("mean", 0) > 70:
            recommendations.append("CPU 사용률이 높습니다. 쿼리 최적화 또는 스케일 아웃을 고려하세요.")
        
        # 메모리 사용률 확인
        if system_metrics.get("memory_percent", {}).get("mean", 0) > 80:
            recommendations.append("메모리 사용률이 높습니다. 인덱스 최적화 또는 메모리 증설을 고려하세요.")
        
        # 느린 쿼리 확인
        if query_metrics.get("slow_query_percentage", 0) > 10:
            recommendations.append("느린 쿼리가 많습니다. 인덱스 구성 및 쿼리 패턴을 검토하세요.")
        
        # 평균 응답 시간 확인
        if query_metrics.get("avg_execution_time", 0) > 0.5:
            recommendations.append("평균 응답 시간이 높습니다. 검색 파라미터 튜닝을 고려하세요.")
        
        if not recommendations:
            recommendations.append("현재 시스템 성능이 양호합니다.")
        
        return recommendations

class MonitoringDashboard:
    """모니터링 대시보드"""
    
    def __init__(self, monitor: MilvusMonitor):
        self.monitor = monitor
        
    def display_realtime_dashboard(self, refresh_interval: int = 10, duration: int = 60):
        """실시간 대시보드 표시"""
        print("🖥️ 실시간 모니터링 대시보드")
        print("=" * 80)
        
        end_time = time.time() + duration
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            
            # 헤더 정보
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n📊 대시보드 업데이트 #{iteration} - {current_time}")
            print("-" * 60)
            
            # 시스템 메트릭
            print("🖥️ 시스템 리소스:")
            cpu_metrics = self.monitor.metrics_collector.get_metrics_summary("cpu_percent", 60)
            mem_metrics = self.monitor.metrics_collector.get_metrics_summary("memory_percent", 60)
            
            if cpu_metrics.get("current") is not None:
                print(f"  CPU: {cpu_metrics['current']:.1f}% (평균: {cpu_metrics['mean']:.1f}%)")
            if mem_metrics.get("current") is not None:
                print(f"  메모리: {mem_metrics['current']:.1f}% (평균: {mem_metrics['mean']:.1f}%)")
            
            # 쿼리 성능
            print("\n⚡ 쿼리 성능:")
            query_metrics = self.monitor.query_tracker.get_performance_metrics(60)
            
            if query_metrics.get("total_queries", 0) > 0:
                print(f"  총 쿼리: {query_metrics['total_queries']}")
                print(f"  평균 응답시간: {query_metrics['avg_execution_time']*1000:.2f}ms")
                print(f"  QPS: {query_metrics['qps']:.1f}")
                print(f"  느린 쿼리: {query_metrics['slow_query_count']}")
            else:
                print("  쿼리 데이터 없음")
            
            # 컬렉션 상태
            print(f"\n📁 컬렉션 상태:")
            print(f"  이름: {self.monitor.collection.name}")
            print(f"  엔티티 수: {self.monitor.collection.num_entities:,}")
            
            # 다음 업데이트까지 대기
            if time.time() + refresh_interval < end_time:
                print(f"\n⏳ {refresh_interval}초 후 업데이트...")
                time.sleep(refresh_interval)
            else:
                break
        
        print("\n✅ 대시보드 모니터링 완료")

class MonitoringManager:
    """모니터링 관리자"""
    
    def __init__(self):
        self.milvus_conn = MilvusConnection()
        self.vector_utils = VectorUtils()
        self.monitor = None
        self.dashboard = None
        
    def create_monitoring_collection(self, collection_name: str, data_size: int = 5000) -> Collection:
        """모니터링용 컬렉션 생성"""
        print(f"🏗️ 모니터링용 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="priority", dtype=DataType.INT32),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="score", dtype=DataType.FLOAT),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Collection for monitoring and metrics testing"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        
        # 테스트 데이터 생성
        print(f"  📊 테스트 데이터 {data_size:,}개 생성 중...")
        
        sources = ["web", "mobile", "api", "batch", "stream"]
        priorities = [1, 2, 3, 4, 5]
        
        contents = []
        source_list = []
        priority_list = []
        timestamp_list = []
        score_list = []
        
        for i in range(data_size):
            contents.append(f"Monitoring test document {i} with various performance characteristics")
            source_list.append(np.random.choice(sources))
            priority_list.append(np.random.choice(priorities))
            timestamp_list.append(int(time.time()) + i)
            score_list.append(np.random.uniform(1.0, 10.0))
        
        # 벡터 생성
        vectors = self.vector_utils.texts_to_vectors(contents)
        
        # 데이터 삽입
        data = [
            contents,
            source_list,
            priority_list,
            timestamp_list,
            score_list,
            vectors.tolist()
        ]
        
        collection.insert(data)
        collection.flush()
        
        # 인덱스 생성
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("vector", index_params)
        
        print(f"  ✅ 모니터링용 컬렉션 생성 완료 ({data_size:,}개 엔티티)")
        return collection
    
    def run_monitoring_demo(self):
        """모니터링 종합 데모"""
        print("📊 Milvus 모니터링 및 메트릭스 실습")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Milvus 연결
            self.milvus_conn.connect()
            print("✅ Milvus 연결 성공\n")
            
            print("=" * 80)
            print(" 🏗️ 모니터링 환경 구축")
            print("=" * 80)
            
            # 모니터링용 컬렉션 생성
            collection = self.create_monitoring_collection("monitoring_test_collection", 3000)
            
            # 모니터링 시스템 초기화
            self.monitor = MilvusMonitor(collection)
            self.dashboard = MonitoringDashboard(self.monitor)
            
            print("\n" + "=" * 80)
            print(" 📊 모니터링 시스템 시작")
            print("=" * 80)
            
            # 모니터링 시작
            self.monitor.start_monitoring()
            
            print("\n" + "=" * 80)
            print(" ⚡ 성능 스트레스 테스트")
            print("=" * 80)
            
            # 스트레스 테스트 실행 (백그라운드)
            stress_results = self.monitor.run_performance_stress_test(duration=30)
            
            print("\n" + "=" * 80)
            print(" 🖥️ 실시간 모니터링 대시보드")
            print("=" * 80)
            
            # 실시간 대시보드 (30초)
            self.dashboard.display_realtime_dashboard(refresh_interval=10, duration=30)
            
            print("\n" + "=" * 80)
            print(" 📋 종합 모니터링 보고서")
            print("=" * 80)
            
            # 모니터링 보고서 생성
            report = self.monitor.generate_monitoring_report()
            
            # 보고서 출력
            print(f"\n📊 모니터링 보고서 ({report['monitoring_period']}):")
            print(f"  생성 시간: {report['generated_at'][:19]}")
            
            # 시스템 메트릭
            print(f"\n🖥️ 시스템 메트릭:")
            for metric_name, summary in report["system_metrics"].items():
                if "error" not in summary:
                    print(f"  {metric_name}: 평균 {summary['mean']:.1f}%, "
                          f"최대 {summary['max']:.1f}%, 현재 {summary['current']:.1f}%")
            
            # 쿼리 성능
            print(f"\n⚡ 쿼리 성능:")
            query_perf = report["query_performance"]
            if "total_queries" in query_perf:
                print(f"  총 쿼리: {query_perf['total_queries']}")
                print(f"  평균 응답시간: {query_perf['avg_execution_time']*1000:.2f}ms")
                print(f"  P95 응답시간: {query_perf['p95_execution_time']*1000:.2f}ms")
                print(f"  QPS: {query_perf['qps']:.1f}")
                print(f"  느린 쿼리율: {query_perf['slow_query_percentage']:.1f}%")
            
            # 느린 쿼리
            slow_queries = report["slow_queries"]
            if slow_queries:
                print(f"\n🐌 상위 느린 쿼리:")
                for i, query in enumerate(slow_queries[:3], 1):
                    print(f"  {i}. {query['query_type']}: {query['execution_time']*1000:.2f}ms "
                          f"(결과: {query['result_count']}개)")
            
            # 권장사항
            print(f"\n💡 권장사항:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
            
            print("\n" + "=" * 80)
            print(" 📈 모니터링 모범 사례")
            print("=" * 80)
            
            print("\n🎯 핵심 메트릭:")
            print("  📊 성능 지표:")
            print("    • QPS (Queries Per Second): 처리량")
            print("    • 응답 시간: P50, P95, P99 지연시간")
            print("    • 오류율: 실패한 쿼리 비율")
            print("    • 처리량: 초당 처리된 데이터량")
            
            print("\n  🖥️ 시스템 리소스:")
            print("    • CPU 사용률: 프로세서 부하")
            print("    • 메모리 사용률: RAM 사용량")
            print("    • 디스크 I/O: 저장소 성능")
            print("    • 네트워크: 대역폭 사용량")
            
            print("\n🚨 알림 전략:")
            print("  ⚡ 실시간 알림:")
            print("    • 임계값 기반: CPU > 80%, Memory > 85%")
            print("    • 트렌드 기반: 성능 저하 패턴 감지")
            print("    • 이상 탐지: 비정상적인 패턴 감지")
            
            print("\n  📧 알림 채널:")
            print("    • 이메일: 중요한 장애 알림")
            print("    • Slack/Teams: 팀 협업 채널")
            print("    • SMS: 긴급 상황 알림")
            print("    • 대시보드: 실시간 시각화")
            
            print("\n📊 대시보드 구성:")
            print("  🔍 실시간 모니터링:")
            print("    • 시스템 개요: 전체 상태 한눈에")
            print("    • 성능 그래프: 시계열 데이터")
            print("    • 로그 분석: 오류 및 경고 추적")
            
            print("\n  📈 분석 도구:")
            print("    • 성능 트렌드: 장기 패턴 분석")
            print("    • 용량 계획: 리소스 예측")
            print("    • 비교 분석: 기간별 성능 비교")
            
            print("\n🛠️ 최적화 권장사항:")
            print("  ⚡ 성능 최적화:")
            print("    • 인덱스 튜닝: 검색 성능 향상")
            print("    • 쿼리 최적화: 비효율적 쿼리 개선")
            print("    • 캐싱 전략: 반복 쿼리 최적화")
            
            print("\n  📐 용량 관리:")
            print("    • 스토리지 최적화: 압축 및 정리")
            print("    • 메모리 관리: 효율적 할당")
            print("    • 스케일링: 수평/수직 확장")
            
            # 모니터링 중지
            self.monitor.stop_monitoring()
            
            # 정리
            print("\n🧹 테스트 환경 정리 중...")
            utility.drop_collection("monitoring_test_collection")
            print("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")
        
        finally:
            if self.monitor:
                self.monitor.stop_monitoring()
            self.milvus_conn.disconnect()
            
        print(f"\n🎉 모니터링 및 메트릭스 실습 완료!")
        print("\n💡 학습 포인트:")
        print("  • 실시간 시스템 메트릭 수집 및 모니터링")
        print("  • 쿼리 성능 추적 및 분석")
        print("  • 알림 시스템 구축 및 임계값 관리")
        print("  • 모니터링 대시보드 및 보고서 생성")
        print("\n🎊 4단계 고급 기능 및 최적화 완료!")
        print("  ✅ 성능 최적화")
        print("  ✅ 고급 인덱싱") 
        print("  ✅ 분산 처리 및 확장성")
        print("  ✅ 실시간 스트리밍")
        print("  ✅ 백업 및 복구")
        print("  ✅ 모니터링 및 메트릭스")

def main():
    """메인 실행 함수"""
    monitoring_manager = MonitoringManager()
    monitoring_manager.run_monitoring_demo()

if __name__ == "__main__":
    main() 