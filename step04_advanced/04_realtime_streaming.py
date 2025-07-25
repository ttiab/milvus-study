#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚡ Milvus 실시간 스트리밍 실습

이 스크립트는 Milvus의 실시간 스트리밍 및 데이터 파이프라인을 실습합니다:
- 실시간 데이터 스트림 시뮬레이션
- 스트리밍 데이터 처리 및 변환
- 실시간 벡터화 및 삽입
- 스트림 기반 검색 및 알림
- 데이터 품질 모니터링
- 실시간 분석 및 대시보드
"""

import os
import sys
import time
import logging
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
import json
import random
from concurrent.futures import ThreadPoolExecutor
import uuid

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingDataSource:
    """실시간 데이터 소스 시뮬레이터"""
    
    def __init__(self, source_type: str = "social_media"):
        self.source_type = source_type
        self.is_streaming = False
        self.stream_queue = queue.Queue(maxsize=1000)
        self.data_templates = self._get_data_templates()
        
    def _get_data_templates(self) -> Dict[str, List[str]]:
        """데이터 템플릿 정의"""
        templates = {
            "social_media": [
                "Breaking news: {topic} trending worldwide #news",
                "Just discovered amazing {topic} tips! Love it! #lifestyle",
                "New {topic} technology is game changing! #tech",
                "Review: {topic} product exceeded expectations #review",
                "Discussion: What do you think about {topic}? #opinion"
            ],
            "news": [
                "{topic} market shows significant growth this quarter",
                "Scientists announce breakthrough in {topic} research", 
                "Government announces new {topic} policy initiatives",
                "Industry experts predict {topic} trends for next year",
                "International summit focuses on {topic} cooperation"
            ],
            "ecommerce": [
                "Customer review: {topic} product quality excellent",
                "New {topic} product launch creates market buzz",
                "Flash sale: {topic} items up to 50% off today",
                "Customer service inquiry about {topic} features",
                "Product recommendation: {topic} perfect for summer"
            ],
            "iot_sensors": [
                "Sensor alert: {topic} temperature anomaly detected",
                "IoT device: {topic} status update normal operations",
                "Smart home: {topic} automation trigger activated",
                "Industrial sensor: {topic} pressure reading elevated",
                "Vehicle telemetry: {topic} performance metric updated"
            ]
        }
        return templates.get(self.source_type, templates["social_media"])
    
    def generate_stream_record(self) -> Dict[str, Any]:
        """스트림 레코드 생성"""
        topics = ["AI", "blockchain", "cloud", "mobile", "security", "analytics", "automation", "sustainability"]
        topic = random.choice(topics)
        template = random.choice(self.data_templates)
        
        # 현실적인 스트림 데이터 생성
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),  # 밀리초
            "source": self.source_type,
            "topic": topic,
            "content": template.format(topic=topic),
            "language": random.choice(["en", "ko", "ja", "zh"]),
            "sentiment": random.choice(["positive", "negative", "neutral"]),
            "confidence": round(random.uniform(0.6, 1.0), 3),
            "priority": random.randint(1, 5),
            "location": random.choice(["US", "EU", "ASIA", "GLOBAL"]),
            "user_id": f"user_{random.randint(1000, 9999)}",
            "metadata": {
                "device_type": random.choice(["mobile", "desktop", "tablet"]),
                "platform": random.choice(["ios", "android", "web"]),
                "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}"
            }
        }
        
        return record
    
    def start_streaming(self, records_per_second: float = 2.0):
        """스트리밍 시작"""
        self.is_streaming = True
        
        def stream_producer():
            while self.is_streaming:
                try:
                    record = self.generate_stream_record()
                    self.stream_queue.put(record, timeout=1)
                    time.sleep(1.0 / records_per_second)
                except queue.Full:
                    logger.warning("Stream queue is full, dropping record")
                except Exception as e:
                    logger.error(f"Stream producer error: {e}")
        
        self.producer_thread = threading.Thread(target=stream_producer)
        self.producer_thread.start()
    
    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False
        if hasattr(self, 'producer_thread'):
            self.producer_thread.join()
    
    def get_records(self, max_records: int = 10) -> List[Dict[str, Any]]:
        """레코드 배치 가져오기"""
        records = []
        for _ in range(max_records):
            try:
                record = self.stream_queue.get_nowait()
                records.append(record)
            except queue.Empty:
                break
        return records

class StreamProcessor:
    """스트림 처리기"""
    
    def __init__(self, vector_utils: VectorUtils):
        self.vector_utils = vector_utils
        self.processing_stats = defaultdict(int)
        self.error_count = 0
        self.processed_count = 0
        
    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """단일 레코드 처리"""
        try:
            # 데이터 검증
            if not self._validate_record(record):
                self.error_count += 1
                return None
            
            # 텍스트 정규화
            content = self._normalize_text(record["content"])
            
            # 벡터화
            vector = self._vectorize_content(content)
            if vector is None:
                self.error_count += 1
                return None
            
            # 처리된 레코드 생성
            processed_record = {
                "stream_id": record["id"],
                "content": content,
                "source": record["source"],
                "topic": record["topic"],
                "language": record["language"],
                "sentiment": record["sentiment"],
                "confidence": record["confidence"],
                "priority": record["priority"],
                "location": record["location"],
                "user_id": record["user_id"],
                "timestamp": record["timestamp"],
                "processed_at": int(time.time() * 1000),
                "vector": vector.tolist(),
                "metadata": json.dumps(record["metadata"])
            }
            
            self.processed_count += 1
            self.processing_stats[record["source"]] += 1
            
            return processed_record
            
        except Exception as e:
            logger.error(f"Record processing error: {e}")
            self.error_count += 1
            return None
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """레코드 유효성 검사"""
        required_fields = ["id", "content", "source", "timestamp"]
        
        for field in required_fields:
            if field not in record or not record[field]:
                logger.warning(f"Missing or empty field: {field}")
                return False
        
        # 콘텐츠 길이 검사
        if len(record["content"]) < 10 or len(record["content"]) > 1000:
            logger.warning(f"Content length out of range: {len(record['content'])}")
            return False
        
        return True
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 기본적인 텍스트 정리
        text = text.strip()
        text = ' '.join(text.split())  # 다중 공백 제거
        
        # 해시태그 정규화
        import re
        text = re.sub(r'#\w+', lambda m: m.group().lower(), text)
        
        return text
    
    def _vectorize_content(self, content: str) -> Optional[np.ndarray]:
        """콘텐츠 벡터화"""
        try:
            vectors = self.vector_utils.text_to_vector(content)
            vector = vectors[0] if len(vectors.shape) > 1 else vectors
            return vector
        except Exception as e:
            logger.error(f"Vectorization error: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        total_processed = sum(self.processing_stats.values())
        error_rate = self.error_count / max(total_processed + self.error_count, 1)
        
        return {
            "processed_count": total_processed,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "source_distribution": dict(self.processing_stats)
        }

class RealTimeSearchEngine:
    """실시간 검색 엔진"""
    
    def __init__(self, collection: Collection, vector_utils: VectorUtils):
        self.collection = collection
        self.vector_utils = vector_utils
        self.search_history = deque(maxlen=1000)
        self.alert_rules = []
        
    def add_alert_rule(self, name: str, query: str, threshold: float = 0.8, 
                      max_results: int = 5):
        """알림 규칙 추가"""
        rule = {
            "name": name,
            "query": query,
            "threshold": threshold,
            "max_results": max_results,
            "last_triggered": 0
        }
        self.alert_rules.append(rule)
    
    def real_time_search(self, query: str, filters: Optional[str] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """실시간 검색 실행"""
        try:
            # 쿼리 벡터화
            query_vectors = self.vector_utils.text_to_vector(query)
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 검색 파라미터
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            
            # 검색 실행
            start_time = time.time()
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filters,
                output_fields=["content", "source", "topic", "sentiment", "timestamp", "location"]
            )
            search_time = time.time() - start_time
            
            # 결과 처리
            processed_results = []
            if results and len(results[0]) > 0:
                for hit in results[0]:
                    result = {
                        "content": hit.entity.get('content'),
                        "source": hit.entity.get('source'),
                        "topic": hit.entity.get('topic'),
                        "sentiment": hit.entity.get('sentiment'),
                        "timestamp": hit.entity.get('timestamp'),
                        "location": hit.entity.get('location'),
                        "similarity": float(hit.distance),
                        "score": 1.0 - float(hit.distance)  # 유사도 점수로 변환
                    }
                    processed_results.append(result)
            
            # 검색 이력 저장
            search_record = {
                "query": query,
                "filters": filters,
                "timestamp": int(time.time() * 1000),
                "search_time": search_time,
                "result_count": len(processed_results),
                "top_score": processed_results[0]["score"] if processed_results else 0
            }
            self.search_history.append(search_record)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Real-time search error: {e}")
            return []
    
    def check_alerts(self, new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """알림 규칙 확인"""
        triggered_alerts = []
        current_time = int(time.time())
        
        for rule in self.alert_rules:
            # 최근 트리거 후 최소 간격 확인 (30초)
            if current_time - rule["last_triggered"] < 30:
                continue
            
            try:
                # 알림 쿼리 실행
                results = self.real_time_search(
                    query=rule["query"],
                    limit=rule["max_results"]
                )
                
                # 임계값 확인
                high_score_results = [r for r in results if r["score"] >= rule["threshold"]]
                
                if high_score_results:
                    alert = {
                        "rule_name": rule["name"],
                        "query": rule["query"],
                        "triggered_at": current_time,
                        "matching_records": len(high_score_results),
                        "top_matches": high_score_results[:3],
                        "max_score": max(r["score"] for r in high_score_results)
                    }
                    triggered_alerts.append(alert)
                    rule["last_triggered"] = current_time
                    
            except Exception as e:
                logger.error(f"Alert check error for rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """검색 분석 정보"""
        if not self.search_history:
            return {"message": "No search history available"}
        
        recent_searches = list(self.search_history)[-100:]  # 최근 100개
        
        # 통계 계산
        avg_search_time = np.mean([s["search_time"] for s in recent_searches])
        avg_result_count = np.mean([s["result_count"] for s in recent_searches])
        avg_top_score = np.mean([s["top_score"] for s in recent_searches if s["top_score"] > 0])
        
        # 검색 빈도 분석
        search_frequency = len(recent_searches) / (time.time() - recent_searches[0]["timestamp"] / 1000) if recent_searches else 0
        
        return {
            "total_searches": len(self.search_history),
            "recent_searches": len(recent_searches),
            "avg_search_time": avg_search_time,
            "avg_result_count": avg_result_count,
            "avg_top_score": avg_top_score,
            "search_frequency": search_frequency,
            "active_alert_rules": len(self.alert_rules)
        }

class RealTimeStreamingManager:
    """실시간 스트리밍 관리 클래스"""
    
    def __init__(self):
        self.milvus_conn = MilvusConnection()
        self.vector_utils = VectorUtils()
        self.stream_sources = {}
        self.stream_processor = StreamProcessor(self.vector_utils)
        self.search_engine = None
        self.is_processing = False
        self.processing_stats = defaultdict(int)
        
    def create_streaming_collection(self, collection_name: str) -> Collection:
        """스트리밍용 컬렉션 생성"""
        print(f"⚡ 스트리밍 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스트리밍 최적화 스키마
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="stream_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="sentiment", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="priority", dtype=DataType.INT32),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="processed_at", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Real-time streaming collection optimized for high throughput"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        
        # 스트리밍 최적화 인덱스 생성
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        print(f"  ✅ 스트리밍 컬렉션 생성 완료")
        return collection
    
    def setup_data_sources(self):
        """데이터 소스 설정"""
        print("🔧 데이터 소스 설정 중...")
        
        source_configs = [
            {"type": "social_media", "rate": 3.0},
            {"type": "news", "rate": 1.5},
            {"type": "ecommerce", "rate": 2.0},
            {"type": "iot_sensors", "rate": 5.0}
        ]
        
        for config in source_configs:
            source = StreamingDataSource(config["type"])
            self.stream_sources[config["type"]] = {
                "source": source,
                "rate": config["rate"]
            }
            print(f"  ✅ {config['type']} 소스 ({config['rate']} rec/sec)")
        
        print(f"  ✅ {len(self.stream_sources)}개 데이터 소스 설정 완료")
    
    def start_streaming_pipeline(self, collection: Collection, duration: int = 60):
        """스트리밍 파이프라인 시작"""
        print(f"🚀 스트리밍 파이프라인 시작 (지속시간: {duration}초)...")
        
        self.is_processing = True
        collection.load()
        
        # 검색 엔진 설정
        self.search_engine = RealTimeSearchEngine(collection, self.vector_utils)
        
        # 알림 규칙 설정
        self.search_engine.add_alert_rule("High Priority AI", "artificial intelligence priority", 0.7)
        self.search_engine.add_alert_rule("Security Alert", "security threat vulnerability", 0.8)
        self.search_engine.add_alert_rule("Breaking News", "breaking news urgent", 0.75)
        
        # 데이터 소스 시작
        for source_info in self.stream_sources.values():
            source_info["source"].start_streaming(source_info["rate"])
        
        # 처리 워커 시작
        def processing_worker():
            batch_size = 10
            batch_interval = 2.0  # 2초마다 배치 처리
            
            while self.is_processing:
                try:
                    # 모든 소스에서 레코드 수집
                    all_records = []
                    for source_info in self.stream_sources.values():
                        records = source_info["source"].get_records(batch_size // len(self.stream_sources))
                        all_records.extend(records)
                    
                    if all_records:
                        # 배치 처리
                        processed_records = []
                        for record in all_records:
                            processed = self.stream_processor.process_record(record)
                            if processed:
                                processed_records.append(processed)
                        
                        if processed_records:
                            # Milvus에 삽입
                            self._insert_batch(collection, processed_records)
                            
                            # 알림 확인
                            alerts = self.search_engine.check_alerts(processed_records)
                            if alerts:
                                self._handle_alerts(alerts)
                            
                            self.processing_stats["total_processed"] += len(processed_records)
                            self.processing_stats["last_batch_size"] = len(processed_records)
                            self.processing_stats["last_processed"] = int(time.time())
                    
                    time.sleep(batch_interval)
                    
                except Exception as e:
                    logger.error(f"Processing worker error: {e}")
        
        # 워커 스레드 시작
        self.processing_thread = threading.Thread(target=processing_worker)
        self.processing_thread.start()
        
        # 진행 상황 모니터링
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            
            print(f"  ⏳ 진행 중... {elapsed}/{duration}초 "
                  f"(처리됨: {self.processing_stats.get('total_processed', 0)}개)")
            
            time.sleep(10)  # 10초마다 상태 출력
        
        # 스트리밍 중지
        self.stop_streaming_pipeline()
        
        print(f"  ✅ 스트리밍 파이프라인 완료")
    
    def _insert_batch(self, collection: Collection, records: List[Dict[str, Any]]):
        """배치 데이터 삽입"""
        if not records:
            return
        
        # 데이터 구조화
        data = [
            [r["stream_id"] for r in records],
            [r["content"] for r in records],
            [r["source"] for r in records],
            [r["topic"] for r in records],
            [r["language"] for r in records],
            [r["sentiment"] for r in records],
            [r["confidence"] for r in records],
            [r["priority"] for r in records],
            [r["location"] for r in records],
            [r["user_id"] for r in records],
            [r["timestamp"] for r in records],
            [r["processed_at"] for r in records],
            [r["vector"] for r in records],
            [r["metadata"] for r in records]
        ]
        
        collection.insert(data)
    
    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """알림 처리"""
        for alert in alerts:
            print(f"  🚨 알림: {alert['rule_name']} - {alert['matching_records']}개 매칭 "
                  f"(최고 점수: {alert['max_score']:.3f})")
    
    def stop_streaming_pipeline(self):
        """스트리밍 파이프라인 중지"""
        self.is_processing = False
        
        # 데이터 소스 중지
        for source_info in self.stream_sources.values():
            source_info["source"].stop_streaming()
        
        # 처리 스레드 대기
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5)
    
    def demonstrate_real_time_search(self, collection: Collection):
        """실시간 검색 데모"""
        print("\n🔍 실시간 검색 데모...")
        
        if not self.search_engine:
            self.search_engine = RealTimeSearchEngine(collection, self.vector_utils)
        
        # 다양한 실시간 검색 시나리오
        search_scenarios = [
            {
                "name": "AI 관련 최신 동향",
                "query": "artificial intelligence machine learning",
                "filters": "priority >= 3"
            },
            {
                "name": "보안 이슈 모니터링", 
                "query": "security vulnerability threat",
                "filters": "sentiment == 'negative'"
            },
            {
                "name": "긍정적 제품 리뷰",
                "query": "product review excellent quality",
                "filters": "sentiment == 'positive' and source == 'ecommerce'"
            },
            {
                "name": "실시간 뉴스 트래킹",
                "query": "breaking news important update",
                "filters": "source == 'news'"
            }
        ]
        
        for scenario in search_scenarios:
            print(f"\n  📋 시나리오: {scenario['name']}")
            print(f"    쿼리: '{scenario['query']}'")
            print(f"    필터: {scenario.get('filters', 'None')}")
            
            start_time = time.time()
            results = self.search_engine.real_time_search(
                query=scenario["query"],
                filters=scenario.get("filters"),
                limit=5
            )
            search_time = time.time() - start_time
            
            print(f"    검색 시간: {search_time*1000:.2f}ms")
            print(f"    결과 수: {len(results)}")
            
            if results:
                print(f"    상위 결과:")
                for i, result in enumerate(results[:3], 1):
                    print(f"      {i}. 점수: {result['score']:.3f}, "
                          f"소스: {result['source']}, 주제: {result['topic']}")
                    print(f"         내용: {result['content'][:60]}...")
            
            time.sleep(1)  # 검색 간 간격
    
    def streaming_analytics_dashboard(self, collection: Collection) -> Dict[str, Any]:
        """스트리밍 분석 대시보드"""
        print("\n📊 스트리밍 분석 대시보드...")
        
        # 컬렉션 통계
        try:
            collection.load()
            total_entities = collection.num_entities
            collection.release()
        except:
            total_entities = 0
        
        # 처리 통계
        processor_stats = self.stream_processor.get_stats()
        
        # 검색 분석
        search_analytics = {}
        if self.search_engine:
            search_analytics = self.search_engine.get_search_analytics()
        
        # 대시보드 데이터
        dashboard_data = {
            "streaming_metrics": {
                "total_entities": total_entities,
                "active_sources": len(self.stream_sources),
                "processing_rate": self.processing_stats.get("total_processed", 0),
                "last_processed": self.processing_stats.get("last_processed", 0)
            },
            "data_quality": {
                "processed_records": processor_stats.get("processed_count", 0),
                "error_count": processor_stats.get("error_count", 0),
                "error_rate": processor_stats.get("error_rate", 0),
                "source_distribution": processor_stats.get("source_distribution", {})
            },
            "search_performance": search_analytics,
            "system_health": {
                "pipeline_status": "running" if self.is_processing else "stopped",
                "memory_usage": "normal",  # 시뮬레이션
                "latency_p95": "< 50ms",   # 시뮬레이션
                "throughput": "2.5k rec/min"  # 시뮬레이션
            }
        }
        
        # 대시보드 출력
        print(f"  📈 스트리밍 메트릭스:")
        metrics = dashboard_data["streaming_metrics"]
        print(f"    총 엔티티 수: {metrics['total_entities']:,}")
        print(f"    활성 소스 수: {metrics['active_sources']}")
        print(f"    처리된 레코드: {metrics['processing_rate']:,}")
        
        print(f"\n  🎯 데이터 품질:")
        quality = dashboard_data["data_quality"]
        print(f"    성공률: {(1-quality['error_rate'])*100:.1f}%")
        print(f"    오류 수: {quality['error_count']}")
        print(f"    소스별 분포: {quality['source_distribution']}")
        
        if search_analytics:
            print(f"\n  🔍 검색 성능:")
            print(f"    총 검색 수: {search_analytics.get('total_searches', 0)}")
            print(f"    평균 응답 시간: {search_analytics.get('avg_search_time', 0)*1000:.2f}ms")
            print(f"    평균 결과 수: {search_analytics.get('avg_result_count', 0):.1f}")
        
        print(f"\n  🏥 시스템 상태:")
        health = dashboard_data["system_health"]
        print(f"    파이프라인: {health['pipeline_status']}")
        print(f"    메모리 사용: {health['memory_usage']}")
        print(f"    지연 시간 P95: {health['latency_p95']}")
        print(f"    처리량: {health['throughput']}")
        
        return dashboard_data
    
    def run_realtime_streaming_demo(self):
        """실시간 스트리밍 종합 데모"""
        print("⚡ Milvus 실시간 스트리밍 실습")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Milvus 연결
            self.milvus_conn.connect()
            print("✅ Milvus 연결 성공\n")
            
            print("=" * 80)
            print(" 🏗️ 스트리밍 인프라 구축")
            print("=" * 80)
            
            # 스트리밍 컬렉션 생성
            collection = self.create_streaming_collection("realtime_streaming")
            
            # 데이터 소스 설정
            self.setup_data_sources()
            
            print("\n" + "=" * 80)
            print(" 🚀 실시간 데이터 파이프라인")
            print("=" * 80)
            
            # 스트리밍 파이프라인 시작 (30초 동안)
            self.start_streaming_pipeline(collection, duration=30)
            
            print("\n" + "=" * 80)
            print(" 🔍 실시간 검색 및 알림")
            print("=" * 80)
            
            # 실시간 검색 데모
            self.demonstrate_real_time_search(collection)
            
            print("\n" + "=" * 80)
            print(" 📊 스트리밍 분석 및 모니터링")
            print("=" * 80)
            
            # 분석 대시보드
            dashboard_data = self.streaming_analytics_dashboard(collection)
            
            print("\n" + "=" * 80)
            print(" 💡 실시간 스트리밍 권장사항")
            print("=" * 80)
            
            print("\n⚡ 스트리밍 아키텍처 설계:")
            print("  📊 데이터 수집 계층:")
            print("    • Kafka/Pulsar: 고성능 메시지 큐")
            print("    • Schema Registry: 스키마 버전 관리")
            print("    • Connect API: 다양한 소스 연동")
            print("    • 백프레셔 제어: 과부하 방지")
            
            print("\n  🔄 스트림 처리:")
            print("    • 배치 처리: 효율적인 벡터화")
            print("    • 파이프라인 병렬화: 처리량 증대")
            print("    • 오류 처리: 재시도 및 DLQ")
            print("    • 백업 큐: 장애 복구")
            
            print("\n  🔍 실시간 검색:")
            print("    • 인메모리 인덱스: 빠른 검색")
            print("    • 증분 인덱싱: 실시간 업데이트")
            print("    • 캐시 계층: 반복 쿼리 최적화")
            print("    • 알림 시스템: 이벤트 기반 응답")
            
            print("\n  📈 모니터링 및 운영:")
            print("    • 실시간 메트릭스: 처리량, 지연시간, 오류율")
            print("    • 알림 시스템: 임계값 기반 자동 알림")
            print("    • 로그 집계: 중앙화된 로그 관리")
            print("    • 성능 대시보드: 시각화된 모니터링")
            
            print("\n  🛠️ 최적화 전략:")
            print("    • 배치 크기 조정: 처리량 vs 지연시간")
            print("    • 벡터 차원 최적화: 정확도 vs 성능")
            print("    • 파티셔닝: 병렬 처리 증대")
            print("    • 압축: 네트워크 및 저장소 효율")
            
            # 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            utility.drop_collection("realtime_streaming")
            print("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")
        
        finally:
            self.stop_streaming_pipeline()
            self.milvus_conn.disconnect()
            
        print(f"\n🎉 실시간 스트리밍 실습 완료!")
        print("\n💡 학습 포인트:")
        print("  • 실시간 데이터 파이프라인 구축 및 관리")
        print("  • 스트리밍 데이터 처리 및 벡터화")
        print("  • 실시간 검색 및 알림 시스템")
        print("  • 스트리밍 분석 및 모니터링 대시보드")
        print("\n🚀 다음 단계:")
        print("  python step04_advanced/05_backup_recovery.py")

def main():
    """메인 실행 함수"""
    streaming_manager = RealTimeStreamingManager()
    streaming_manager.run_realtime_streaming_demo()

if __name__ == "__main__":
    main() 