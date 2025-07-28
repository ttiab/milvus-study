#!/usr/bin/env python3
"""
Milvus 유사도 메트릭 실습 예제

이 스크립트는 다양한 유사도 메트릭의 작동 방식을 실제로 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityMetricsDemo:
    """유사도 메트릭 데모 클래스"""
    
    def __init__(self):
        self.demo_vectors = self._generate_demo_vectors()
        
    def _generate_demo_vectors(self) -> List[np.ndarray]:
        """데모용 벡터 생성"""
        vectors = [
            np.array([1.0, 0.0, 0.0]),      # 기준 벡터
            np.array([2.0, 0.0, 0.0]),      # 같은 방향, 다른 크기
            np.array([0.0, 1.0, 0.0]),      # 수직
            np.array([1.0, 1.0, 0.0]),      # 45도
            np.array([-1.0, 0.0, 0.0]),     # 반대 방향
            np.array([0.5, 0.5, 0.707]),    # 3차원
        ]
        return vectors
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """유클리드 거리 계산"""
        return np.linalg.norm(vec1 - vec2)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product != 0 else 0
    
    def inner_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """내적 계산"""
        return np.dot(vec1, vec2)
    
    def hamming_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> int:
        """해밍 거리 계산 (이진 벡터용)"""
        return np.sum(vec1 != vec2)
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """자카드 유사도 계산"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    
    def demonstrate_float_metrics(self):
        """부동소수점 벡터 메트릭 시연"""
        print("\n" + "="*60)
        print("부동소수점 벡터 유사도 메트릭 비교")
        print("="*60)
        
        base_vector = self.demo_vectors[0]
        print(f"기준 벡터: {base_vector}")
        print("-" * 60)
        
        for i, vec in enumerate(self.demo_vectors[1:], 1):
            print(f"\n벡터 {i}: {vec}")
            
            l2_dist = self.euclidean_distance(base_vector, vec)
            cosine_sim = self.cosine_similarity(base_vector, vec)
            inner_prod = self.inner_product(base_vector, vec)
            
            print(f"  L2 거리:      {l2_dist:.4f}")
            print(f"  코사인 유사도: {cosine_sim:.4f}")
            print(f"  내적:         {inner_prod:.4f}")
    
    def demonstrate_binary_metrics(self):
        """이진 벡터 메트릭 시연"""
        print("\n" + "="*60)
        print("이진 벡터 유사도 메트릭 비교")
        print("="*60)
        
        # 이진 벡터 생성
        binary_vectors = [
            np.array([1, 0, 1, 1, 0]),
            np.array([1, 1, 1, 0, 0]),
            np.array([0, 1, 0, 1, 1]),
            np.array([1, 0, 1, 1, 1]),
        ]
        
        base_binary = binary_vectors[0]
        print(f"기준 이진 벡터: {base_binary}")
        print("-" * 60)
        
        for i, vec in enumerate(binary_vectors[1:], 1):
            print(f"\n이진 벡터 {i}: {vec}")
            
            hamming_dist = self.hamming_distance(base_binary, vec)
            print(f"  해밍 거리: {hamming_dist}")
    
    def demonstrate_set_metrics(self):
        """집합 기반 메트릭 시연"""
        print("\n" + "="*60)
        print("집합 기반 유사도 메트릭 비교")
        print("="*60)
        
        # 집합 데이터 (사용자 관심사 예제)
        user_interests = [
            {"영화", "음악", "책", "여행"},
            {"영화", "음악", "게임", "스포츠"},
            {"책", "여행", "요리", "사진"},
            {"영화", "책", "여행", "음악"},
        ]
        
        base_set = user_interests[0]
        print(f"기준 사용자 관심사: {base_set}")
        print("-" * 60)
        
        for i, interests in enumerate(user_interests[1:], 1):
            print(f"\n사용자 {i} 관심사: {interests}")
            
            jaccard_sim = self.jaccard_similarity(base_set, interests)
            print(f"  자카드 유사도: {jaccard_sim:.4f}")
    
    def visualize_metrics(self):
        """메트릭 시각화"""
        print("\n" + "="*60)
        print("유사도 메트릭 시각화")
        print("="*60)
        
        # 2D 벡터로 시각화 (처음 2차원만 사용)
        vectors_2d = [vec[:2] for vec in self.demo_vectors]
        base_vec = vectors_2d[0]
        
        # 메트릭 계산
        l2_distances = []
        cosine_similarities = []
        inner_products = []
        
        for vec in vectors_2d[1:]:
            l2_distances.append(self.euclidean_distance(base_vec, vec))
            cosine_similarities.append(self.cosine_similarity(base_vec, vec))
            inner_products.append(self.inner_product(base_vec, vec))
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 벡터 시각화
        axes[0, 0].quiver([0]*len(vectors_2d), [0]*len(vectors_2d), 
                         [v[0] for v in vectors_2d], [v[1] for v in vectors_2d],
                         angles='xy', scale_units='xy', scale=1,
                         color=['red'] + ['blue']*(len(vectors_2d)-1))
        axes[0, 0].set_xlim(-2, 3)
        axes[0, 0].set_ylim(-2, 2)
        axes[0, 0].set_title('벡터 시각화')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        # 메트릭 비교 막대그래프
        x_labels = [f'Vec{i}' for i in range(1, len(vectors_2d))]
        
        axes[0, 1].bar(x_labels, l2_distances, color='skyblue')
        axes[0, 1].set_title('L2 거리')
        axes[0, 1].set_ylabel('거리')
        
        axes[1, 0].bar(x_labels, cosine_similarities, color='lightgreen')
        axes[1, 0].set_title('코사인 유사도')
        axes[1, 0].set_ylabel('유사도')
        
        axes[1, 1].bar(x_labels, inner_products, color='lightcoral')
        axes[1, 1].set_title('내적')
        axes[1, 1].set_ylabel('내적 값')
        
        plt.tight_layout()
        plt.savefig('similarity_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("그래프가 'similarity_metrics_comparison.png'로 저장되었습니다.")
        plt.show()
    
    def demonstrate_normalization_effect(self):
        """정규화 효과 시연"""
        print("\n" + "="*60)
        print("벡터 정규화가 메트릭에 미치는 영향")
        print("="*60)
        
        # 크기가 다른 같은 방향 벡터들
        vectors = [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([10.0, 10.0]),
        ]
        
        print("정규화 전:")
        for i, vec in enumerate(vectors):
            print(f"벡터 {i}: {vec}")
            if i > 0:
                cosine_sim = self.cosine_similarity(vectors[0], vec)
                inner_prod = self.inner_product(vectors[0], vec)
                print(f"  vs 벡터 0 - 코사인: {cosine_sim:.4f}, 내적: {inner_prod:.4f}")
        
        # 정규화
        normalized_vectors = [vec / np.linalg.norm(vec) for vec in vectors]
        
        print("\n정규화 후:")
        for i, vec in enumerate(normalized_vectors):
            print(f"정규화된 벡터 {i}: {vec}")
            if i > 0:
                cosine_sim = self.cosine_similarity(normalized_vectors[0], vec)
                inner_prod = self.inner_product(normalized_vectors[0], vec)
                print(f"  vs 벡터 0 - 코사인: {cosine_sim:.4f}, 내적: {inner_prod:.4f}")

class MilvusMetricsDemo:
    """Milvus에서 실제 메트릭 사용 데모"""
    
    def __init__(self):
        self.collection_name = "metrics_demo"
        self.dim = 128
        
    def connect_to_milvus(self):
        """Milvus 연결"""
        try:
            connections.connect("default", host="localhost", port="19530")
            logger.info("Milvus에 성공적으로 연결되었습니다.")
            return True
        except Exception as e:
            logger.error(f"Milvus 연결 실패: {e}")
            return False
    
    def create_collection_with_metric(self, metric_type: str):
        """특정 메트릭으로 컬렉션 생성"""
        # 기존 컬렉션 삭제
        try:
            collection = Collection(self.collection_name)
            collection.drop()
        except:
            pass
        
        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, f"{metric_type} 메트릭 데모 컬렉션")
        
        # 컬렉션 생성
        collection = Collection(self.collection_name, schema)
        
        # 인덱스 생성
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        logger.info(f"{metric_type} 메트릭으로 컬렉션 생성 완료")
        return collection
    
    def insert_demo_data(self, collection: Collection, num_vectors: int = 1000):
        """데모 데이터 삽입"""
        # 랜덤 벡터 생성
        vectors = np.random.rand(num_vectors, self.dim).astype(np.float32)
        
        # 일부 벡터는 정규화
        for i in range(0, num_vectors, 2):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
        
        # 데이터 삽입
        entities = [vectors.tolist()]
        collection.insert(entities)
        collection.flush()
        
        logger.info(f"{num_vectors}개의 벡터를 삽입했습니다.")
        return vectors
    
    def compare_search_results(self, collection: Collection, query_vector: np.ndarray, metric_type: str):
        """검색 결과 비교"""
        collection.load()
        
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=5
        )
        
        print(f"\n{metric_type} 메트릭 검색 결과:")
        for i, result in enumerate(results[0]):
            print(f"  순위 {i+1}: ID={result.id}, 점수={result.distance:.4f}")
        
        collection.release()
        return results
    
    def run_metric_comparison(self):
        """다양한 메트릭으로 검색 비교"""
        if not self.connect_to_milvus():
            print("Milvus가 실행되지 않아 메트릭 비교를 건너뜁니다.")
            return
        
        metrics = ["L2", "COSINE", "IP"]
        query_vector = np.random.rand(self.dim).astype(np.float32)
        
        print("\n" + "="*60)
        print("Milvus에서 다양한 메트릭 비교")
        print("="*60)
        print(f"쿼리 벡터 크기: {np.linalg.norm(query_vector):.4f}")
        
        for metric in metrics:
            try:
                collection = self.create_collection_with_metric(metric)
                self.insert_demo_data(collection, 100)
                self.compare_search_results(collection, query_vector, metric)
                
                # 컬렉션 정리
                collection.drop()
                
            except Exception as e:
                logger.error(f"{metric} 메트릭 테스트 중 오류: {e}")

def main():
    """메인 함수"""
    print("Milvus 유사도 메트릭 실습 시작")
    print("="*60)
    
    # 1. 기본 메트릭 데모
    demo = SimilarityMetricsDemo()
    demo.demonstrate_float_metrics()
    demo.demonstrate_binary_metrics()
    demo.demonstrate_set_metrics()
    demo.demonstrate_normalization_effect()
    
    # 2. 시각화 (matplotlib이 설치된 경우)
    try:
        demo.visualize_metrics()
    except ImportError:
        print("matplotlib이 설치되지 않아 시각화를 건너뜁니다.")
        print("시각화를 원한다면 'pip install matplotlib seaborn'을 실행하세요.")
    
    # 3. Milvus 실제 사용 데모
    milvus_demo = MilvusMetricsDemo()
    milvus_demo.run_metric_comparison()
    
    print("\n" + "="*60)
    print("실습 완료!")
    print("="*60)
    print("\n주요 학습 포인트:")
    print("1. L2 거리는 벡터 간 실제 공간적 거리를 측정")
    print("2. 코사인 유사도는 벡터의 방향성만 고려")
    print("3. 내적은 방향과 크기를 모두 고려")
    print("4. 정규화된 벡터에서는 내적과 코사인이 동일")
    print("5. 데이터 특성에 따라 적절한 메트릭 선택이 중요")

if __name__ == "__main__":
    main() 