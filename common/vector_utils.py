"""
벡터 변환 및 처리 유틸리티

텍스트와 이미지를 벡터로 변환하고, 벡터 연산을 수행하는 유틸리티 클래스
"""

import os
import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from PIL import Image
import logging

# ML 라이브러리 임포트
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorUtils:
    """벡터 변환 및 처리 유틸리티 클래스"""
    
    def __init__(self):
        """초기화"""
        self.text_model = None
        self.image_model = None
        self.image_processor = None
        
        # 기본 모델 설정
        self.default_text_model = os.getenv(
            "DEFAULT_TEXT_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.default_image_model = os.getenv(
            "DEFAULT_IMAGE_MODEL",
            "openai/clip-vit-base-patch32"
        )
    
    def load_text_model(self, model_name: Optional[str] = None) -> bool:
        """
        텍스트 임베딩 모델 로드
        
        Args:
            model_name: 모델 이름 (기본값: sentence-transformers/all-MiniLM-L6-v2)
            
        Returns:
            bool: 로드 성공 여부
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers가 설치되지 않았습니다.")
            return False
            
        try:
            model_name = model_name or self.default_text_model
            self.text_model = SentenceTransformer(model_name)
            logger.info(f"텍스트 모델 로드 완료: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"텍스트 모델 로드 실패: {e}")
            return False
    
    def load_image_model(self, model_name: Optional[str] = None) -> bool:
        """
        이미지 임베딩 모델 로드
        
        Args:
            model_name: 모델 이름 (기본값: openai/clip-vit-base-patch32)
            
        Returns:
            bool: 로드 성공 여부
        """
        if not CLIP_AVAILABLE:
            logger.error("transformers 라이브러리가 설치되지 않았습니다.")
            return False
            
        try:
            model_name = model_name or self.default_image_model
            self.image_model = CLIPModel.from_pretrained(model_name)
            self.image_processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"이미지 모델 로드 완료: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"이미지 모델 로드 실패: {e}")
            return False
    
    def text_to_vector(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        텍스트를 벡터로 변환
        
        Args:
            texts: 변환할 텍스트 (단일 문자열 또는 리스트)
            normalize: 벡터 정규화 여부
            
        Returns:
            np.ndarray: 변환된 벡터
        """
        if self.text_model is None:
            self.load_text_model()
            
        if self.text_model is None:
            raise RuntimeError("텍스트 모델이 로드되지 않았습니다.")
        
        try:
            # 단일 문자열을 리스트로 변환
            if isinstance(texts, str):
                texts = [texts]
            
            # 벡터 변환
            vectors = self.text_model.encode(
                texts, 
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            logger.info(f"{len(texts)}개 텍스트를 벡터로 변환 완료")
            return vectors
            
        except Exception as e:
            logger.error(f"텍스트 벡터 변환 실패: {e}")
            raise
    
    def texts_to_vectors(
        self, 
        texts: List[str], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        여러 텍스트를 벡터로 변환 (text_to_vector의 alias)
        
        Args:
            texts: 변환할 텍스트 리스트
            normalize: 벡터 정규화 여부
            
        Returns:
            np.ndarray: 변환된 벡터들
        """
        return self.text_to_vector(texts, normalize)
    
    def image_to_vector(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        이미지를 벡터로 변환
        
        Args:
            images: 변환할 이미지 (경로, PIL Image, 또는 리스트)
            normalize: 벡터 정규화 여부
            
        Returns:
            np.ndarray: 변환된 벡터
        """
        if self.image_model is None or self.image_processor is None:
            self.load_image_model()
            
        if self.image_model is None:
            raise RuntimeError("이미지 모델이 로드되지 않았습니다.")
        
        try:
            # 단일 이미지를 리스트로 변환
            if not isinstance(images, list):
                images = [images]
            
            # 이미지 로드 및 전처리
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    # 파일 경로인 경우
                    img = Image.open(img).convert('RGB')
                elif not isinstance(img, Image.Image):
                    raise ValueError(f"지원하지 않는 이미지 타입: {type(img)}")
                
                processed_images.append(img)
            
            # 벡터 변환
            inputs = self.image_processor(
                images=processed_images, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)
                
                if normalize:
                    image_features = torch.nn.functional.normalize(
                        image_features, p=2, dim=1
                    )
            
            vectors = image_features.numpy()
            logger.info(f"{len(processed_images)}개 이미지를 벡터로 변환 완료")
            return vectors
            
        except Exception as e:
            logger.error(f"이미지 벡터 변환 실패: {e}")
            raise
    
    def calculate_similarity(
        self, 
        vector1: np.ndarray, 
        vector2: np.ndarray, 
        metric: str = "cosine"
    ) -> float:
        """
        두 벡터 간의 유사도 계산
        
        Args:
            vector1: 첫 번째 벡터
            vector2: 두 번째 벡터  
            metric: 유사도 메트릭 (cosine, euclidean, dot)
            
        Returns:
            float: 유사도 점수
        """
        try:
            if metric == "cosine":
                # 코사인 유사도
                dot_product = np.dot(vector1, vector2)
                norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                return dot_product / norms if norms != 0 else 0.0
                
            elif metric == "euclidean":
                # 유클리드 거리 (거리이므로 유사도로 변환)
                distance = np.linalg.norm(vector1 - vector2)
                return 1.0 / (1.0 + distance)
                
            elif metric == "dot":
                # 내적
                return np.dot(vector1, vector2)
                
            else:
                raise ValueError(f"지원하지 않는 메트릭: {metric}")
                
        except Exception as e:
            logger.error(f"유사도 계산 실패: {e}")
            raise
    
    def batch_similarity(
        self, 
        query_vector: np.ndarray, 
        target_vectors: np.ndarray, 
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        쿼리 벡터와 여러 타겟 벡터들 간의 유사도 일괄 계산
        
        Args:
            query_vector: 쿼리 벡터 (1D)
            target_vectors: 타겟 벡터들 (2D: [num_vectors, vector_dim])
            metric: 유사도 메트릭
            
        Returns:
            np.ndarray: 유사도 점수 배열
        """
        try:
            if metric == "cosine":
                # 벡터 정규화
                query_norm = query_vector / np.linalg.norm(query_vector)
                target_norms = target_vectors / np.linalg.norm(
                    target_vectors, axis=1, keepdims=True
                )
                
                # 코사인 유사도 계산
                similarities = np.dot(target_norms, query_norm)
                return similarities
                
            elif metric == "euclidean":
                # 유클리드 거리 계산
                distances = np.linalg.norm(
                    target_vectors - query_vector, axis=1
                )
                return 1.0 / (1.0 + distances)
                
            elif metric == "dot":
                # 내적 계산
                return np.dot(target_vectors, query_vector)
                
            else:
                raise ValueError(f"지원하지 않는 메트릭: {metric}")
                
        except Exception as e:
            logger.error(f"배치 유사도 계산 실패: {e}")
            raise
    
    def reduce_dimension(
        self, 
        vectors: np.ndarray, 
        target_dim: int, 
        method: str = "pca"
    ) -> np.ndarray:
        """
        벡터 차원 축소
        
        Args:
            vectors: 입력 벡터들
            target_dim: 목표 차원
            method: 차원 축소 방법 (pca, random)
            
        Returns:
            np.ndarray: 차원이 축소된 벡터들
        """
        try:
            if method == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=target_dim)
                reduced_vectors = pca.fit_transform(vectors)
                logger.info(f"PCA로 차원을 {vectors.shape[1]}에서 {target_dim}으로 축소")
                return reduced_vectors
                
            elif method == "random":
                # 랜덤 프로젝션
                np.random.seed(42)
                projection_matrix = np.random.randn(vectors.shape[1], target_dim)
                reduced_vectors = np.dot(vectors, projection_matrix)
                logger.info(f"랜덤 프로젝션으로 차원 축소")
                return reduced_vectors
                
            else:
                raise ValueError(f"지원하지 않는 차원 축소 방법: {method}")
                
        except Exception as e:
            logger.error(f"차원 축소 실패: {e}")
            raise
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        벡터 정규화
        
        Args:
            vectors: 정규화할 벡터들
            
        Returns:
            np.ndarray: 정규화된 벡터들
        """
        try:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / np.where(norms == 0, 1, norms)
            return normalized
            
        except Exception as e:
            logger.error(f"벡터 정규화 실패: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        로드된 모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        info = {
            "text_model_loaded": self.text_model is not None,
            "image_model_loaded": self.image_model is not None,
            "text_model_name": self.default_text_model if self.text_model else None,
            "image_model_name": self.default_image_model if self.image_model else None
        }
        
        if self.text_model:
            try:
                # 텍스트 모델의 임베딩 차원 확인
                sample_vector = self.text_to_vector("test")
                info["text_embedding_dim"] = sample_vector.shape[-1]
            except:
                info["text_embedding_dim"] = None
        
        return info 