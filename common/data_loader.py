"""
데이터 로딩 및 전처리 유틸리티

다양한 형태의 데이터를 로드하고 전처리하는 유틸리티 클래스
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

# 이미지 처리 라이브러리
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 텍스트 처리 라이브러리
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """데이터 로딩 및 전처리 유틸리티 클래스"""
    
    def __init__(self, base_path: str = "./datasets"):
        """
        초기화
        
        Args:
            base_path: 데이터셋 기본 경로
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 경로 설정
        self.text_path = self.base_path / "text"
        self.image_path = self.base_path / "images"
        self.vector_path = self.base_path / "vectors"
        
        # 디렉토리 생성
        for path in [self.text_path, self.image_path, self.vector_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def load_text_data(
        self, 
        file_path: str, 
        file_type: str = "auto",
        encoding: str = "utf-8"
    ) -> List[Dict[str, Any]]:
        """
        텍스트 데이터 로드
        
        Args:
            file_path: 파일 경로
            file_type: 파일 타입 (json, csv, txt, auto)
            encoding: 파일 인코딩
            
        Returns:
            List[Dict]: 로드된 텍스트 데이터
        """
        try:
            file_path = Path(file_path)
            
            # 파일 타입 자동 감지
            if file_type == "auto":
                file_type = file_path.suffix.lower().lstrip('.')
            
            if file_type == "json":
                return self._load_json(file_path, encoding)
            elif file_type == "csv":
                return self._load_csv(file_path, encoding)
            elif file_type == "txt":
                return self._load_txt(file_path, encoding)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
                
        except Exception as e:
            logger.error(f"텍스트 데이터 로드 실패: {e}")
            return []
    
    def _load_json(self, file_path: Path, encoding: str) -> List[Dict[str, Any]]:
        """JSON 파일 로드"""
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # 리스트가 아닌 경우 리스트로 변환
        if not isinstance(data, list):
            data = [data]
        
        logger.info(f"JSON 파일 로드 완료: {len(data)}개 항목")
        return data
    
    def _load_csv(self, file_path: Path, encoding: str) -> List[Dict[str, Any]]:
        """CSV 파일 로드"""
        df = pd.read_csv(file_path, encoding=encoding)
        data = df.to_dict('records')
        
        logger.info(f"CSV 파일 로드 완료: {len(data)}개 항목")
        return data
    
    def _load_txt(self, file_path: Path, encoding: str) -> List[Dict[str, Any]]:
        """텍스트 파일 로드 (한 줄씩)"""
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        data = [{"text": line.strip(), "id": i} for i, line in enumerate(lines) if line.strip()]
        
        logger.info(f"텍스트 파일 로드 완료: {len(data)}개 항목")
        return data
    
    def load_image_data(
        self, 
        directory_path: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        이미지 데이터 로드
        
        Args:
            directory_path: 이미지 디렉토리 경로
            extensions: 허용할 확장자 목록
            include_metadata: 메타데이터 포함 여부
            
        Returns:
            List[Dict]: 이미지 정보 리스트
        """
        if not PIL_AVAILABLE:
            logger.error("PIL 라이브러리가 설치되지 않았습니다.")
            return []
        
        try:
            directory_path = Path(directory_path)
            image_data = []
            
            for ext in extensions:
                for image_path in directory_path.glob(f"*{ext}"):
                    try:
                        # 이미지 정보 수집
                        image_info = {
                            "path": str(image_path),
                            "filename": image_path.name,
                            "extension": ext
                        }
                        
                        if include_metadata:
                            with Image.open(image_path) as img:
                                image_info.update({
                                    "width": img.width,
                                    "height": img.height,
                                    "mode": img.mode,
                                    "format": img.format,
                                    "size_bytes": image_path.stat().st_size
                                })
                        
                        image_data.append(image_info)
                        
                    except Exception as e:
                        logger.warning(f"이미지 처리 실패 {image_path}: {e}")
                        continue
            
            logger.info(f"이미지 데이터 로드 완료: {len(image_data)}개 파일")
            return image_data
            
        except Exception as e:
            logger.error(f"이미지 데이터 로드 실패: {e}")
            return []
    
    def preprocess_text(
        self, 
        texts: Union[str, List[str]], 
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        language: str = "english"
    ) -> Union[str, List[str]]:
        """
        텍스트 전처리
        
        Args:
            texts: 전처리할 텍스트(들)
            lowercase: 소문자 변환 여부
            remove_punctuation: 구두점 제거 여부
            remove_stopwords: 불용어 제거 여부
            language: 언어 (불용어 제거 시 사용)
            
        Returns:
            전처리된 텍스트(들)
        """
        import re
        import string
        
        def process_single_text(text: str) -> str:
            # 소문자 변환
            if lowercase:
                text = text.lower()
            
            # 구두점 제거
            if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            # 불용어 제거
            if remove_stopwords and NLTK_AVAILABLE:
                try:
                    from nltk.corpus import stopwords
                    from nltk.tokenize import word_tokenize
                    
                    stop_words = set(stopwords.words(language))
                    word_tokens = word_tokenize(text)
                    text = ' '.join([w for w in word_tokens if w not in stop_words])
                except:
                    logger.warning("NLTK 불용어 제거 실패 - 건너뜀")
            
            # 여러 공백을 하나로 통합
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # 단일 텍스트인 경우
        if isinstance(texts, str):
            return process_single_text(texts)
        
        # 리스트인 경우
        return [process_single_text(text) for text in texts]
    
    def create_sample_text_dataset(
        self, 
        size: int = 1000,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        샘플 텍스트 데이터셋 생성
        
        Args:
            size: 생성할 데이터 개수
            save_path: 저장할 파일 경로
            
        Returns:
            List[Dict]: 생성된 샘플 데이터
        """
        sample_texts = [
            "기계 학습은 인공지능의 한 분야입니다.",
            "벡터 데이터베이스는 유사도 검색에 최적화되어 있습니다.",
            "Milvus는 오픈소스 벡터 데이터베이스입니다.",
            "임베딩은 텍스트를 벡터로 변환하는 과정입니다.",
            "딥러닝 모델은 신경망을 기반으로 합니다.",
            "자연어 처리는 컴퓨터가 인간의 언어를 이해하는 기술입니다.",
            "추천 시스템은 사용자의 선호도를 분석합니다.",
            "검색 엔진은 정보 검색을 위한 시스템입니다.",
            "데이터 마이닝은 대용량 데이터에서 패턴을 찾습니다.",
            "컴퓨터 비전은 이미지와 비디오를 분석합니다."
        ]
        
        import random
        random.seed(42)
        
        dataset = []
        for i in range(size):
            # 랜덤하게 텍스트 선택 및 변형
            base_text = random.choice(sample_texts)
            
            # 간단한 변형 추가
            if random.random() < 0.3:
                base_text = f"질문: {base_text}"
            elif random.random() < 0.3:
                base_text = f"설명: {base_text}"
            
            dataset.append({
                "id": i,
                "text": base_text,
                "category": random.choice(["AI", "ML", "NLP", "CV", "DB"]),
                "length": len(base_text)
            })
        
        # 파일 저장
        if save_path:
            save_path = Path(save_path)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"샘플 데이터셋 저장: {save_path}")
        
        logger.info(f"샘플 텍스트 데이터셋 생성 완료: {len(dataset)}개 항목")
        return dataset
    
    def save_vectors(
        self, 
        vectors: np.ndarray, 
        metadata: List[Dict[str, Any]], 
        filename: str
    ) -> bool:
        """
        벡터와 메타데이터 저장
        
        Args:
            vectors: 저장할 벡터들
            metadata: 메타데이터 리스트
            filename: 파일명
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            file_path = self.vector_path / filename
            
            # NumPy 배열로 저장
            np.savez_compressed(
                file_path,
                vectors=vectors,
                metadata=np.array(metadata, dtype=object)
            )
            
            logger.info(f"벡터 데이터 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"벡터 저장 실패: {e}")
            return False
    
    def load_vectors(self, filename: str) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """
        저장된 벡터 데이터 로드
        
        Args:
            filename: 파일명
            
        Returns:
            Tuple: (벡터 배열, 메타데이터 리스트)
        """
        try:
            file_path = self.vector_path / filename
            
            if not file_path.exists():
                logger.error(f"파일이 존재하지 않습니다: {file_path}")
                return None
            
            data = np.load(file_path, allow_pickle=True)
            vectors = data['vectors']
            metadata = data['metadata'].tolist()
            
            logger.info(f"벡터 데이터 로드 완료: {vectors.shape}")
            return vectors, metadata
            
        except Exception as e:
            logger.error(f"벡터 로드 실패: {e}")
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        데이터셋 정보 조회
        
        Returns:
            Dict: 데이터셋 정보
        """
        info = {
            "base_path": str(self.base_path),
            "text_files": len(list(self.text_path.glob("*"))),
            "image_files": len(list(self.image_path.glob("*"))),
            "vector_files": len(list(self.vector_path.glob("*.npz"))),
        }
        
        # 각 경로의 파일 목록
        info["text_file_list"] = [f.name for f in self.text_path.glob("*")]
        info["image_file_list"] = [f.name for f in self.image_path.glob("*")]
        info["vector_file_list"] = [f.name for f in self.vector_path.glob("*.npz")]
        
        return info 