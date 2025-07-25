#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 3단계: 이미지 유사도 검색 시스템

실제 이미지 검색 애플리케이션을 구현합니다:
- 이미지 기반 유사도 검색
- 시각적 특징 분석
- 이미지 분류 및 태깅
- 역방향 이미지 검색
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType

# 이미지 처리를 위한 추가 라이브러리
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL(Pillow) 라이브러리가 설치되지 않았습니다. 이미지 생성 기능이 제한됩니다.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib 라이브러리가 설치되지 않았습니다. 시각화 기능이 제한됩니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageSimilaritySearchEngine:
    """이미지 유사도 검색 엔진"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        self.image_data_dir = "temp_images"  # 임시 이미지 저장 디렉토리
        
        # 이미지 저장 디렉토리 생성
        if not os.path.exists(self.image_data_dir):
            os.makedirs(self.image_data_dir)
    
    def create_image_collection(self, collection_name: str = "image_search") -> Collection:
        """이미지 검색용 컬렉션 생성"""
        print(f"\n📁 이미지 검색 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 - 이미지 메타데이터 포함
        fields = [
            FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="color_scheme", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="style", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="format", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="width", dtype=DataType.INT64),
            FieldSchema(name="height", dtype=DataType.INT64),
            FieldSchema(name="file_size", dtype=DataType.INT64),  # bytes
            FieldSchema(name="created_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="upload_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="photographer", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="camera_model", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="like_count", dtype=DataType.INT64),
            FieldSchema(name="download_count", dtype=DataType.INT64),
            FieldSchema(name="quality_score", dtype=DataType.FLOAT),
            FieldSchema(name="is_featured", dtype=DataType.BOOL),
            FieldSchema(name="is_public", dtype=DataType.BOOL),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),  # CLIP 모델 차원
            FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # 텍스트 임베딩
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="고급 이미지 검색 컬렉션",
            enable_dynamic_field=True
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def generate_sample_images(self, count: int = 200) -> List[Dict[str, Any]]:
        """샘플 이미지 메타데이터 생성 (실제 이미지 파일 대신 메타데이터만)"""
        print(f"\n📊 샘플 이미지 메타데이터 {count}개 생성 중...")
        
        # 이미지 카테고리
        categories = [
            "Photography", "Digital Art", "Illustration", "Nature", "Architecture", 
            "Portrait", "Landscape", "Abstract", "Technology", "Food", "Fashion", 
            "Sports", "Travel", "Animals", "Business"
        ]
        
        # 스타일
        styles = [
            "realistic", "abstract", "minimalist", "vintage", "modern", "artistic",
            "documentary", "studio", "candid", "macro", "panoramic", "black_white",
            "colorful", "monochrome", "HDR", "long_exposure"
        ]
        
        # 색상 스키마
        color_schemes = [
            "warm_tones", "cool_tones", "monochromatic", "complementary", "triadic",
            "analogous", "neutral", "vibrant", "pastel", "dark", "bright", "earth_tones"
        ]
        
        # 포맷
        formats = ["JPEG", "PNG", "TIFF", "RAW", "GIF", "WebP"]
        
        # 카메라 모델
        camera_models = [
            "Canon EOS R5", "Nikon D850", "Sony A7R IV", "Fujifilm X-T4",
            "Leica Q2", "Olympus OM-D E-M1 Mark III", "Panasonic GH5",
            "iPhone 14 Pro", "Samsung Galaxy S23", "Google Pixel 7"
        ]
        
        # 위치
        locations = [
            "Seoul, Korea", "Tokyo, Japan", "New York, USA", "Paris, France",
            "London, UK", "Sydney, Australia", "Toronto, Canada", "Berlin, Germany",
            "Rome, Italy", "Barcelona, Spain", "Amsterdam, Netherlands", "Prague, Czech"
        ]
        
        # 사진작가
        photographers = [f"Photographer_{i}" for i in range(1, 31)]
        
        images = []
        
        for i in range(count):
            category = np.random.choice(categories)
            style = np.random.choice(styles)
            color_scheme = np.random.choice(color_schemes)
            format_type = np.random.choice(formats)
            camera = np.random.choice(camera_models)
            location = np.random.choice(locations)
            photographer = np.random.choice(photographers)
            
            # 파일명 생성
            filename = f"{category.lower()}_{style}_{i+1:04d}.{format_type.lower()}"
            
            # 제목 생성
            title_templates = [
                f"{style.title()} {category} Photography",
                f"Beautiful {category} in {style.title()} Style",
                f"{category} - {style.title()} Composition",
                f"Professional {category} Shot",
                f"Artistic {category} Capture"
            ]
            title = np.random.choice(title_templates)
            
            # 설명 생성
            description = f"A stunning {style} {category.lower()} photograph featuring {color_scheme.replace('_', ' ')} color palette. "
            description += f"Captured in {location} using {camera}. "
            description += f"This {format_type} image showcases {['excellent composition', 'perfect lighting', 'artistic vision', 'technical precision'][i%4]}."
            
            # 태그 생성
            base_tags = [category.lower(), style, color_scheme]
            if np.random.random() > 0.5:
                base_tags.extend([location.split(',')[0].lower(), camera.split()[0].lower()])
            tags = ", ".join(base_tags)
            
            # 이미지 차원 생성 (다양한 해상도)
            resolutions = [(1920, 1080), (3840, 2160), (2560, 1440), (1280, 720), (4000, 3000), (6000, 4000)]
            width, height = resolutions[np.random.randint(0, len(resolutions))]
            
            # 파일 크기 (대략적 계산)
            pixel_count = width * height
            if format_type == "RAW":
                file_size = pixel_count * 3  # 대략 3 bytes per pixel for RAW
            elif format_type == "TIFF":
                file_size = pixel_count * 2
            else:
                file_size = pixel_count // 10  # JPEG 압축 고려
            
            # 메타데이터 생성
            quality_score = np.random.uniform(2.0, 5.0)
            view_count = np.random.randint(50, 5000)
            like_count = int(view_count * np.random.uniform(0.05, 0.25))
            download_count = int(view_count * np.random.uniform(0.02, 0.15))
            
            # 날짜 생성
            created_date = f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            upload_date = created_date if np.random.random() > 0.3 else f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            
            image_data = {
                "filename": filename,
                "title": title,
                "description": description,
                "category": category,
                "tags": tags,
                "color_scheme": color_scheme,
                "style": style,
                "format": format_type,
                "width": width,
                "height": height,
                "file_size": file_size,
                "created_date": created_date,
                "upload_date": upload_date,
                "photographer": photographer,
                "location": location,
                "camera_model": camera,
                "view_count": view_count,
                "like_count": like_count,
                "download_count": download_count,
                "quality_score": quality_score,
                "is_featured": np.random.random() > 0.8,  # 20% 확률로 featured
                "is_public": np.random.random() > 0.1  # 90% 확률로 public
            }
            
            images.append(image_data)
        
        print(f"  ✅ {count}개 이미지 메타데이터 생성 완료")
        return images
    
    def generate_mock_image_vectors(self, images: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """모의 이미지 벡터 생성 (실제 이미지가 없으므로 설명 기반으로 벡터 생성)"""
        print(f"\n🖼️ 이미지 벡터 생성 중...")
        
        # 이미지 설명 기반으로 텍스트 벡터 생성 (실제 이미지 벡터 대신)
        descriptions = []
        image_descriptions = []
        
        for img in images:
            # 텍스트 설명 벡터용
            text_desc = f"{img['title']} {img['description']}"
            descriptions.append(text_desc)
            
            # 이미지 벡터용 (더 시각적 특징 중심)
            visual_desc = f"{img['category']} {img['style']} {img['color_scheme']} photography"
            image_descriptions.append(visual_desc)
        
        # 텍스트 벡터 생성 (384차원)
        print("  텍스트 설명 벡터화 중...")
        description_vectors = self.vector_utils.texts_to_vectors(descriptions)
        
        # 모의 이미지 벡터 생성 (512차원으로 확장)
        print("  모의 이미지 벡터 생성 중...")
        base_vectors = self.vector_utils.texts_to_vectors(image_descriptions)
        
        # 384차원을 512차원으로 확장 (패딩으로 간단히 처리)
        image_vectors = []
        for vec in base_vectors:
            # 384차원 벡터를 512차원으로 확장
            extended_vec = np.pad(vec, (0, 512 - len(vec)), mode='constant', constant_values=0)
            # 약간의 노이즈 추가로 다양성 확보
            extended_vec += np.random.normal(0, 0.1, size=extended_vec.shape)
            image_vectors.append(extended_vec)
        
        image_vectors = np.array(image_vectors)
        
        print(f"  ✅ 벡터 생성 완료 - 이미지: {image_vectors.shape}, 설명: {description_vectors.shape}")
        return image_vectors, description_vectors
    
    def insert_images(self, collection: Collection, images: List[Dict[str, Any]]) -> None:
        """이미지 데이터 삽입"""
        print(f"\n💾 이미지 데이터 삽입 중...")
        
        # 벡터 생성
        image_vectors, description_vectors = self.generate_mock_image_vectors(images)
        
        # 데이터를 2단계 패턴으로 구성 (List[List])
        data = [
            [img["filename"] for img in images],
            [img["title"] for img in images],
            [img["description"] for img in images],
            [img["category"] for img in images],
            [img["tags"] for img in images],
            [img["color_scheme"] for img in images],
            [img["style"] for img in images],
            [img["format"] for img in images],
            [img["width"] for img in images],
            [img["height"] for img in images],
            [img["file_size"] for img in images],
            [img["created_date"] for img in images],
            [img["upload_date"] for img in images],
            [img["photographer"] for img in images],
            [img["location"] for img in images],
            [img["camera_model"] for img in images],
            [img["view_count"] for img in images],
            [img["like_count"] for img in images],
            [img["download_count"] for img in images],
            [img["quality_score"] for img in images],
            [img["is_featured"] for img in images],
            [img["is_public"] for img in images],
            image_vectors.tolist(),
            description_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(images)
        print(f"  ✅ {total_inserted}개 이미지 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 이미지 삽입 완료")
    
    def create_indexes(self, collection: Collection) -> None:
        """인덱스 생성"""
        print(f"\n🔍 인덱스 생성 중...")
        
        # 이미지 벡터 인덱스
        image_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        # 설명 벡터 인덱스
        desc_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        print("  이미지 벡터 인덱스 생성 중...")
        collection.create_index("image_vector", image_index_params)
        
        print("  설명 벡터 인덱스 생성 중...")
        collection.create_index("description_vector", desc_index_params)
        
        print(f"  ✅ 모든 인덱스 생성 완료")
    
    def visual_similarity_search_demo(self, collection: Collection) -> None:
        """시각적 유사도 검색 데모"""
        print("\n" + "="*80)
        print(" 🎨 시각적 유사도 검색 데모")
        print("="*80)
        
        collection.load()
        
        # 다양한 시각적 검색 시나리오
        search_scenarios = [
            {
                "description": "자연 풍경 사진 검색",
                "query": "beautiful nature landscape mountain forest scenic",
                "category_filter": "Nature"
            },
            {
                "description": "도시 건축 사진 검색",
                "query": "modern architecture urban building cityscape design",
                "category_filter": "Architecture"
            },
            {
                "description": "인물 사진 검색",
                "query": "portrait professional people person face photography",
                "category_filter": "Portrait"
            },
            {
                "description": "추상 예술 작품 검색",
                "query": "abstract art creative colorful artistic expression",
                "category_filter": "Abstract"
            }
        ]
        
        for i, scenario in enumerate(search_scenarios, 1):
            print(f"\n{i}. {scenario['description']}")
            print(f"   검색 쿼리: '{scenario['query']}'")
            
            # 텍스트 기반 이미지 검색 (CLIP 스타일)
            query_vectors = self.vector_utils.text_to_vector(scenario['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 벡터를 512차원으로 확장
            extended_query = np.pad(query_vector, (0, 512 - len(query_vector)), mode='constant', constant_values=0)
            
            # 이미지 벡터 기반 검색
            start_time = time.time()
            results = collection.search(
                data=[extended_query.tolist()],
                anns_field="image_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["title", "category", "style", "color_scheme", "photographer", "quality_score", "view_count"]
            )
            search_time = time.time() - start_time
            
            print(f"   검색 시간: {search_time:.4f}초")
            print(f"   결과 수: {len(results[0])}")
            
            for j, hit in enumerate(results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')}")
                print(f"        스타일: {entity.get('style')}, 색상: {entity.get('color_scheme')}")
                print(f"        작가: {entity.get('photographer')}, 품질: {entity.get('quality_score'):.2f}")
                print(f"        유사도: {similarity:.3f}, 조회수: {entity.get('view_count')}")
            
            # 카테고리 필터링 검색
            if scenario.get('category_filter'):
                print(f"\n   📁 카테고리 필터링 검색 ({scenario['category_filter']})")
                
                filtered_results = collection.search(
                    data=[extended_query.tolist()],
                    anns_field="image_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 200}},
                    limit=3,
                    expr=f"category == '{scenario['category_filter']}'",
                    output_fields=["title", "style", "location", "camera_model"]
                )
                
                print(f"   필터링된 결과 수: {len(filtered_results[0])}")
                for j, hit in enumerate(filtered_results[0]):
                    similarity = 1 - hit.distance
                    entity = hit.entity
                    print(f"     {j+1}. {entity.get('title')}")
                    print(f"        스타일: {entity.get('style')}, 위치: {entity.get('location')}")
                    print(f"        카메라: {entity.get('camera_model')}, 유사도: {similarity:.3f}")
    
    def reverse_image_search_demo(self, collection: Collection) -> None:
        """역방향 이미지 검색 데모"""
        print("\n" + "="*80)
        print(" 🔍 역방향 이미지 검색 데모")
        print("="*80)
        
        collection.load()
        
        # 기존 이미지를 "업로드된 이미지"로 가정하여 유사한 이미지 찾기
        print("사용자가 업로드한 이미지와 유사한 이미지를 찾습니다...")
        
        # 임의의 이미지를 "업로드된 이미지"로 선택
        sample_results = collection.search(
            data=[[0.1] * 512],  # 임의의 벡터
            anns_field="image_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=1,
            output_fields=["title", "category", "style", "color_scheme", "image_vector", "description", "tags"]
        )
        
        if sample_results and len(sample_results[0]) > 0:
            uploaded_image = sample_results[0][0].entity
            uploaded_vector = uploaded_image.get('image_vector')
            
            print(f"\n📤 업로드된 이미지:")
            print(f"   제목: {uploaded_image.get('title')}")
            print(f"   카테고리: {uploaded_image.get('category')}")
            print(f"   스타일: {uploaded_image.get('style')}")
            print(f"   색상 스키마: {uploaded_image.get('color_scheme')}")
            print(f"   설명: {uploaded_image.get('description')[:100]}...")
            print(f"   태그: {uploaded_image.get('tags')}")
            
            # 유사한 이미지 검색
            print(f"\n🔍 시각적으로 유사한 이미지 검색:")
            
            similar_images = collection.search(
                data=[uploaded_vector],
                anns_field="image_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=6,  # 첫 번째는 자기 자신이므로 6개 검색
                output_fields=["title", "category", "style", "color_scheme", "photographer", "quality_score"]
            )
            
            # 자기 자신 제외
            for j, hit in enumerate(similar_images[0][1:], 1):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"   {j}. {entity.get('title')}")
                print(f"      카테고리: {entity.get('category')}, 스타일: {entity.get('style')}")
                print(f"      색상: {entity.get('color_scheme')}, 작가: {entity.get('photographer')}")
                print(f"      품질점수: {entity.get('quality_score'):.2f}, 유사도: {similarity:.3f}")
            
            # 같은 스타일의 이미지만 검색
            style = uploaded_image.get('style')
            print(f"\n🎨 같은 스타일({style}) 이미지 검색:")
            
            style_results = collection.search(
                data=[uploaded_vector],
                anns_field="image_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=4,
                expr=f"style == '{style}'",
                output_fields=["title", "photographer", "location", "quality_score"]
            )
            
            for j, hit in enumerate(style_results[0][1:], 1):  # 자기 자신 제외
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"   {j}. {entity.get('title')}")
                print(f"      작가: {entity.get('photographer')}, 위치: {entity.get('location')}")
                print(f"      품질점수: {entity.get('quality_score'):.2f}, 유사도: {similarity:.3f}")
    
    def text_to_image_search_demo(self, collection: Collection) -> None:
        """텍스트-이미지 교차 검색 데모"""
        print("\n" + "="*80)
        print(" 💬 텍스트-이미지 교차 검색 데모")
        print("="*80)
        
        collection.load()
        
        # 자연어 설명으로 이미지 검색
        text_queries = [
            {
                "query": "해질녘의 아름다운 산 풍경",
                "description": "한국어 자연어로 풍경 이미지 검색"
            },
            {
                "query": "모던한 건축물과 도시의 야경",
                "description": "건축/도시 관련 이미지 검색"
            },
            {
                "query": "따뜻한 색감의 인물 사진",
                "description": "색감과 스타일을 지정한 인물 사진 검색"
            },
            {
                "query": "창의적이고 추상적인 디지털 아트",
                "description": "예술 작품 검색"
            }
        ]
        
        for i, case in enumerate(text_queries, 1):
            print(f"\n{i}. {case['description']}")
            print(f"   텍스트 쿼리: '{case['query']}'")
            
            # 설명 벡터로 검색 (텍스트-이미지 매칭)
            query_vectors = self.vector_utils.text_to_vector(case['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 설명 벡터 기반 검색
            start_time = time.time()
            desc_results = collection.search(
                data=[query_vector.tolist()],
                anns_field="description_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["title", "description", "category", "tags", "quality_score", "like_count"]
            )
            search_time = time.time() - start_time
            
            print(f"   검색 시간: {search_time:.4f}초")
            print(f"   매칭된 이미지 수: {len(desc_results[0])}")
            
            for j, hit in enumerate(desc_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"\n     🖼️ 이미지 {j+1} (유사도: {similarity:.3f})")
                print(f"     제목: {entity.get('title')}")
                print(f"     설명: {entity.get('description')[:80]}...")
                print(f"     카테고리: {entity.get('category')}, 품질: {entity.get('quality_score'):.2f}")
                print(f"     태그: {entity.get('tags')}")
                print(f"     좋아요: {entity.get('like_count')}")
            
            # 고품질 이미지만 필터링
            print(f"\n   ⭐ 고품질 이미지 필터링 (품질점수 >= 4.0)")
            
            quality_results = collection.search(
                data=[query_vector.tolist()],
                anns_field="description_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=3,
                expr="quality_score >= 4.0 and is_public == True",
                output_fields=["title", "quality_score", "view_count", "photographer"]
            )
            
            print(f"   고품질 이미지 수: {len(quality_results[0])}")
            for j, hit in enumerate(quality_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')}")
                print(f"        품질점수: {entity.get('quality_score'):.2f}, 조회수: {entity.get('view_count')}")
                print(f"        작가: {entity.get('photographer')}, 유사도: {similarity:.3f}")
    
    def advanced_search_demo(self, collection: Collection) -> None:
        """고급 이미지 검색 데모"""
        print("\n" + "="*80)
        print(" 🚀 고급 이미지 검색 데모")
        print("="*80)
        
        collection.load()
        
        # 복합 조건 검색
        print("\n1. 복합 조건 검색")
        print("   조건: 인기 있는 풍경 사진 (조회수 > 1000, 좋아요 > 50)")
        
        # 풍경 관련 쿼리
        landscape_query = "beautiful landscape nature scenic view"
        query_vectors = self.vector_utils.text_to_vector(landscape_query)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
        
        # 복합 필터 검색
        complex_results = collection.search(
            data=[query_vector.tolist()],
            anns_field="description_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=5,
            expr="view_count > 1000 and like_count > 50 and is_public == True",
            output_fields=["title", "category", "view_count", "like_count", "photographer", "location"]
        )
        
        print(f"   결과 수: {len(complex_results[0])}")
        for j, hit in enumerate(complex_results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        위치: {entity.get('location')}, 작가: {entity.get('photographer')}")
            print(f"        조회수: {entity.get('view_count')}, 좋아요: {entity.get('like_count')}")
            print(f"        유사도: {similarity:.3f}")
        
        # 2. 해상도 기반 검색
        print("\n2. 고해상도 이미지 검색")
        print("   조건: 4K 이상 해상도 (3840x2160 이상)")
        
        # 고해상도 이미지 검색
        hd_results = collection.search(
            data=[query_vector.tolist()],
            anns_field="description_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=5,
            expr="width >= 3840 and height >= 2160",
            output_fields=["title", "width", "height", "file_size", "format", "camera_model"]
        )
        
        print(f"   고해상도 이미지 수: {len(hd_results[0])}")
        for j, hit in enumerate(hd_results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            file_size_mb = entity.get('file_size') / (1024 * 1024)
            print(f"     {j+1}. {entity.get('title')}")
            print(f"        해상도: {entity.get('width')}x{entity.get('height')}")
            print(f"        파일크기: {file_size_mb:.1f}MB, 포맷: {entity.get('format')}")
            print(f"        카메라: {entity.get('camera_model')}, 유사도: {similarity:.3f}")
        
        # 3. 시간 기반 검색
        print("\n3. 최신 업로드 이미지 검색")
        print("   조건: 2024년 하반기 업로드")
        
        recent_results = collection.search(
            data=[query_vector.tolist()],
            anns_field="description_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=4,
            expr="upload_date >= '2024-07-01'",
            output_fields=["title", "upload_date", "created_date", "is_featured", "quality_score"]
        )
        
        print(f"   최신 이미지 수: {len(recent_results[0])}")
        for j, hit in enumerate(recent_results[0]):
            similarity = 1 - hit.distance
            entity = hit.entity
            featured = "⭐" if entity.get('is_featured') else ""
            print(f"     {j+1}. {entity.get('title')} {featured}")
            print(f"        업로드: {entity.get('upload_date')}, 촬영: {entity.get('created_date')}")
            print(f"        품질점수: {entity.get('quality_score'):.2f}, 유사도: {similarity:.3f}")
    
    def image_analytics_demo(self, collection: Collection) -> None:
        """이미지 분석 및 통계 데모"""
        print("\n" + "="*80)
        print(" 📊 이미지 컬렉션 분석 데모")
        print("="*80)
        
        collection.load()
        
        # 카테고리별 통계
        print("\n1. 카테고리별 인기도 분석")
        
        categories = ["Photography", "Nature", "Architecture", "Portrait", "Abstract"]
        
        for category in categories:
            # 카테고리별 평균 조회수와 품질점수 조회
            category_results = collection.search(
                data=[[0.0] * 384],  # 더미 벡터
                anns_field="description_vector",
                param={"metric_type": "COSINE", "params": {"ef": 50}},
                limit=10,
                expr=f"category == '{category}'",
                output_fields=["view_count", "like_count", "quality_score", "is_featured"]
            )
            
            if category_results and len(category_results[0]) > 0:
                def safe_get_view_count(hit):
                    try:
                        return hit.entity.get('view_count') or 0
                    except:
                        return 0
                
                def safe_get_quality_score(hit):
                    try:
                        return hit.entity.get('quality_score') or 0
                    except:
                        return 0
                
                def safe_get_is_featured(hit):
                    try:
                        return hit.entity.get('is_featured') or False
                    except:
                        return False
                
                view_counts = [safe_get_view_count(hit) for hit in category_results[0]]
                quality_scores = [safe_get_quality_score(hit) for hit in category_results[0]]
                featured_count = sum(1 for hit in category_results[0] if safe_get_is_featured(hit))
                
                avg_views = np.mean(view_counts) if view_counts else 0
                avg_quality = np.mean(quality_scores) if quality_scores else 0
                
                print(f"   {category:12} - 평균 조회수: {avg_views:6.0f}, 평균 품질: {avg_quality:.2f}, 추천 이미지: {featured_count}")
        
        # 2. 스타일별 분포
        print("\n2. 인기 스타일 분석")
        
        styles = ["realistic", "abstract", "minimalist", "vintage", "modern"]
        
        for style in styles:
            style_results = collection.search(
                data=[[0.0] * 384],
                anns_field="description_vector",
                param={"metric_type": "COSINE", "params": {"ef": 50}},
                limit=5,
                expr=f"style == '{style}'",
                output_fields=["like_count", "download_count", "quality_score"]
            )
            
            if style_results and len(style_results[0]) > 0:
                def safe_get_like_count(hit):
                    try:
                        return hit.entity.get('like_count') or 0
                    except:
                        return 0
                
                def safe_get_download_count(hit):
                    try:
                        return hit.entity.get('download_count') or 0
                    except:
                        return 0
                
                like_counts = [safe_get_like_count(hit) for hit in style_results[0]]
                download_counts = [safe_get_download_count(hit) for hit in style_results[0]]
                
                avg_likes = np.mean(like_counts) if like_counts else 0
                avg_downloads = np.mean(download_counts) if download_counts else 0
                
                print(f"   {style:12} - 평균 좋아요: {avg_likes:5.0f}, 평균 다운로드: {avg_downloads:5.0f}")
        
        # 3. 고품질 이미지 검색
        print("\n3. 최고 품질 이미지 TOP 5")
        
        top_quality_results = collection.search(
            data=[[0.0] * 384],
            anns_field="description_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=20,  # 더 많이 검색해서 품질순 정렬
            expr="quality_score >= 4.5",
            output_fields=["title", "quality_score", "category", "photographer", "view_count", "like_count"]
        )
        
        if top_quality_results and len(top_quality_results[0]) > 0:
            # 품질점수 기준으로 정렬
            def get_quality_score_for_sort(x):
                try:
                    return x.entity.get('quality_score') or 0
                except:
                    return 0
            sorted_results = sorted(top_quality_results[0], 
                                  key=get_quality_score_for_sort, 
                                  reverse=True)
            
            for j, hit in enumerate(sorted_results[:5], 1):
                entity = hit.entity
                print(f"   {j}. {entity.get('title')}")
                print(f"      품질점수: {entity.get('quality_score'):.2f}, 카테고리: {entity.get('category')}")
                print(f"      작가: {entity.get('photographer')}")
                print(f"      조회수: {entity.get('view_count')}, 좋아요: {entity.get('like_count')}")


def main():
    """메인 실행 함수"""
    print("🚀 이미지 유사도 검색 시스템 실습")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 이미지 검색 엔진 초기화
            search_engine = ImageSimilaritySearchEngine(conn)
            
            # 이미지 컬렉션 생성
            print("\n" + "="*80)
            print(" 🖼️ 이미지 검색 시스템 구축")
            print("="*80)
            
            # 컬렉션 생성
            image_collection = search_engine.create_image_collection()
            
            # 샘플 이미지 데이터 생성 및 삽입
            images = search_engine.generate_sample_images(200)
            search_engine.insert_images(image_collection, images)
            
            # 인덱스 생성
            search_engine.create_indexes(image_collection)
            
            # 다양한 검색 데모 실행
            search_engine.visual_similarity_search_demo(image_collection)
            search_engine.reverse_image_search_demo(image_collection)
            search_engine.text_to_image_search_demo(image_collection)
            search_engine.advanced_search_demo(image_collection)
            search_engine.image_analytics_demo(image_collection)
            
            # 컬렉션 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            if utility.has_collection("image_search"):
                utility.drop_collection("image_search")
            print("✅ 정리 완료")
            
        print("\n🎉 이미지 유사도 검색 시스템 실습 완료!")
        
        print("\n💡 학습 포인트:")
        print("  • 다양한 시각적 특징을 활용한 이미지 검색")
        print("  • 텍스트-이미지 교차 검색으로 직관적 검색 경험")
        print("  • 역방향 이미지 검색으로 중복/유사 이미지 탐지")
        print("  • 메타데이터 기반 고급 필터링")
        print("  • 이미지 컬렉션 분석 및 인사이트 도출")
        
        print("\n🚀 다음 단계:")
        print("  python step03_use_cases/03_recommendation_system.py")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 