#!/usr/bin/env python3
"""
Milvus 학습 프로젝트 - 3단계: 텍스트 유사도 검색 시스템

실제 텍스트 검색 애플리케이션을 구현합니다:
- 문서 검색 엔진
- Q&A 매칭 시스템  
- 유사 문서 추천
- 시맨틱 검색
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSimilaritySearchEngine:
    """텍스트 유사도 검색 엔진"""
    
    def __init__(self, connection: MilvusConnection):
        self.connection = connection
        self.vector_utils = VectorUtils()
        self.data_loader = DataLoader()
        self.collections = {}
        
    def create_document_collection(self, collection_name: str = "document_search") -> Collection:
        """문서 검색용 컬렉션 생성"""
        print(f"\n📁 문서 검색 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의 - 풍부한 메타데이터 포함
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="word_count", dtype=DataType.INT64),
            FieldSchema(name="reading_time", dtype=DataType.INT64),  # 예상 읽기 시간 (분)
            FieldSchema(name="difficulty_level", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="quality_score", dtype=DataType.FLOAT),
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="like_count", dtype=DataType.INT64),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="summary_vector", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="고급 문서 검색 컬렉션",
            enable_dynamic_field=True
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def create_qa_collection(self, collection_name: str = "qa_pairs") -> Collection:
        """Q&A 매칭용 컬렉션 생성"""
        print(f"\n📁 Q&A 매칭 컬렉션 '{collection_name}' 생성 중...")
        
        # 기존 컬렉션 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"  기존 컬렉션 삭제됨")
        
        # 스키마 정의
        fields = [
            FieldSchema(name="qa_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="confidence_score", dtype=DataType.FLOAT),
            FieldSchema(name="created_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="updated_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="is_verified", dtype=DataType.BOOL),
            FieldSchema(name="usage_count", dtype=DataType.INT64),
            FieldSchema(name="question_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="answer_vector", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Q&A 매칭 컬렉션"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        print(f"  ✅ 컬렉션 생성 완료")
        return collection
    
    def generate_sample_documents(self, count: int = 1000) -> List[Dict[str, Any]]:
        """샘플 문서 데이터 생성"""
        print(f"\n📊 샘플 문서 {count}개 생성 중...")
        
        # 다양한 카테고리와 도메인
        categories = ["Technology", "Science", "Business", "Health", "Education", "Entertainment", "Sports", "Politics"]
        doc_types = ["article", "research_paper", "blog_post", "news", "tutorial", "review", "case_study", "whitepaper"]
        languages = ["ko", "en", "ja", "zh"]
        difficulties = ["beginner", "intermediate", "advanced", "expert"]
        authors = [f"Author_{i}" for i in range(1, 51)]
        
        # 샘플 제목과 내용 템플릿
        title_templates = [
            "{category}에서의 혁신적인 접근 방법",
            "{category} 분야의 최신 동향 분석",
            "{category} 전문가가 말하는 핵심 포인트",
            "{category}의 미래 전망과 기회",
            "{category} 입문자를 위한 완벽 가이드",
            "실무진이 알아야 할 {category} 필수 지식",
            "{category} 성공 사례 심층 분석",
            "{category} 트렌드와 시장 인사이트"
        ]
        
        content_templates = [
            "이 문서는 {category} 분야의 {topic}에 대한 포괄적인 분석을 제공합니다. 최신 연구 결과와 실무 경험을 바탕으로 {detail}에 대해 자세히 설명하며, 실제 적용 사례와 함께 향후 전망을 제시합니다.",
            "{category} 영역에서 {topic}의 중요성이 날로 증가하고 있습니다. 본 연구에서는 {detail}를 중심으로 현재 상황을 분석하고, 전문가 인터뷰와 데이터 분석을 통해 실용적인 인사이트를 도출했습니다.",
            "현대 {category} 환경에서 {topic}는 핵심적인 역할을 담당하고 있습니다. 이 문서는 {detail}에 초점을 맞춰 이론적 배경부터 실제 구현까지 단계별로 설명하며, 모범 사례를 통해 학습 효과를 극대화합니다."
        ]
        
        documents = []
        
        for i in range(count):
            category = np.random.choice(categories)
            doc_type = np.random.choice(doc_types)
            difficulty = np.random.choice(difficulties)
            author = np.random.choice(authors)
            language = np.random.choice(languages, p=[0.7, 0.2, 0.05, 0.05])  # 한국어 비중 높음
            
            # 제목 생성
            title_template = np.random.choice(title_templates)
            title = title_template.format(category=category)
            
            # 내용 생성
            content_template = np.random.choice(content_templates)
            topic = f"{category.lower()} topic {i%20+1}"
            detail = f"detailed analysis {i%15+1}"
            content = content_template.format(category=category, topic=topic, detail=detail)
            
            # 추가 내용 확장
            content += f" 특히 {category} 분야에서는 {topic}와 관련된 다양한 접근 방법이 연구되고 있으며, {detail}는 핵심적인 요소로 인식되고 있습니다."
            content += f" 이러한 관점에서 볼 때, {difficulty} 수준의 이해가 필요하며, 실무 적용을 위해서는 단계적 접근이 권장됩니다."
            
            # 요약 생성
            summary = f"{category} 분야의 {topic}에 대한 {difficulty} 수준의 분석. {detail}를 중심으로 한 실용적 가이드."
            
            # 메타데이터 생성
            word_count = len(content.split())
            reading_time = max(1, word_count // 200)  # 분당 200단어 기준
            quality_score = np.random.uniform(1.0, 5.0)
            view_count = np.random.randint(100, 10000)
            like_count = int(view_count * np.random.uniform(0.01, 0.15))
            
            # 발행일 생성
            publish_date = f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            
            # 태그 생성
            tags = [category.lower(), doc_type, difficulty, f"topic_{i%10}"]
            if np.random.random() > 0.5:
                tags.append(f"author_{author.split('_')[1]}")
            
            document = {
                "title": title,
                "content": content,
                "summary": summary,
                "category": category,
                "author": author,
                "publish_date": publish_date,
                "tags": ", ".join(tags),
                "language": language,
                "doc_type": doc_type,
                "word_count": word_count,
                "reading_time": reading_time,
                "difficulty_level": difficulty,
                "quality_score": quality_score,
                "view_count": view_count,
                "like_count": like_count
            }
            
            documents.append(document)
        
        print(f"  ✅ {count}개 문서 생성 완료")
        return documents
    
    def generate_sample_qa_pairs(self, count: int = 500) -> List[Dict[str, Any]]:
        """샘플 Q&A 쌍 생성"""
        print(f"\n📊 샘플 Q&A {count}개 생성 중...")
        
        categories = ["Technology", "Science", "Business", "Health", "Education"]
        domains = ["Programming", "AI/ML", "Web Development", "Data Science", "Cybersecurity", 
                  "Biology", "Physics", "Chemistry", "Medicine", "Finance", "Marketing", 
                  "Management", "Nutrition", "Psychology", "Mathematics"]
        difficulties = ["beginner", "intermediate", "advanced"]
        
        # Q&A 템플릿
        qa_templates = [
            {
                "question": "{domain}에서 {topic}를 시작하려면 어떻게 해야 하나요?",
                "answer": "{domain} 분야에서 {topic}를 시작하기 위해서는 먼저 기본 개념을 이해하는 것이 중요합니다. {detail}부터 시작하여 단계적으로 학습을 진행하시기 바랍니다."
            },
            {
                "question": "{topic}의 주요 장점과 단점은 무엇인가요?",
                "answer": "{topic}의 주요 장점으로는 {advantage}이 있으며, 단점으로는 {disadvantage}이 있습니다. {domain} 분야에서는 이러한 특성을 고려하여 적절히 활용하는 것이 중요합니다."
            },
            {
                "question": "{domain} 전문가가 되기 위한 학습 로드맵을 알려주세요.",
                "answer": "{domain} 전문가가 되기 위해서는 {step1}, {step2}, {step3} 순서로 학습하시기 바랍니다. 각 단계마다 충분한 실습과 프로젝트 경험을 쌓는 것이 중요합니다."
            },
            {
                "question": "{topic} 구현 시 주의해야 할 점은 무엇인가요?",
                "answer": "{topic} 구현 시에는 {caution1}과 {caution2}를 특히 주의해야 합니다. {domain} 분야의 모범 사례를 참고하여 안정적이고 효율적인 구현을 위해 노력하시기 바랍니다."
            }
        ]
        
        qa_pairs = []
        
        for i in range(count):
            category = np.random.choice(categories)
            domain = np.random.choice(domains)
            difficulty = np.random.choice(difficulties)
            template = np.random.choice(qa_templates)
            
            # 변수 생성
            topic = f"{domain.lower()} concept {i%20+1}"
            detail = f"fundamental principle {i%10+1}"
            advantage = f"efficiency and scalability {i%5+1}"
            disadvantage = f"complexity and cost {i%3+1}"
            step1 = f"basic {domain.lower()} theory"
            step2 = f"intermediate {domain.lower()} practice"
            step3 = f"advanced {domain.lower()} application"
            caution1 = f"performance optimization {i%7+1}"
            caution2 = f"security consideration {i%6+1}"
            
            # 질문과 답변 생성
            question = template["question"].format(
                domain=domain, topic=topic, detail=detail,
                advantage=advantage, disadvantage=disadvantage,
                step1=step1, step2=step2, step3=step3,
                caution1=caution1, caution2=caution2
            )
            
            answer = template["answer"].format(
                domain=domain, topic=topic, detail=detail,
                advantage=advantage, disadvantage=disadvantage,
                step1=step1, step2=step2, step3=step3,
                caution1=caution1, caution2=caution2
            )
            
            # 메타데이터 생성
            confidence_score = np.random.uniform(0.6, 1.0)
            usage_count = np.random.randint(1, 100)
            is_verified = np.random.random() > 0.3  # 70% 검증됨
            
            # 날짜 생성
            created_date = f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            updated_date = created_date if np.random.random() > 0.3 else f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            
            qa_pair = {
                "question": question,
                "answer": answer,
                "category": category,
                "domain": domain,
                "difficulty": difficulty,
                "confidence_score": confidence_score,
                "created_date": created_date,
                "updated_date": updated_date,
                "is_verified": is_verified,
                "usage_count": usage_count
            }
            
            qa_pairs.append(qa_pair)
        
        print(f"  ✅ {count}개 Q&A 쌍 생성 완료")
        return qa_pairs
    
    def insert_documents(self, collection: Collection, documents: List[Dict[str, Any]]) -> None:
        """문서 데이터 삽입"""
        print(f"\n💾 문서 데이터 삽입 중...")
        
        # 텍스트 벡터화
        print("  텍스트 벡터화 중...")
        titles = [doc["title"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        summaries = [doc["summary"] for doc in documents]
        
        # 제목+내용 결합 텍스트 벡터화
        combined_texts = [f"{title} {content}" for title, content in zip(titles, contents)]
        text_vectors = self.vector_utils.texts_to_vectors(combined_texts)
        
        # 요약 벡터화
        summary_vectors = self.vector_utils.texts_to_vectors(summaries)
        
        # 데이터를 2단계 패턴으로 구성 (List[List])
        data = [
            [doc["title"] for doc in documents],
            [doc["content"] for doc in documents], 
            [doc["summary"] for doc in documents],
            [doc["category"] for doc in documents],
            [doc["author"] for doc in documents],
            [doc["publish_date"] for doc in documents],
            [doc["tags"] for doc in documents],
            [doc["language"] for doc in documents],
            [doc["doc_type"] for doc in documents],
            [doc["word_count"] for doc in documents],
            [doc["reading_time"] for doc in documents],
            [doc["difficulty_level"] for doc in documents],
            [doc["quality_score"] for doc in documents],
            [doc["view_count"] for doc in documents],
            [doc["like_count"] for doc in documents],
            text_vectors.tolist(),
            summary_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(documents)
        print(f"  ✅ {total_inserted}개 문서 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 문서 삽입 완료")
    
    def insert_qa_pairs(self, collection: Collection, qa_pairs: List[Dict[str, Any]]) -> None:
        """Q&A 데이터 삽입"""
        print(f"\n💾 Q&A 데이터 삽입 중...")
        
        # 벡터화
        print("  텍스트 벡터화 중...")
        questions = [qa["question"] for qa in qa_pairs]
        answers = [qa["answer"] for qa in qa_pairs]
        
        question_vectors = self.vector_utils.texts_to_vectors(questions)
        answer_vectors = self.vector_utils.texts_to_vectors(answers)
        
        # 데이터를 2단계 패턴으로 구성 (List[List])
        data = [
            [qa["question"] for qa in qa_pairs],
            [qa["answer"] for qa in qa_pairs],
            [qa["category"] for qa in qa_pairs],
            [qa["domain"] for qa in qa_pairs],
            [qa["difficulty"] for qa in qa_pairs],
            [qa["confidence_score"] for qa in qa_pairs],
            [qa["created_date"] for qa in qa_pairs],
            [qa["updated_date"] for qa in qa_pairs],  # 추가된 필드
            [qa["is_verified"] for qa in qa_pairs],
            [qa["usage_count"] for qa in qa_pairs],
            question_vectors.tolist(),
            answer_vectors.tolist()
        ]
        
        # 삽입 (2단계 패턴)
        result = collection.insert(data)
        total_inserted = len(qa_pairs)
        print(f"  ✅ {total_inserted}개 Q&A 삽입 완료")
        
        # 데이터 플러시
        print("  데이터 플러시 중...")
        collection.flush()
        print(f"  ✅ 총 {total_inserted}개 Q&A 삽입 완료")
    
    def create_indexes(self, collection: Collection, vector_fields: List[str]) -> None:
        """인덱스 생성"""
        print(f"\n🔍 인덱스 생성 중...")
        
        for field_name in vector_fields:
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",  # 코사인 유사도 사용
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            print(f"  {field_name} 인덱스 생성 중...")
            collection.create_index(field_name, index_params)
            print(f"  ✅ {field_name} 인덱스 생성 완료")
        
        print(f"  ✅ 모든 인덱스 생성 완료")
    
    def document_search_demo(self, collection: Collection) -> None:
        """문서 검색 데모"""
        print("\n" + "="*80)
        print(" 📖 문서 검색 시스템 데모")
        print("="*80)
        
        collection.load()
        
        # 다양한 검색 시나리오
        search_queries = [
            {
                "query": "인공지능과 머신러닝의 최신 동향",
                "description": "AI/ML 관련 기술 문서 검색"
            },
            {
                "query": "비즈니스 전략과 경영 혁신 방법",
                "description": "비즈니스 분야 전문 자료 검색"
            },
            {
                "query": "건강과 의학 연구의 새로운 발견",
                "description": "헬스케어 관련 연구 자료 검색"
            },
            {
                "query": "교육 기술과 학습 효과 향상",
                "description": "교육 분야 혁신 사례 검색"
            }
        ]
        
        for i, search_case in enumerate(search_queries, 1):
            print(f"\n{i}. {search_case['description']}")
            print(f"   검색어: '{search_case['query']}'")
            
            # 쿼리 벡터화
            query_vectors = self.vector_utils.text_to_vector(search_case['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 기본 검색 (텍스트 벡터 기반)
            start_time = time.time()
            results = collection.search(
                data=[query_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["title", "category", "author", "quality_score", "reading_time", "difficulty_level"]
            )
            search_time = time.time() - start_time
            
            print(f"   검색 시간: {search_time:.4f}초")
            print(f"   결과 수: {len(results[0])}")
            
            for j, hit in enumerate(results[0]):
                similarity = 1 - hit.distance  # 코사인 유사도
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')[:60]}...")
                print(f"        카테고리: {entity.get('category')}, 저자: {entity.get('author')}")
                print(f"        품질점수: {entity.get('quality_score'):.2f}, 읽기시간: {entity.get('reading_time')}분")
                print(f"        난이도: {entity.get('difficulty_level')}, 유사도: {similarity:.3f}")
            
            # 필터링 검색 (고품질 문서만)
            print(f"\n   📊 고품질 문서 필터링 검색 (품질점수 >= 4.0)")
            
            filtered_results = collection.search(
                data=[query_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=3,
                expr="quality_score >= 4.0",
                output_fields=["title", "category", "quality_score", "view_count", "like_count"]
            )
            
            print(f"   고품질 문서 결과 수: {len(filtered_results[0])}")
            for j, hit in enumerate(filtered_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')[:60]}...")
                print(f"        품질점수: {entity.get('quality_score'):.2f}, 조회수: {entity.get('view_count')}, 좋아요: {entity.get('like_count')}")
                print(f"        유사도: {similarity:.3f}")
    
    def qa_matching_demo(self, collection: Collection) -> None:
        """Q&A 매칭 데모"""
        print("\n" + "="*80)
        print(" 🤔 Q&A 매칭 시스템 데모")
        print("="*80)
        
        collection.load()
        
        # 사용자 질문 시나리오
        user_questions = [
            {
                "question": "프로그래밍을 처음 시작하는데 어떤 언어부터 배워야 할까요?",
                "description": "프로그래밍 입문자 질문"
            },
            {
                "question": "데이터 과학자가 되기 위한 학습 경로를 알려주세요",
                "description": "커리어 전환 관련 질문"
            },
            {
                "question": "인공지능 프로젝트에서 성능을 개선하는 방법은?",
                "description": "AI 프로젝트 최적화 질문"
            },
            {
                "question": "비즈니스 분석을 위한 필수 도구들이 궁금합니다",
                "description": "비즈니스 도구 추천 질문"
            }
        ]
        
        for i, user_q in enumerate(user_questions, 1):
            print(f"\n{i}. {user_q['description']}")
            print(f"   사용자 질문: '{user_q['question']}'")
            
            # 질문 벡터화
            query_vectors = self.vector_utils.text_to_vector(user_q['question'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            # 유사한 질문 검색
            start_time = time.time()
            question_results = collection.search(
                data=[query_vector],
                anns_field="question_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=3,
                output_fields=["question", "answer", "domain", "difficulty", "confidence_score", "is_verified"]
            )
            search_time = time.time() - start_time
            
            print(f"   검색 시간: {search_time:.4f}초")
            print(f"   매칭된 Q&A 수: {len(question_results[0])}")
            
            for j, hit in enumerate(question_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"\n     📌 매칭 Q&A {j+1} (유사도: {similarity:.3f})")
                print(f"     질문: {entity.get('question')[:80]}...")
                print(f"     답변: {entity.get('answer')[:100]}...")
                print(f"     도메인: {entity.get('domain')}, 난이도: {entity.get('difficulty')}")
                print(f"     신뢰도: {entity.get('confidence_score'):.2f}, 검증됨: {entity.get('is_verified')}")
            
            # 검증된 답변만 필터링
            print(f"\n   ✅ 검증된 답변만 검색")
            verified_results = collection.search(
                data=[query_vector],
                anns_field="question_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=2,
                expr="is_verified == True and confidence_score >= 0.8",
                output_fields=["question", "answer", "domain", "confidence_score"]
            )
            
            print(f"   검증된 답변 수: {len(verified_results[0])}")
            for j, hit in enumerate(verified_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('question')[:60]}...")
                print(f"        신뢰도: {entity.get('confidence_score'):.2f}, 유사도: {similarity:.3f}")
    
    def semantic_search_demo(self, collection: Collection) -> None:
        """시맨틱 검색 데모"""
        print("\n" + "="*80)
        print(" 🧠 시맨틱 검색 데모")
        print("="*80)
        
        collection.load()
        
        # 시맨틱 검색 시나리오 - 키워드가 다르지만 의미가 유사한 검색
        semantic_cases = [
            {
                "query": "기계가 사람처럼 생각하는 방법",
                "expected": "인공지능, 머신러닝 관련 문서",
                "description": "AI를 다른 표현으로 검색"
            },
            {
                "query": "회사 수익을 늘리는 전략",
                "expected": "비즈니스 성장, 마케팅 관련 문서",
                "description": "비즈니스 성장을 다른 표현으로 검색"
            },
            {
                "query": "몸의 면역력을 강화하는 방법",
                "expected": "건강, 의학 관련 문서",
                "description": "건강 관리를 다른 표현으로 검색"
            }
        ]
        
        for i, case in enumerate(semantic_cases, 1):
            print(f"\n{i}. {case['description']}")
            print(f"   검색어: '{case['query']}'")
            print(f"   기대 결과: {case['expected']}")
            
            # 텍스트 벡터로 검색
            query_vectors = self.vector_utils.text_to_vector(case['query'])
            query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors
            
            results = collection.search(
                data=[query_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=5,
                output_fields=["title", "category", "summary", "tags"]
            )
            
            print(f"   검색 결과:")
            for j, hit in enumerate(results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')[:70]}...")
                print(f"        카테고리: {entity.get('category')}, 유사도: {similarity:.3f}")
                print(f"        요약: {entity.get('summary')[:80]}...")
                print(f"        태그: {entity.get('tags')}")
            
            # 요약 벡터로도 검색해서 비교
            print(f"\n   📋 요약 기반 검색 결과:")
            summary_results = collection.search(
                data=[query_vector],
                anns_field="summary_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=3,
                output_fields=["title", "summary"]
            )
            
            for j, hit in enumerate(summary_results[0]):
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"     {j+1}. {entity.get('title')[:70]}...")
                print(f"        유사도: {similarity:.3f}")
                print(f"        요약: {entity.get('summary')}")
    
    def recommendation_demo(self, collection: Collection) -> None:
        """문서 추천 데모"""
        print("\n" + "="*80)
        print(" 🎯 문서 추천 시스템 데모")
        print("="*80)
        
        collection.load()
        
        # 사용자가 읽은 문서를 기반으로 유사한 문서 추천
        print("사용자가 최근에 읽은 문서를 기반으로 추천을 생성합니다...")
        
        # 임의의 문서를 "사용자가 읽은 문서"로 가정
        sample_results = collection.search(
            data=[[0.1] * 384],  # 임의의 벡터
            anns_field="text_vector",
            param={"metric_type": "COSINE", "params": {"ef": 200}},
            limit=1,
            output_fields=["title", "category", "text_vector", "tags", "author"]
        )
        
        if sample_results and len(sample_results[0]) > 0:
            read_doc = sample_results[0][0].entity
            read_vector = read_doc.get('text_vector')
            
            print(f"\n📖 사용자가 읽은 문서:")
            print(f"   제목: {read_doc.get('title')}")
            print(f"   카테고리: {read_doc.get('category')}")
            print(f"   저자: {read_doc.get('author')}")
            print(f"   태그: {read_doc.get('tags')}")
            
            # 유사한 문서 추천
            print(f"\n🎯 이 문서를 바탕으로 한 추천:")
            
            recommendations = collection.search(
                data=[read_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=6,  # 첫 번째는 자기 자신이므로 6개 검색
                output_fields=["title", "category", "author", "quality_score", "reading_time", "view_count"]
            )
            
            # 자기 자신 제외
            for j, hit in enumerate(recommendations[0][1:], 1):  # 첫 번째 결과 제외
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"   {j}. {entity.get('title')[:70]}...")
                print(f"      카테고리: {entity.get('category')}, 저자: {entity.get('author')}")
                print(f"      품질점수: {entity.get('quality_score'):.2f}, 읽기시간: {entity.get('reading_time')}분")
                print(f"      유사도: {similarity:.3f}, 조회수: {entity.get('view_count')}")
            
            # 카테고리별 추천
            category = read_doc.get('category')
            print(f"\n📚 같은 카테고리({category}) 내 추천:")
            
            category_recommendations = collection.search(
                data=[read_vector],
                anns_field="text_vector",
                param={"metric_type": "COSINE", "params": {"ef": 200}},
                limit=4,
                expr=f"category == '{category}'",
                output_fields=["title", "author", "quality_score"]
            )
            
            for j, hit in enumerate(category_recommendations[0][1:], 1):  # 자기 자신 제외
                similarity = 1 - hit.distance
                entity = hit.entity
                print(f"   {j}. {entity.get('title')[:70]}...")
                print(f"      저자: {entity.get('author')}, 품질점수: {entity.get('quality_score'):.2f}")
                print(f"      유사도: {similarity:.3f}")


def main():
    """메인 실행 함수"""
    print("🚀 텍스트 유사도 검색 시스템 실습")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Milvus 연결
        with MilvusConnection() as conn:
            print("✅ Milvus 연결 성공")
            
            # 검색 엔진 초기화
            search_engine = TextSimilaritySearchEngine(conn)
            
            # 1. 문서 검색 시스템
            print("\n" + "="*80)
            print(" 📖 문서 검색 시스템 구축")
            print("="*80)
            
            # 문서 컬렉션 생성
            doc_collection = search_engine.create_document_collection()
            
            # 샘플 문서 생성 및 삽입
            documents = search_engine.generate_sample_documents(1000)
            search_engine.insert_documents(doc_collection, documents)
            
            # 인덱스 생성
            search_engine.create_indexes(doc_collection, ["text_vector", "summary_vector"])
            
            # 문서 검색 데모
            search_engine.document_search_demo(doc_collection)
            
            # 시맨틱 검색 데모
            search_engine.semantic_search_demo(doc_collection)
            
            # 추천 시스템 데모
            search_engine.recommendation_demo(doc_collection)
            
            # 2. Q&A 매칭 시스템
            print("\n" + "="*80)
            print(" 🤔 Q&A 매칭 시스템 구축")
            print("="*80)
            
            # Q&A 컬렉션 생성
            qa_collection = search_engine.create_qa_collection()
            
            # 샘플 Q&A 생성 및 삽입
            qa_pairs = search_engine.generate_sample_qa_pairs(500)
            search_engine.insert_qa_pairs(qa_collection, qa_pairs)
            
            # 인덱스 생성
            search_engine.create_indexes(qa_collection, ["question_vector", "answer_vector"])
            
            # Q&A 매칭 데모
            search_engine.qa_matching_demo(qa_collection)
            
            # 컬렉션 정리
            print("\n🧹 테스트 컬렉션 정리 중...")
            if utility.has_collection("document_search"):
                utility.drop_collection("document_search")
            if utility.has_collection("qa_pairs"):
                utility.drop_collection("qa_pairs")
            print("✅ 정리 완료")
            
        print("\n🎉 텍스트 유사도 검색 시스템 실습 완료!")
        
        print("\n💡 학습 포인트:")
        print("  • 다양한 메타데이터를 활용한 고급 문서 검색")
        print("  • 시맨틱 검색으로 키워드 한계 극복")
        print("  • Q&A 매칭으로 지능형 고객지원 구현")
        print("  • 벡터 유사도 기반 개인화 추천")
        print("  • 필터링과 벡터 검색의 효과적 조합")
        
        print("\n🚀 다음 단계:")
        print("  python step03_use_cases/02_image_similarity_search.py")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 