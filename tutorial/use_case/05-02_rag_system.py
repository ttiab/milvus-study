"""
RAG (Retrieval-Augmented Generation) 시스템

1. 문서 임베딩 시스템
1-1. 문서 컬렉션 생성
1-2. 문서 분할 및 전처리
1-3. 문서 임베딩 및 저장
1-4. 인덱스 생성

2. 검색 및 생성 시스템
2-1. 유사도 검색
2-2. 컨텍스트 기반 응답 생성
2-3. RAG 파이프라인 구현
2-4. 실시간 Q&A 데모

기존 VectorUtils와 Milvus 연동을 활용한 RAG 시스템 구현
"""

import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from tutorial.common.vector_utils import VectorUtils
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema


# 간단한 응답 생성을 위한 템플릿 (실제 환경에서는 OpenAI API나 다른 LLM 사용)
class SimpleResponseGenerator:
    """간단한 응답 생성기 (실제로는 OpenAI API나 다른 LLM 사용)"""

    def __init__(self):
        self.templates = {
            "default": "검색된 정보를 바탕으로 답변드리겠습니다:\n\n{context}\n\n위 정보를 종합하면, {query}에 대한 답변은 다음과 같습니다:",
            "technical": "기술적 질문에 대한 답변:\n\n관련 문서:\n{context}\n\n{query}에 대한 기술적 설명:",
            "general": "일반적인 질문에 대한 답변:\n\n참고 자료:\n{context}\n\n{query}에 대해 설명드리면:"
        }

    def generate_response(self, query: str, context_docs: List[Dict], response_type: str = "default") -> str:
        """검색된 문서를 기반으로 응답 생성"""
        # 컨텍스트 문서 정리
        context_texts = []
        for i, doc in enumerate(context_docs, 1):
            context_texts.append(f"{i}. {doc.get('content', '')[:200]}...")

        context = "\n".join(context_texts)

        # 템플릿 기반 응답 생성
        template = self.templates.get(response_type, self.templates["default"])
        response = template.format(context=context, query=query)

        # 간단한 후처리
        if "기술" in query or "구현" in query or "방법" in query:
            response += "\n\n구체적인 구현 방법이나 추가 정보가 필요하시면 말씀해 주세요."

        return response


@dataclass
class Document:
    """문서 데이터 클래스"""
    doc_id: str
    title: str
    content: str
    category: str = ""
    metadata: Dict = None
    created_at: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """문서 처리 및 분할 클래스"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?;:]', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 현재 청크에 문장을 추가했을 때 크기 확인
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 새 청크 시작
                current_chunk = sentence

        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_document(self, document: Document) -> List[Dict]:
        """문서를 처리하여 청크로 분할"""
        # 텍스트 전처리
        clean_content = self.clean_text(document.content)

        # 청크로 분할
        chunks = self.split_into_chunks(clean_content)

        # 청크 데이터 생성
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "doc_id": document.doc_id,
                "chunk_id": f"{document.doc_id}_chunk_{i}",
                "title": document.title,
                "content": chunk,
                "category": document.category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": document.metadata,
                "created_at": document.created_at
            }
            chunk_data.append(chunk_info)

        return chunk_data


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) 시스템"""

    def __init__(self, milvus_uri: str = "http://localhost:19530", milvus_token: str = "root:Milvus"):
        self.client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self.vector_utils = VectorUtils()
        self.doc_processor = DocumentProcessor()
        self.response_generator = SimpleResponseGenerator()

        # 컬렉션 이름
        self.collection_name = "rag_documents"

        # 벡터 차원 (sentence-transformers 기본값)
        self.vector_dim = 384

        print("🚀 RAG 시스템 초기화 완료")

    def create_document_collection(self):
        """RAG 문서 컬렉션 생성"""
        print("\n📁 RAG 문서 컬렉션 생성 중...")

        # 기존 컬렉션 삭제
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"  기존 컬렉션 '{self.collection_name}' 삭제")

        # 스키마 정의
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG 시스템용 문서 컬렉션",
            enable_dynamic_field=True
        )

        self.client.create_collection(collection_name=self.collection_name, schema=schema)
        print(f"  ✅ 컬렉션 '{self.collection_name}' 생성 완료")

        return True

    def create_vector_index(self):
        """벡터 인덱스 생성"""
        print("\n🔍 벡터 인덱스 생성 중...")

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            index_type="HNSW",
            field_name="content_vector",
            metric_type="COSINE",
            params={
                "M": 16,
                "efConstruction": 200
            }
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        print("  ✅ 벡터 인덱스 생성 완료")
        return True

    def add_documents(self, documents: List[Document]) -> bool:
        """문서들을 처리하여 벡터 데이터베이스에 저장"""
        print(f"\n💾 {len(documents)}개 문서 처리 및 저장 중...")

        all_chunks = []

        # 각 문서를 청크로 분할
        for doc in documents:
            chunks = self.doc_processor.process_document(doc)
            all_chunks.extend(chunks)

        print(f"  총 {len(all_chunks)}개 청크 생성됨")

        # 콘텐츠 벡터화
        print("  콘텐츠 벡터화 중...")
        contents = [chunk['content'] for chunk in all_chunks]
        content_vectors = self.vector_utils.texts_to_vectors(contents)

        # 데이터 준비
        data = []
        for i, chunk in enumerate(all_chunks):
            data_item = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "category": chunk["category"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "created_at": chunk["created_at"],
                "content_vector": content_vectors[i].tolist()
            }
            data.append(data_item)

        # 데이터 삽입
        result = self.client.upsert(collection_name=self.collection_name, data=data)

        # 데이터 플러시
        self.client.flush(collection_name=self.collection_name)

        print(f"  ✅ {len(all_chunks)}개 청크 저장 완료")
        return True

    def search_similar_documents(self, query: str, top_k: int = 5, category_filter: str = None) -> List[Dict]:
        """유사 문서 검색"""
        # 쿼리 벡터화
        query_vectors = self.vector_utils.text_to_vector(query)
        query_vector = query_vectors[0] if len(query_vectors.shape) > 1 else query_vectors

        # 검색 파라미터
        search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

        # 카테고리 필터링
        filter_expr = None
        if category_filter:
            filter_expr = f"category == '{category_filter}'"

        # 컬렉션 로드
        self.client.load_collection(collection_name=self.collection_name)

        # 벡터 검색 - 올바른 파라미터 사용
        try:
            # Alternative 1: Using filter parameter
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector.tolist()],
                "limit": top_k,
                "search_params": search_params,
                "output_fields": ["doc_id", "title", "content", "category", "chunk_index"]
            }

            # Add filter only if it exists
            if filter_expr:
                search_kwargs["filter"] = filter_expr

            results = self.client.search(**search_kwargs)

        except Exception as e:
            print(f"Search error: {e}")
            # Alternative 2: Try with different parameter structure
            try:
                print("Trying alternative search method...")
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector.tolist()],
                    anns_field="content_vector",
                    param=search_params,
                    limit=top_k,
                    expression=filter_expr,  # Some versions use 'expression'
                    output_fields=["doc_id", "title", "content", "category", "chunk_index"]
                )
            except Exception as e2:
                print(f"Alternative search also failed: {e2}")
                return []

        # 결과 정리
        similar_docs = []
        for hit in results[0]:
            similarity = 1 - hit['distance']
            entity = hit['entity']
            doc_info = {
                "similarity": similarity,
                "doc_id": entity.get('doc_id'),
                "title": entity.get('title'),
                "content": entity.get('content'),
                "category": entity.get('category'),
                "chunk_index": entity.get('chunk_index')
            }
            similar_docs.append(doc_info)

        return similar_docs

    def generate_answer(self, query: str, top_k: int = 3, category_filter: str = None) -> Dict[str, Any]:
        """RAG 파이프라인: 검색 + 생성"""
        start_time = time.time()

        # 1. 유사 문서 검색
        similar_docs = self.search_similar_documents(query, top_k, category_filter)
        search_time = time.time() - start_time

        # 2. 응답 생성
        response_start = time.time()

        # 응답 타입 결정
        response_type = "technical" if any(word in query for word in ["구현", "방법", "기술", "코드"]) else "general"

        # 응답 생성
        answer = self.response_generator.generate_response(query, similar_docs, response_type)
        generation_time = time.time() - response_start

        total_time = time.time() - start_time

        return {
            "query": query,
            "answer": answer,
            "similar_documents": similar_docs,
            "performance": {
                "search_time": search_time,
                "generation_time": generation_time,
                "total_time": total_time
            },
            "metadata": {
                "top_k": top_k,
                "category_filter": category_filter,
                "response_type": response_type
            }
        }


def generate_sample_documents() -> List[Document]:
    """샘플 문서 생성"""
    print("\n📚 샘플 문서 생성 중...")

    documents = [
        Document(
            doc_id="tech_001",
            title="벡터 데이터베이스 소개",
            content="""
            벡터 데이터베이스는 고차원 벡터 데이터를 효율적으로 저장하고 검색할 수 있는 특수한 데이터베이스입니다. 
            전통적인 관계형 데이터베이스와 달리 벡터 데이터베이스는 임베딩 벡터를 기본 데이터 타입으로 사용합니다.

            주요 특징:
            1. 고차원 벡터 저장 및 인덱싱
            2. 유사도 기반 검색 (코사인 유사도, 유클리드 거리 등)
            3. 확장 가능한 아키텍처
            4. 실시간 검색 성능

            벡터 데이터베이스는 AI 애플리케이션에서 핵심적인 역할을 합니다. 특히 RAG (Retrieval-Augmented Generation) 
            시스템에서 문서 검색을 위해 널리 사용됩니다. Milvus, Pinecone, Weaviate 등이 대표적인 벡터 데이터베이스입니다.

            벡터 데이터베이스를 사용하면 의미적 유사성을 기반으로 한 검색이 가능합니다. 예를 들어, "강아지"와 "반려동물" 
            같은 의미적으로 유사한 단어들을 찾을 수 있습니다.
            """,
            category="기술",
            created_at="2024-12-24"
        ),

        Document(
            doc_id="ai_001",
            title="RAG 시스템 구현 가이드",
            content="""
            RAG (Retrieval-Augmented Generation)는 정보 검색과 텍스트 생성을 결합한 AI 기술입니다.
            RAG 시스템은 크게 두 부분으로 구성됩니다: 검색기(Retriever)와 생성기(Generator).

            RAG 시스템 구현 단계:
            1. 문서 수집 및 전처리
            2. 문서 분할 (chunking)
            3. 임베딩 생성 및 벡터 데이터베이스 저장
            4. 인덱스 생성
            5. 쿼리 처리 및 유사 문서 검색
            6. 검색된 문서를 활용한 응답 생성

            RAG의 장점:
            - 최신 정보 활용 가능
            - 환각(hallucination) 문제 완화
            - 투명한 정보 출처 제공
            - 도메인 특화 지식 활용

            구현 시 고려사항:
            - 적절한 청크 크기 설정
            - 임베딩 모델 선택
            - 검색 정확도 최적화
            - 응답 품질 평가

            RAG 시스템은 챗봇, 질의응답 시스템, 문서 요약 등 다양한 용도로 활용됩니다.
            """,
            category="AI",
            created_at="2024-12-24"
        ),

        Document(
            doc_id="ml_001",
            title="임베딩과 벡터 표현",
            content="""
            임베딩(Embedding)은 텍스트, 이미지, 음성 등의 데이터를 고차원 벡터로 변환하는 기술입니다.
            임베딩을 통해 컴퓨터가 인간의 언어를 수치적으로 이해할 수 있게 됩니다.

            임베딩의 종류:
            1. 단어 임베딩: Word2Vec, GloVe, FastText
            2. 문장 임베딩: BERT, Sentence-BERT, Universal Sentence Encoder
            3. 이미지 임베딩: CNN 기반 특징 추출, CLIP
            4. 다중 모달 임베딩: CLIP, ALIGN

            임베딩의 특성:
            - 의미적 유사성 보존
            - 차원 축소
            - 연산 가능성
            - 전이 학습 가능

            임베딩 활용 분야:
            - 정보 검색
            - 추천 시스템
            - 분류 및 클러스터링
            - 이상 탐지

            좋은 임베딩의 조건:
            1. 의미적으로 유사한 데이터는 가까운 위치에
            2. 의미적으로 다른 데이터는 먼 위치에
            3. 안정적이고 일관된 표현
            4. 하위 작업에 적합한 표현력

            Sentence-BERT는 문장 수준의 임베딩 생성에 특화된 모델로, RAG 시스템에서 널리 사용됩니다.
            """,
            category="머신러닝",
            created_at="2024-12-24"
        ),

        Document(
            doc_id="dev_001",
            title="Python 벡터 처리 라이브러리",
            content="""
            Python에서 벡터 처리를 위한 주요 라이브러리들을 소개합니다.

            NumPy:
            - 기본적인 배열 연산
            - 벡터화된 연산
            - 브로드캐스팅 기능
            - 메모리 효율적인 처리

            SciPy:
            - 고급 수학 함수
            - 희소 행렬 처리
            - 최적화 알고리즘
            - 통계 함수

            Scikit-learn:
            - 머신러닝 알고리즘
            - 전처리 도구
            - 차원 축소 (PCA, t-SNE)
            - 모델 평가 메트릭

            PyTorch:
            - 딥러닝 프레임워크
            - 자동 미분
            - GPU 가속
            - 동적 계산 그래프

            Transformers (Hugging Face):
            - 사전 훈련된 모델
            - 토크나이저
            - 파이프라인
            - 모델 허브

            Sentence-Transformers:
            - 문장 임베딩 특화
            - 다양한 사전 훈련 모델
            - 간단한 API
            - 의미적 검색 지원

            벡터 처리 팁:
            1. 배치 처리로 성능 향상
            2. 적절한 데이터 타입 선택
            3. 메모리 사용량 모니터링
            4. GPU 활용 고려
            """,
            category="개발",
            created_at="2024-12-24"
        ),

        Document(
            doc_id="tutorial_001",
            title="Milvus 벡터 데이터베이스 활용법",
            content="""
            Milvus는 오픈소스 벡터 데이터베이스로, 대규모 벡터 검색에 최적화되어 있습니다.

            Milvus 주요 기능:
            1. 다양한 벡터 인덱스 지원 (HNSW, IVF, Annoy 등)
            2. 실시간 삽입 및 검색
            3. 수평적 확장 가능
            4. 하이브리드 검색 (벡터 + 스칼라)
            5. 다중 테넌시 지원

            Milvus 사용 단계:
            1. 컬렉션 스키마 정의
            2. 컬렉션 생성
            3. 데이터 삽입
            4. 인덱스 생성
            5. 컬렉션 로드
            6. 검색 수행

            검색 파라미터:
            - metric_type: COSINE, L2, IP 등
            - params: 인덱스별 특화 파라미터
            - expr: 필터링 조건
            - output_fields: 반환할 필드

            성능 최적화:
            1. 적절한 인덱스 선택
            2. 벡터 차원 최적화
            3. 배치 삽입 활용
            4. 메모리 관리
            5. 파티셔닝 활용

            Milvus는 PyMilvus SDK를 통해 Python에서 쉽게 사용할 수 있습니다.
            MilvusClient 클래스는 간단한 API를 제공하여 빠른 개발이 가능합니다.
            """,
            category="데이터베이스",
            created_at="2024-12-24"
        )
    ]

    print(f"  ✅ {len(documents)}개 샘플 문서 생성 완료")
    return documents


# ================================
# RAG 시스템 데모 실행
# ================================

if __name__ == "__main__":
    print("=" * 80)
    print(" 🤖 RAG (Retrieval-Augmented Generation) 시스템 데모")
    print("=" * 80)

    # RAG 시스템 초기화
    rag_system = RAGSystem()

    # 1. 문서 컬렉션 생성
    rag_system.create_document_collection()

    # 2. 샘플 문서 생성 및 저장
    sample_docs = generate_sample_documents()
    rag_system.add_documents(sample_docs)

    # 3. 인덱스 생성
    rag_system.create_vector_index()

    print("\n" + "=" * 80)
    print(" 🔍 RAG 시스템 질의응답 데모")
    print("=" * 80)

    # 4. 질의응답 데모
    demo_queries = [
        {
            "query": "벡터 데이터베이스란 무엇인가요?",
            "description": "기본적인 개념 질문"
        },
        {
            "query": "RAG 시스템을 구현하려면 어떤 단계를 거쳐야 하나요?",
            "description": "구현 방법 질문"
        },
        {
            "query": "임베딩의 종류에는 어떤 것들이 있나요?",
            "description": "분류 정보 질문"
        },
        {
            "query": "Milvus에서 성능을 최적화하는 방법",
            "description": "기술적 최적화 질문"
        },
        {
            "query": "Python에서 벡터 처리에 사용할 수 있는 라이브러리",
            "description": "도구 추천 질문"
        }
    ]

    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"질문 {i}: {demo['description']}")
        print(f"{'=' * 60}")
        print(f"🤔 사용자: {demo['query']}")

        # RAG 시스템으로 답변 생성
        result = rag_system.generate_answer(demo['query'], top_k=3)

        print(f"\n🤖 AI 어시스턴트:")
        print(result['answer'])

        print(f"\n📊 성능 정보:")
        perf = result['performance']
        print(f"  검색 시간: {perf['search_time']:.3f}초")
        print(f"  생성 시간: {perf['generation_time']:.3f}초")
        print(f"  총 시간: {perf['total_time']:.3f}초")

        print(f"\n📚 참조 문서 ({len(result['similar_documents'])}개):")
        for j, doc in enumerate(result['similar_documents'], 1):
            print(f"  {j}. {doc['title']} (유사도: {doc['similarity']:.3f})")
            print(f"     카테고리: {doc['category']}")
            print(f"     내용 미리보기: {doc['content'][:100]}...")

        # 잠시 대기 (데모 효과)
        time.sleep(1)

    print("\n" + "=" * 80)
    print(" 🎯 카테고리별 검색 데모")
    print("=" * 80)

    # 5. 카테고리별 검색 데모
    category_queries = [
        {
            "query": "기술적 구현 방법",
            "category": "기술",
            "description": "기술 카테고리 내 검색"
        },
        {
            "query": "머신러닝 알고리즘",
            "category": "머신러닝",
            "description": "머신러닝 카테고리 내 검색"
        },
        {
            "query": "개발 도구",
            "category": "개발",
            "description": "개발 카테고리 내 검색"
        }
    ]

    for demo in category_queries:
        print(f"\n📂 {demo['description']}")
        print(f"질문: {demo['query']}")
        print(f"카테고리 필터: {demo['category']}")

        # 카테고리 필터링 검색
        result = rag_system.generate_answer(
            demo['query'],
            top_k=2,
            category_filter=demo['category']
        )

        print(f"\n답변:")
        print(result['answer'])

        print(f"\n참조 문서:")
        for doc in result['similar_documents']:
            print(f"  - {doc['title']} (유사도: {doc['similarity']:.3f})")

    print("\n" + "=" * 80)
    print(" ✅ RAG 시스템 데모 완료")
    print("=" * 80)
    print("\n💡 실제 프로덕션 환경에서는:")
    print("  1. OpenAI API나 Hugging Face 모델을 사용한 고품질 응답 생성")
    print("  2. 더 정교한 문서 전처리 및 청킹 전략")
    print("  3. 벡터 검색 결과의 re-ranking")
    print("  4. 사용자 피드백 기반 시스템 개선")
    print("  5. 실시간 문서 업데이트 및 인덱싱")