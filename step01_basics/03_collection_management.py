#!/usr/bin/env python3
"""
1.3 컬렉션 관리

Milvus 컬렉션의 생성, 관리, 삭제 방법을 학습합니다.
- 컬렉션 스키마 설계
- 다양한 데이터 타입 활용
- 컬렉션 생성/삭제
- 컬렉션 정보 조회
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility
from common.connection import MilvusConnection


def print_section(title):
    """섹션 제목 출력"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def learn_data_types():
    """Milvus 데이터 타입 학습"""
    print_section("3.1 Milvus 데이터 타입")
    
    print("Milvus에서 지원하는 주요 데이터 타입:")
    print()
    
    # 스칼라 데이터 타입
    print("📊 스칼라 데이터 타입:")
    scalar_types = [
        ("BOOL", "불린", "True/False 값"),
        ("INT8", "8비트 정수", "-128 ~ 127"),
        ("INT16", "16비트 정수", "-32,768 ~ 32,767"),
        ("INT32", "32비트 정수", "-2^31 ~ 2^31-1"),
        ("INT64", "64비트 정수", "-2^63 ~ 2^63-1"),
        ("FLOAT", "32비트 실수", "IEEE 754 단정밀도"),
        ("DOUBLE", "64비트 실수", "IEEE 754 배정밀도"),
        ("VARCHAR", "가변 문자열", "최대 길이 지정 필요")
    ]
    
    for dtype, name, description in scalar_types:
        print(f"   • {dtype:10} : {name:12} - {description}")
    
    print("\n🔢 벡터 데이터 타입:")
    vector_types = [
        ("FLOAT_VECTOR", "실수 벡터", "일반적인 임베딩 벡터"),
        ("BINARY_VECTOR", "이진 벡터", "압축된 벡터 표현"),
        ("FLOAT16_VECTOR", "16비트 실수 벡터", "메모리 절약형"),
        ("BFLOAT16_VECTOR", "BFloat16 벡터", "AI 가속기 최적화")
    ]
    
    for dtype, name, description in vector_types:
        print(f"   • {dtype:17} : {name:12} - {description}")
    
    print("\n💡 중요 사항:")
    print("   • 각 컬렉션은 하나의 Primary Key 필드 필요")
    print("   • VARCHAR 필드는 max_length 파라미터 필수")
    print("   • 벡터 필드는 dim (차원) 파라미터 필수")
    print("   • auto_id=True 시 Primary Key 자동 생성")
    
    return True


def basic_schema_creation():
    """기본 스키마 생성"""
    print_section("3.2 기본 스키마 생성")
    
    print("간단한 텍스트 검색용 컬렉션 스키마를 만들어보겠습니다:")
    print()
    
    try:
        # 1. 필드 정의
        print("1. 필드 정의:")
        
        # Primary Key 필드
        id_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,  # 자동 ID 생성
            description="Primary key"
        )
        print(f"   ✅ ID 필드: {id_field.name} ({id_field.dtype})")
        
        # 텍스트 필드
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=1000,  # 최대 1000자
            description="Original text content"
        )
        print(f"   ✅ 텍스트 필드: {text_field.name} ({text_field.dtype}, max_length={text_field.params['max_length']})")
        
        # 벡터 필드
        vector_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,  # 384차원 벡터 (sentence-transformers 기본 크기)
            description="Text embedding vector"
        )
        print(f"   ✅ 벡터 필드: {vector_field.name} ({vector_field.dtype}, dim={vector_field.params['dim']})")
        
        # 2. 스키마 생성
        print("\n2. 스키마 생성:")
        schema = CollectionSchema(
            fields=[id_field, text_field, vector_field],
            description="Basic text search collection",
            enable_dynamic_field=False  # 동적 필드 비활성화
        )
        
        print(f"   ✅ 스키마 생성 완료!")
        print(f"   📝 설명: {schema.description}")
        print(f"   🔧 동적 필드: {schema.enable_dynamic_field}")
        print(f"   📊 필드 수: {len(schema.fields)}")
        
        # 3. 스키마 정보 출력
        print("\n3. 스키마 상세 정보:")
        for i, field in enumerate(schema.fields):
            print(f"   필드 {i+1}: {field.name}")
            print(f"     타입: {field.dtype}")
            print(f"     Primary: {field.is_primary}")
            print(f"     설명: {field.description}")
            if hasattr(field, 'params') and field.params:
                for key, value in field.params.items():
                    print(f"     {key}: {value}")
            print()
        
        return schema
        
    except Exception as e:
        print(f"❌ 스키마 생성 실패: {e}")
        return None


def advanced_schema_creation():
    """고급 스키마 생성"""
    print_section("3.3 고급 스키마 생성")
    
    print("다양한 필드 타입을 포함한 고급 스키마를 만들어보겠습니다:")
    print()
    
    try:
        # 1. 다양한 필드 정의
        print("1. 다양한 필드 정의:")
        
        fields = []
        
        # Primary Key
        fields.append(FieldSchema(
            name="doc_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,  # 수동 ID 관리
            description="Document ID"
        ))
        print("   ✅ doc_id: Primary Key (수동 관리)")
        
        # 문서 제목
        fields.append(FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="Document title"
        ))
        print("   ✅ title: 문서 제목 (최대 200자)")
        
        # 문서 내용
        fields.append(FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=5000,
            description="Document content"
        ))
        print("   ✅ content: 문서 내용 (최대 5000자)")
        
        # 카테고리
        fields.append(FieldSchema(
            name="category",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="Document category"
        ))
        print("   ✅ category: 카테고리 (최대 50자)")
        
        # 점수
        fields.append(FieldSchema(
            name="score",
            dtype=DataType.FLOAT,
            description="Document relevance score"
        ))
        print("   ✅ score: 점수 (실수형)")
        
        # 생성 시간
        fields.append(FieldSchema(
            name="created_time",
            dtype=DataType.INT64,
            description="Creation timestamp"
        ))
        print("   ✅ created_time: 생성 시간 (타임스탬프)")
        
        # 활성 상태
        fields.append(FieldSchema(
            name="is_active",
            dtype=DataType.BOOL,
            description="Document active status"
        ))
        print("   ✅ is_active: 활성 상태 (불린)")
        
        # 임베딩 벡터
        fields.append(FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=512,  # 더 큰 차원
            description="Document embedding vector"
        ))
        print("   ✅ vector: 임베딩 벡터 (512차원)")
        
        # 2. 고급 스키마 생성
        print("\n2. 고급 스키마 생성:")
        advanced_schema = CollectionSchema(
            fields=fields,
            description="Advanced document collection with multiple field types",
            enable_dynamic_field=True,  # 동적 필드 활성화
            primary_field="doc_id"
        )
        
        print("   ✅ 고급 스키마 생성 완료!")
        print(f"   📝 설명: {advanced_schema.description}")
        print(f"   🔧 동적 필드: {advanced_schema.enable_dynamic_field}")
        print(f"   🔑 Primary Key: {advanced_schema.primary_field}")
        print(f"   📊 필드 수: {len(advanced_schema.fields)}")
        
        # 3. 동적 필드 설명
        print("\n3. 동적 필드 기능:")
        print("   💡 enable_dynamic_field=True로 설정하면:")
        print("      • 스키마에 정의되지 않은 필드도 삽입 가능")
        print("      • 런타임에 필드 추가 가능")
        print("      • 유연한 데이터 구조 지원")
        print("      • 단, 벡터 필드는 반드시 스키마에 정의 필요")
        
        return advanced_schema
        
    except Exception as e:
        print(f"❌ 고급 스키마 생성 실패: {e}")
        return None


def collection_operations():
    """컬렉션 기본 작업"""
    print_section("3.4 컬렉션 기본 작업")
    
    print("컬렉션 생성, 조회, 삭제 작업을 실습해보겠습니다:")
    print()
    
    try:
        with MilvusConnection() as conn:
            # 1. 테스트용 스키마 준비
            print("1. 테스트용 스키마 준비:")
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Test collection for basic operations"
            )
            print("   ✅ 테스트 스키마 준비 완료")
            
            # 2. 컬렉션 생성
            collection_name = "test_basic_operations"
            print(f"\n2. 컬렉션 생성: {collection_name}")
            
            # 기존 컬렉션 삭제 (있다면)
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"   🗑️  기존 컬렉션 삭제됨")
            
            # 새 컬렉션 생성
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default'
            )
            print(f"   ✅ 컬렉션 생성 완료: {collection_name}")
            
            # 3. 컬렉션 정보 조회
            print(f"\n3. 컬렉션 정보 조회:")
            print(f"   이름: {collection.name}")
            print(f"   설명: {collection.description}")
            print(f"   엔티티 수: {collection.num_entities}")
            print(f"   스키마 필드 수: {len(collection.schema.fields)}")
            
            # 필드 상세 정보
            print(f"\n   📊 필드 상세 정보:")
            for field in collection.schema.fields:
                field_info = f"      • {field.name}: {field.dtype}"
                if field.is_primary:
                    field_info += " (Primary Key)"
                if hasattr(field, 'params') and field.params:
                    params = ", ".join([f"{k}={v}" for k, v in field.params.items()])
                    field_info += f" [{params}]"
                print(field_info)
            
            # 4. 컬렉션 목록 확인
            print(f"\n4. 현재 컬렉션 목록:")
            collections = utility.list_collections()
            for i, coll_name in enumerate(collections, 1):
                status = "✅ 방금 생성" if coll_name == collection_name else "📁 기존"
                print(f"   {i}. {coll_name} {status}")
            
            # 5. 컬렉션 존재 확인
            print(f"\n5. 컬렉션 존재 확인:")
            exists = utility.has_collection(collection_name)
            print(f"   {collection_name} 존재: {'✅ True' if exists else '❌ False'}")
            
            # 6. 컬렉션 삭제
            print(f"\n6. 컬렉션 삭제:")
            utility.drop_collection(collection_name)
            print(f"   🗑️  {collection_name} 삭제 완료")
            
            # 삭제 확인
            exists_after = utility.has_collection(collection_name)
            print(f"   삭제 후 존재 확인: {'❌ 여전히 존재' if exists_after else '✅ 삭제됨'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 컬렉션 작업 중 오류: {e}")
        return False


def collection_with_custom_class():
    """커스텀 클래스를 사용한 컬렉션 관리"""
    print_section("3.5 커스텀 클래스를 사용한 컬렉션 관리")
    
    print("우리가 만든 MilvusConnection 클래스로 컬렉션을 관리해보겠습니다:")
    print()
    
    try:
        with MilvusConnection() as conn:
            # 1. 스키마 정의
            print("1. 스키마 정의:")
            fields = [
                FieldSchema(
                    name="article_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Article unique ID"
                ),
                FieldSchema(
                    name="title",
                    dtype=DataType.VARCHAR,
                    max_length=300,
                    description="Article title"
                ),
                FieldSchema(
                    name="summary",
                    dtype=DataType.VARCHAR,
                    max_length=1000,
                    description="Article summary"
                ),
                FieldSchema(
                    name="publish_date",
                    dtype=DataType.INT64,
                    description="Publication timestamp"
                ),
                FieldSchema(
                    name="view_count",
                    dtype=DataType.INT32,
                    description="View count"
                ),
                FieldSchema(
                    name="rating",
                    dtype=DataType.FLOAT,
                    description="User rating (0.0-5.0)"
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=256,
                    description="Article content embedding"
                )
            ]
            
            print(f"   ✅ {len(fields)}개 필드 정의 완료")
            
            # 2. 커스텀 클래스로 컬렉션 생성
            print("\n2. 커스텀 클래스로 컬렉션 생성:")
            collection_name = "news_articles"
            
            # 기존 컬렉션 정리
            if conn.get_collection(collection_name):
                conn.drop_collection(collection_name)
                print(f"   🗑️  기존 컬렉션 삭제")
            
            # 새 컬렉션 생성
            collection = conn.create_collection(
                collection_name=collection_name,
                fields=fields,
                description="News articles collection with rich metadata",
                auto_id=False
            )
            
            if collection:
                print(f"   ✅ 컬렉션 생성 성공: {collection_name}")
            else:
                print(f"   ❌ 컬렉션 생성 실패")
                return False
            
            # 3. 컬렉션 정보 조회
            print("\n3. 컬렉션 정보 조회:")
            info = conn.get_collection_info(collection_name)
            
            if info:
                print(f"   이름: {info['name']}")
                print(f"   설명: {info['description']}")
                print(f"   엔티티 수: {info['num_entities']}")
                
                print(f"\n   📊 스키마 정보:")
                schema = info['schema']
                print(f"   Primary Field: {schema.primary_field}")
                print(f"   Auto ID: {schema.auto_id}")
                print(f"   Dynamic Field: {schema.enable_dynamic_field}")
                
                print(f"\n   📝 필드 목록:")
                for field in schema.fields:
                    field_str = f"      • {field.name}: {field.dtype}"
                    if field.is_primary:
                        field_str += " 🔑"
                    if hasattr(field, 'params') and field.params:
                        params = []
                        for k, v in field.params.items():
                            params.append(f"{k}={v}")
                        field_str += f" ({', '.join(params)})"
                    print(field_str)
            
            # 4. 컬렉션 목록 확인
            print(f"\n4. 현재 컬렉션 목록:")
            collections = conn.list_collections()
            for i, coll in enumerate(collections, 1):
                marker = "🆕" if coll == collection_name else "📁"
                print(f"   {i}. {marker} {coll}")
            
            # 5. 정리
            print(f"\n5. 테스트 컬렉션 정리:")
            if conn.drop_collection(collection_name):
                print(f"   ✅ {collection_name} 삭제 완료")
            else:
                print(f"   ❌ {collection_name} 삭제 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 커스텀 클래스 컬렉션 관리 중 오류: {e}")
        return False


def collection_best_practices():
    """컬렉션 관리 베스트 프랙티스"""
    print_section("3.6 컬렉션 관리 베스트 프랙티스")
    
    print("효율적인 컬렉션 설계 및 관리 방법:")
    print()
    
    # 1. 네이밍 규칙
    print("1. ✅ 네이밍 규칙:")
    print("   • 소문자와 언더스코어 사용: user_documents")
    print("   • 의미있는 이름 사용: article_embeddings")
    print("   • 너무 긴 이름 피하기: < 64자")
    print("   • 특수문자 사용 금지")
    
    # 2. 스키마 설계 원칙
    print("\n2. ✅ 스키마 설계 원칙:")
    print("   • Primary Key는 INT64 타입 권장")
    print("   • VARCHAR 필드는 적절한 max_length 설정")
    print("   • 벡터 차원은 모델과 일치하게 설정")
    print("   • 필요한 필드만 정의 (성능 최적화)")
    print("   • 동적 필드는 필요시에만 활성화")
    
    # 3. 데이터 타입 선택
    print("\n3. ✅ 데이터 타입 선택 가이드:")
    print("   • ID: INT64 (범위가 큰 정수)")
    print("   • 텍스트: VARCHAR (max_length 신중히 설정)")
    print("   • 시간: INT64 (Unix timestamp)")
    print("   • 점수: FLOAT 또는 DOUBLE")
    print("   • 플래그: BOOL")
    print("   • 벡터: FLOAT_VECTOR (일반적)")
    
    # 4. 벡터 차원 고려사항
    print("\n4. ✅ 벡터 차원 고려사항:")
    dimension_guide = [
        ("384", "all-MiniLM-L6-v2", "가벼운 텍스트 임베딩"),
        ("512", "all-mpnet-base-v2", "고품질 텍스트 임베딩"),
        ("768", "BERT-base", "BERT 계열 모델"),
        ("1024", "BERT-large", "큰 언어 모델"),
        ("1536", "text-embedding-ada-002", "OpenAI 임베딩")
    ]
    
    print("   모델별 권장 차원:")
    for dim, model, desc in dimension_guide:
        print(f"   • {dim:4}차원: {model:25} - {desc}")
    
    # 5. 성능 최적화 팁
    print("\n5. ✅ 성능 최적화 팁:")
    print("   • 자주 검색하는 필드는 스칼라 필드로 분리")
    print("   • VARCHAR 필드 크기를 실제 필요한 만큼만 설정")
    print("   • 불필요한 필드는 제거")
    print("   • 벡터 차원은 정확히 맞춰 설정")
    print("   • Primary Key는 auto_id=True 권장 (성능상 유리)")
    
    # 6. 컬렉션 라이프사이클 관리
    print("\n6. ✅ 컬렉션 라이프사이클 관리:")
    print("   📅 개발 단계:")
    print("      • 스키마 실험 및 검증")
    print("      • 테스트 데이터로 성능 확인")
    print("   🚀 배포 단계:")
    print("      • 스키마 고정 및 문서화")
    print("      • 백업 및 복구 계획 수립")
    print("   🔧 운영 단계:")
    print("      • 정기적인 성능 모니터링")
    print("      • 필요시 스키마 마이그레이션")
    
    # 7. 주의사항
    print("\n7. ⚠️  주의사항:")
    print("   • 컬렉션 생성 후 스키마 변경 불가")
    print("   • 컬렉션 삭제 시 모든 데이터 손실")
    print("   • 벡터 차원 변경 시 재생성 필요")
    print("   • 대용량 컬렉션 삭제 시 시간 소요")
    
    return True


def main():
    """메인 함수"""
    print("📁 Milvus 학습 프로젝트 - 1.3 컬렉션 관리")
    print(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 실습 단계들
    sections = [
        ("데이터 타입 학습", learn_data_types),
        ("기본 스키마 생성", basic_schema_creation),
        ("고급 스키마 생성", advanced_schema_creation),
        ("컬렉션 기본 작업", collection_operations),
        ("커스텀 클래스 활용", collection_with_custom_class),
        ("베스트 프랙티스", collection_best_practices)
    ]
    
    results = []
    
    for section_name, section_func in sections:
        try:
            print(f"\n🚀 {section_name} 시작...")
            result = section_func()
            results.append((section_name, result))
            
            if result:
                print(f"✅ {section_name} 완료!")
            else:
                print(f"❌ {section_name} 실패!")
                
        except Exception as e:
            print(f"❌ {section_name} 중 오류: {e}")
            results.append((section_name, False))
    
    # 결과 요약
    print_section("실습 결과 요약")
    
    passed = 0
    for section_name, result in results:
        status = "✅ 완료" if result else "❌ 실패"
        print(f"{section_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 섹션: {len(results)}개")
    print(f"완료: {passed}개")
    print(f"실패: {len(results) - passed}개")
    
    if passed == len(results):
        print("\n🎉 모든 컬렉션 관리 실습이 완료되었습니다!")
        print("\n다음 실습으로 진행하세요:")
        print("   python step01_basics/04_data_insertion.py")
    else:
        print(f"\n⚠️  {len(results) - passed}개 섹션에서 문제가 발생했습니다.")
        print("문제를 해결한 후 다시 실행해주세요.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 