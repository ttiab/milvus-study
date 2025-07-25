#!/usr/bin/env python3
"""
Milvus 연결 테스트 스크립트

Milvus 서버 연결 상태를 확인하고 기본 동작을 테스트합니다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.connection import MilvusConnection
from common.vector_utils import VectorUtils
from common.data_loader import DataLoader


def test_connection():
    """Milvus 연결 테스트"""
    print("=" * 50)
    print("Milvus 연결 테스트 시작")
    print("=" * 50)
    
    # 연결 객체 생성
    conn = MilvusConnection()
    
    # 연결 시도
    print("\n1. Milvus 서버 연결 중...")
    if conn.connect():
        print("✅ 연결 성공!")
    else:
        print("❌ 연결 실패!")
        return False
    
    # 연결 상태 확인
    print("\n2. 연결 상태 확인 중...")
    if conn.check_connection():
        print("✅ 연결 상태 정상!")
    else:
        print("❌ 연결 상태 이상!")
        return False
    
    # 컬렉션 목록 조회
    print("\n3. 기존 컬렉션 목록 조회...")
    collections = conn.list_collections()
    print(f"기존 컬렉션: {collections}")
    
    # 연결 해제
    print("\n4. 연결 해제...")
    conn.disconnect()
    print("✅ 연결 해제 완료!")
    
    return True


def test_vector_utils():
    """벡터 유틸리티 테스트"""
    print("\n" + "=" * 50)
    print("벡터 유틸리티 테스트 시작")
    print("=" * 50)
    
    vector_utils = VectorUtils()
    
    # 모델 정보 확인
    print("\n1. 모델 정보 확인...")
    model_info = vector_utils.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 텍스트 벡터 변환 테스트
    print("\n2. 텍스트 벡터 변환 테스트...")
    try:
        if vector_utils.load_text_model():
            test_texts = ["안녕하세요", "벡터 변환 테스트"]
            vectors = vector_utils.text_to_vector(test_texts)
            print(f"✅ 텍스트 벡터 변환 성공! 형태: {vectors.shape}")
        else:
            print("❌ 텍스트 모델 로드 실패")
    except Exception as e:
        print(f"❌ 텍스트 벡터 변환 실패: {e}")
    
    return True


def test_data_loader():
    """데이터 로더 테스트"""
    print("\n" + "=" * 50)
    print("데이터 로더 테스트 시작")
    print("=" * 50)
    
    data_loader = DataLoader()
    
    # 데이터셋 정보 확인
    print("\n1. 데이터셋 정보 확인...")
    dataset_info = data_loader.get_dataset_info()
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # 샘플 데이터 생성
    print("\n2. 샘플 텍스트 데이터 생성...")
    try:
        sample_data = data_loader.create_sample_text_dataset(
            size=10,
            save_path="datasets/text/sample_texts.json"
        )
        print(f"✅ 샘플 데이터 생성 성공! {len(sample_data)}개 항목")
        print("  첫 번째 항목:", sample_data[0])
    except Exception as e:
        print(f"❌ 샘플 데이터 생성 실패: {e}")
    
    return True


def test_basic_operations():
    """기본 Milvus 작업 테스트"""
    print("\n" + "=" * 50)
    print("기본 Milvus 작업 테스트 시작")
    print("=" * 50)
    
    try:
        from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
        
        with MilvusConnection() as conn:
            # 테스트 컬렉션 스키마 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            
            collection_name = "test_collection"
            
            # 기존 컬렉션 삭제 (있다면)
            if conn.get_collection(collection_name):
                conn.drop_collection(collection_name)
                print(f"기존 '{collection_name}' 컬렉션 삭제")
            
            # 컬렉션 생성
            print(f"\n1. '{collection_name}' 컬렉션 생성...")
            collection = conn.create_collection(
                collection_name=collection_name,
                fields=fields,
                description="테스트용 컬렉션"
            )
            
            if collection:
                print("✅ 컬렉션 생성 성공!")
                
                # 컬렉션 정보 조회
                print("\n2. 컬렉션 정보 조회...")
                info = conn.get_collection_info(collection_name)
                if info:
                    print(f"  이름: {info['name']}")
                    print(f"  설명: {info['description']}")
                    print(f"  엔티티 수: {info['num_entities']}")
                
                # 컬렉션 삭제
                print(f"\n3. '{collection_name}' 컬렉션 삭제...")
                if conn.drop_collection(collection_name):
                    print("✅ 컬렉션 삭제 성공!")
                
            else:
                print("❌ 컬렉션 생성 실패!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 기본 작업 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("Milvus 학습 프로젝트 - 환경 테스트")
    print(f"테스트 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Milvus 연결", test_connection),
        ("벡터 유틸리티", test_vector_utils),
        ("데이터 로더", test_data_loader),
        ("기본 Milvus 작업", test_basic_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 테스트: {len(results)}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(results) - passed}개")
    
    if passed == len(results):
        print("\n🎉 모든 테스트가 통과했습니다!")
        print("Milvus 학습 환경이 정상적으로 구성되었습니다.")
    else:
        print(f"\n⚠️  {len(results) - passed}개 테스트가 실패했습니다.")
        print("환경 설정을 다시 확인해주세요.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 