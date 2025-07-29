"""컬렉션 관리 베스트 프랙티스"""
print("컬렉션 관리 베스트 프랙티스")
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