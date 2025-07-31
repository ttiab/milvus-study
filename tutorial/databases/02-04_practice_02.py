from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

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

# 컬렉션 생성
collection_name = "advanced_schema_collection"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(collection_name=collection_name, schema=advanced_schema)

res = client.describe_collection(
    collection_name=collection_name
)
print(res)