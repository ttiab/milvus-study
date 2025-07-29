"""
기본 스키마 생성
"""
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 필드 정의
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="Primary key")
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000, description='Original text content')
vector_field = FieldSchema(name='embedding',dtype=DataType.FLOAT_VECTOR,dim=384,description='Text embedding vector')

# 스키마 생성
schema = CollectionSchema(fields=[id_field, text_field, vector_field], description='Basic text search collection', enable_dynamic=False)

# 컬렉션 생성
collection_name = "basic_text_collection"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(collection_name=collection_name, schema=schema)

res = client.describe_collection(
    collection_name=collection_name
)
print(res)