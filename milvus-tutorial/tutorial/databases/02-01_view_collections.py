from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 컬렉션 리스트
res = client.list_collections()
print(res)

# 컬렉션 설명
res = client.describe_collection(
    collection_name="my_new_collection"
)
print(res)

# 컬렉션명 변셩
client.rename_collection(
    old_name="my_new_collection",
    new_name="my_collection"
)
res = client.list_collections()
print(res)

# 컬렉션 속성 설정
client.alter_collection_properties(
    collection_name="my_collection",
    properties={"collection.ttl.seconds": 60}
)

# 컬렉션 속성 삭제
client.drop_collection_properties(
    collection_name="my_collection",
    property_keys=[
        "collection.ttl.seconds"
    ]
)