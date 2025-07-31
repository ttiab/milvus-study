from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# DB 삭제
client.drop_database(
    db_name="my_database_1"
)

# DB 목록 조회
print(f"DB 목록 {client.list_databases()}")

# DB 생성
createdDB = client.create_database(db_name="my_database_1")
print(f"DB 생성: {createdDB}")

# DB 보기
viewDB = client.describe_database(db_name="default")
print(viewDB)


"""
DB 속성
database.replica.number int 지정된 데이터베이스의 복제본 수입니다.

database.resource_groups string 지정된 데이터베이스와 연결된 리소스 그룹의 이름을 쉼표로 구분한 목록입니다.

database.diskQuota.mb int 지정한 데이터베이스의 디스크 공간 최대 크기(MB)입니다.

database.max.collections int 지정한 데이터베이스에 허용되는 최대 컬렉션 수입니다.

database.force.deny.writing boolean 지정한 데이터베이스에서 쓰기 작업을 거부하도록 강제할지 여부입니다.

database.force.deny.reading boolean 지정한 데이터베이스에서 읽기 작업을 거부하도록 할지 여부입니다.
"""

# DB 속성 변경
client.alter_database_properties(
    db_name="my_database_1",
    properties={
        "database.max.collections": 10
    }
)

# DB 속성 삭제
client.drop_database_properties(
    db_name="my_database_1",
    property_keys=[
        "database.max.collections"
    ]
)
