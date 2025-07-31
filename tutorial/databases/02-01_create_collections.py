# DB  컬렉션을 묶는 상위 단위
# Collection 테이블 같은 단위
# Schema collection 을 정의 하는 설계도
# Field 스키마의 구성 요소

# 컬렉션은 Fields 과 Entities 가 있는 2차원 테이블
# 스키마, 인텍스. 매개변수, 메트릭 유형 및 생성 시 로드할지 여부를 정의하여 컬렉션을 만들 수 있음
# 컬렉션 만들기 위해서는 세가지 단계를 거친다
# 스키마 만들기 -> 인덱스 매개변수 설정(선택사항) -> 컬렉션 만들기


# 스키마 만들기
from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 3.1. Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 3.2. Add fields to schema
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)

# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(
    field_name="my_id",
    index_type="AUTOINDEX"
)

index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
# 3.5. Create a collection with the index loaded simultaneously
client.create_collection(
    collection_name="customized_setup_1",
    schema=schema,
    index_params=index_params
)

res = client.get_load_state(
    collection_name="customized_setup_1"
)

print(res)
# {'state': <LoadState: Loaded>}


# 3.6. Create a collection and index it separately
client.create_collection(
    collection_name="customized_setup_2",
    schema=schema,
)

res = client.get_load_state(
    collection_name="customized_setup_2"
)

print(res)
# {'state': <LoadState: NotLoad>}

# index_params를 포함해 컬렉션을 만들면 자동으로 인덱스 생성과 메모리 로딩까지 한 번에 수행됩니다.
# 인덱스를 따로 만들거나 로드하려면 추가 작업이 필요합니다.