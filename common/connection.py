"""
Milvus 연결 관리 모듈

Milvus 데이터베이스 연결, 컬렉션 관리, 기본 CRUD 작업을 위한 유틸리티 클래스
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from pymilvus import (
    connections, 
    Collection, 
    FieldSchema, 
    CollectionSchema, 
    DataType,
    utility
)

# 환경 변수 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusConnection:
    """Milvus 데이터베이스 연결 및 관리 클래스"""
    
    def __init__(self, alias: str = "default"):
        """
        초기화
        
        Args:
            alias: 연결 별칭
        """
        self.alias = alias
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.user = os.getenv("MILVUS_USER", "")
        self.password = os.getenv("MILVUS_PASSWORD", "")
        self.connected = False
        
    def connect(self) -> bool:
        """
        Milvus 서버에 연결
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            self.connected = True
            logger.info(f"Milvus 서버에 연결되었습니다: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Milvus 연결 실패: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """연결 해제"""
        try:
            connections.disconnect(self.alias)
            self.connected = False
            logger.info("Milvus 연결이 해제되었습니다.")
        except Exception as e:
            logger.error(f"연결 해제 실패: {e}")
    
    def check_connection(self) -> bool:
        """연결 상태 확인"""
        try:
            # utility.get_server_version()으로 연결 테스트
            version = utility.get_server_version()
            logger.info(f"Milvus 서버 버전: {version}")
            return True
        except Exception as e:
            logger.error(f"연결 확인 실패: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 조회"""
        try:
            collections = utility.list_collections()
            logger.info(f"컬렉션 목록: {collections}")
            return collections
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []
    
    def create_collection(
        self,
        collection_name: str,
        fields: List[FieldSchema],
        description: str = "",
        auto_id: bool = True
    ) -> Optional[Collection]:
        """
        컬렉션 생성
        
        Args:
            collection_name: 컬렉션 이름
            fields: 필드 스키마 목록
            description: 컬렉션 설명
            auto_id: 자동 ID 생성 여부
            
        Returns:
            Collection: 생성된 컬렉션 객체
        """
        try:
            # 스키마 생성
            schema = CollectionSchema(
                fields=fields,
                description=description,
                auto_id=auto_id
            )
            
            # 컬렉션 생성
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.alias
            )
            
            logger.info(f"컬렉션 '{collection_name}' 생성 완료")
            return collection
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            return None
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        기존 컬렉션 조회
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            Collection: 컬렉션 객체
        """
        try:
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name, using=self.alias)
                return collection
            else:
                logger.warning(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")
                return None
                
        except Exception as e:
            logger.error(f"컬렉션 조회 실패: {e}")
            return None
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        컬렉션 삭제
        
        Args:
            collection_name: 삭제할 컬렉션 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"컬렉션 '{collection_name}' 삭제 완료")
                return True
            else:
                logger.warning(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")
                return False
                
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        컬렉션 정보 조회
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            Dict: 컬렉션 정보
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                return None
            
            info = {
                "name": collection.name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "schema": collection.schema,
                "indexes": collection.indexes
            }
            
            return info
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return None
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.disconnect()


def get_default_connection() -> MilvusConnection:
    """기본 Milvus 연결 객체 반환"""
    return MilvusConnection() 