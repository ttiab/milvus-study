#!/usr/bin/env python3
"""
1.2 기본 연결

Milvus 서버와의 기본 연결 방법을 학습합니다.
- 다양한 연결 방법
- 연결 상태 확인
- 서버 정보 조회
- 연결 관리 베스트 프랙티스
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import connections, utility
from common.connection import MilvusConnection


def print_section(title):
    """섹션 제목 출력"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def basic_connection_method():
    """기본 연결 방법"""
    print_section("2.1 기본 연결 방법")
    
    print("pymilvus 라이브러리를 사용한 직접 연결:")
    print()
    
    try:
        # 방법 1: 기본 연결
        print("1. 기본 연결 (localhost:19530)")
        connections.connect(
            alias="default",
            host='localhost',
            port='19530'
        )
        print("✅ 기본 연결 성공!")
        
        # 서버 버전 확인
        version = utility.get_server_version()
        print(f"📊 Milvus 서버 버전: {version}")
        
        # 연결 해제
        connections.disconnect("default")
        print("🔌 연결 해제 완료")
        
    except Exception as e:
        print(f"❌ 기본 연결 실패: {e}")
        return False
    
    print("\n" + "-"*50)
    
    try:
        # 방법 2: 명시적 파라미터 연결
        print("\n2. 명시적 파라미터 연결")
        connections.connect(
            alias="explicit",
            host='localhost',
            port='19530',
            user='',  # 사용자명 (없으면 빈 문자열)
            password='',  # 비밀번호 (없으면 빈 문자열)
            secure=False  # SSL 사용 여부
        )
        print("✅ 명시적 연결 성공!")
        
        # 연결 목록 확인
        print("🔗 현재 연결 목록:")
        for alias in connections.list_connections():
            print(f"   - {alias}")
        
        # 연결 해제
        connections.disconnect("explicit")
        print("🔌 연결 해제 완료")
        
    except Exception as e:
        print(f"❌ 명시적 연결 실패: {e}")
        return False
    
    return True


def connection_with_class():
    """클래스를 사용한 연결 방법"""
    print_section("2.2 클래스를 사용한 연결")
    
    print("우리가 만든 MilvusConnection 클래스 사용:")
    print()
    
    try:
        # 일반적인 사용 방법
        print("1. 일반적인 사용 방법:")
        conn = MilvusConnection()
        
        if conn.connect():
            print("✅ 클래스 연결 성공!")
            
            # 연결 상태 확인
            if conn.check_connection():
                print("✅ 연결 상태 정상!")
            
            # 컬렉션 목록 조회
            collections = conn.list_collections()
            print(f"📁 컬렉션 목록: {collections}")
            
            # 연결 해제
            conn.disconnect()
            
        else:
            print("❌ 클래스 연결 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 클래스 연결 중 오류: {e}")
        return False
    
    print("\n" + "-"*50)
    
    try:
        # 컨텍스트 매니저 사용 방법
        print("\n2. 컨텍스트 매니저 사용 (권장):")
        with MilvusConnection() as conn:
            print("✅ 컨텍스트 매니저로 연결 성공!")
            
            # 서버 정보 조회
            if conn.check_connection():
                print("✅ 서버 상태 정상!")
                
                # 기본 정보 출력
                print("📊 서버 정보:")
                try:
                    version = utility.get_server_version()
                    print(f"   버전: {version}")
                except:
                    print("   버전: 조회 실패")
                
                # 컬렉션 정보
                collections = conn.list_collections()
                print(f"   컬렉션 수: {len(collections)}")
                
        print("🔌 컨텍스트 매니저 자동 연결 해제")
        
    except Exception as e:
        print(f"❌ 컨텍스트 매니저 사용 중 오류: {e}")
        return False
    
    return True


def connection_status_check():
    """연결 상태 확인 방법"""
    print_section("2.3 연결 상태 확인")
    
    print("다양한 방법으로 연결 상태를 확인해보겠습니다:")
    print()
    
    try:
        with MilvusConnection() as conn:
            # 방법 1: 기본 연결 확인
            print("1. 기본 연결 확인:")
            if conn.connected:
                print("✅ 연결 플래그: True")
            else:
                print("❌ 연결 플래그: False")
            
            # 방법 2: 서버 응답 확인
            print("\n2. 서버 응답 확인:")
            if conn.check_connection():
                print("✅ 서버 응답: 정상")
            else:
                print("❌ 서버 응답: 이상")
            
            # 방법 3: 연결 목록 확인
            print("\n3. 활성 연결 목록:")
            active_connections = connections.list_connections()
            for alias in active_connections:
                print(f"   ✅ {alias}")
            
            # 방법 4: 서버 기본 정보 조회
            print("\n4. 서버 상세 정보:")
            try:
                version = utility.get_server_version()
                print(f"   📊 버전: {version}")
                
                # 컬렉션 수 확인
                collections = conn.list_collections()
                print(f"   📁 컬렉션 수: {len(collections)}")
                
                # 현재 시간 기록
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"   ⏰ 확인 시간: {current_time}")
                
            except Exception as e:
                print(f"   ❌ 상세 정보 조회 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 연결 상태 확인 중 오류: {e}")
        return False


def connection_error_handling():
    """연결 오류 처리"""
    print_section("2.4 연결 오류 처리")
    
    print("일반적인 연결 오류와 처리 방법을 살펴보겠습니다:")
    print()
    
    # 잘못된 호스트로 연결 시도
    print("1. 잘못된 호스트 연결 시도:")
    try:
        connections.connect(
            alias="wrong_host",
            host='wrong_host',
            port='19530'
        )
        print("❌ 이 메시지가 보이면 안 됩니다!")
        
    except Exception as e:
        print(f"✅ 예상된 오류 캐치: {type(e).__name__}")
        print(f"   오류 메시지: {str(e)[:100]}...")
    
    print("\n" + "-"*30)
    
    # 잘못된 포트로 연결 시도
    print("\n2. 잘못된 포트 연결 시도:")
    try:
        connections.connect(
            alias="wrong_port",
            host='localhost',
            port='99999'  # 존재하지 않는 포트
        )
        print("❌ 이 메시지가 보이면 안 됩니다!")
        
    except Exception as e:
        print(f"✅ 예상된 오류 캐치: {type(e).__name__}")
        print(f"   오류 메시지: {str(e)[:100]}...")
    
    print("\n" + "-"*30)
    
    # 안전한 연결 방법 예제
    print("\n3. 안전한 연결 방법 예제:")
    
    def safe_connect(host='localhost', port='19530', timeout=5):
        """안전한 연결 함수"""
        try:
            # 타임아웃을 고려한 연결
            connections.connect(
                alias="safe_connection",
                host=host,
                port=port
            )
            
            # 연결 테스트
            version = utility.get_server_version()
            print(f"✅ 안전한 연결 성공! 버전: {version}")
            
            # 연결 해제
            connections.disconnect("safe_connection")
            return True
            
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            return False
    
    # 안전한 연결 테스트
    result = safe_connect()
    
    print("\n💡 연결 실패 시 확인사항:")
    print("   1. Milvus 서버가 실행 중인가? (docker-compose ps)")
    print("   2. 포트가 열려있는가? (netstat -an | grep 19530)")
    print("   3. 방화벽이 차단하고 있는가?")
    print("   4. 호스트명/IP가 올바른가?")
    
    return True


def connection_best_practices():
    """연결 관리 베스트 프랙티스"""
    print_section("2.5 연결 관리 베스트 프랙티스")
    
    print("효율적인 연결 관리 방법을 알아보겠습니다:")
    print()
    
    # 1. 컨텍스트 매니저 사용
    print("1. ✅ 권장: 컨텍스트 매니저 사용")
    print("   - 자동으로 연결 해제")
    print("   - 예외 발생 시에도 안전")
    print("   - 리소스 누수 방지")
    print()
    print("   ```python")
    print("   with MilvusConnection() as conn:")
    print("       # 작업 수행")
    print("       pass")
    print("   # 자동으로 연결 해제됨")
    print("   ```")
    
    print("\n" + "-"*30)
    
    # 2. 연결 재사용
    print("\n2. ✅ 권장: 연결 재사용")
    try:
        # 연결 생성
        conn = MilvusConnection()
        conn.connect()
        
        print("   연결 생성 후 여러 작업 수행:")
        
        # 여러 작업 수행
        for i in range(3):
            collections = conn.list_collections()
            print(f"   작업 {i+1}: 컬렉션 {len(collections)}개 확인")
        
        # 연결 해제
        conn.disconnect()
        print("   ✅ 하나의 연결로 여러 작업 완료")
        
    except Exception as e:
        print(f"   ❌ 연결 재사용 중 오류: {e}")
    
    print("\n" + "-"*30)
    
    # 3. 연결 풀링 (개념 설명)
    print("\n3. 💡 고급: 연결 풀링 개념")
    print("   - 대용량 애플리케이션에서 유용")
    print("   - 여러 연결을 미리 생성하여 재사용")
    print("   - 연결 생성/해제 오버헤드 감소")
    print("   - 현재 PyMilvus는 기본적으로 연결 풀링 지원")
    
    print("\n" + "-"*30)
    
    # 4. 오류 처리 패턴
    print("\n4. ✅ 권장: 견고한 오류 처리")
    print()
    
    def robust_connection_example():
        """견고한 연결 예제"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with MilvusConnection() as conn:
                    # 연결 테스트
                    conn.check_connection()
                    print(f"   ✅ 연결 성공 (시도 {attempt + 1})")
                    return True
                    
            except Exception as e:
                print(f"   ❌ 연결 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    print("   ❌ 모든 재시도 실패")
                    return False
    
    print("   견고한 연결 테스트:")
    robust_connection_example()
    
    print("\n💡 연결 관리 요약:")
    print("   ✅ 컨텍스트 매니저 사용")
    print("   ✅ 적절한 오류 처리")
    print("   ✅ 연결 재사용")
    print("   ✅ 자원 정리")
    print("   ❌ 연결 누수 방지")
    
    return True


def main():
    """메인 함수"""
    print("🔗 Milvus 학습 프로젝트 - 1.2 기본 연결")
    print(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 실습 단계들
    sections = [
        ("기본 연결 방법", basic_connection_method),
        ("클래스 연결 방법", connection_with_class),
        ("연결 상태 확인", connection_status_check),
        ("연결 오류 처리", connection_error_handling),
        ("연결 베스트 프랙티스", connection_best_practices)
    ]
    
    results = []
    
    for section_name, section_func in sections:
        try:
            print(f"\n🚀 {section_name} 실습 시작...")
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
        print("\n🎉 모든 연결 실습이 완료되었습니다!")
        print("\n다음 실습으로 진행하세요:")
        print("   python step01_basics/03_collection_management.py")
    else:
        print(f"\n⚠️  {len(results) - passed}개 섹션에서 문제가 발생했습니다.")
        print("문제를 해결한 후 다시 실행해주세요.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 