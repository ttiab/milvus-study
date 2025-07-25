#!/usr/bin/env python3
"""
1.1 환경 구축

Milvus 개발 환경 구축 및 설정 확인
- Docker 환경 확인
- Milvus 서버 상태 점검
- Python 라이브러리 테스트
- 환경 변수 설정 확인
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.connection import MilvusConnection


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_docker():
    """Docker 환경 확인"""
    print_header("Docker 환경 확인")
    
    try:
        # Docker 버전 확인
        print("1. Docker 버전 확인...")
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print("❌ Docker가 설치되지 않았거나 실행되지 않습니다.")
            return False
        
        # Docker Compose 버전 확인
        print("\n2. Docker Compose 버전 확인...")
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print("❌ Docker Compose가 설치되지 않았습니다.")
            return False
        
        # Docker 서비스 상태 확인
        print("\n3. Docker 서비스 상태 확인...")
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker 서비스가 정상 실행 중입니다.")
        else:
            print("❌ Docker 서비스가 실행되지 않습니다.")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ Docker가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print(f"❌ Docker 확인 중 오류 발생: {e}")
        return False


def check_milvus_containers():
    """Milvus 컨테이너 상태 확인"""
    print_header("Milvus 컨테이너 상태 확인")
    
    try:
        # 컨테이너 목록 확인
        print("1. Milvus 관련 컨테이너 확인...")
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=milvus', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print("현재 실행 중인 Milvus 컨테이너:")
            print(result.stdout)
            
            # 각 서비스별 상태 확인
            services = ['milvus-standalone', 'milvus-etcd', 'milvus-minio']
            running_services = []
            
            for service in services:
                check_result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={service}', '--format', '{{.Names}}'],
                    capture_output=True, text=True
                )
                if service in check_result.stdout:
                    running_services.append(service)
                    print(f"✅ {service}: 실행 중")
                else:
                    print(f"❌ {service}: 중지됨")
            
            if len(running_services) == len(services):
                print("\n✅ 모든 Milvus 서비스가 정상 실행 중입니다!")
                return True
            else:
                print(f"\n⚠️  {len(services) - len(running_services)}개 서비스가 실행되지 않고 있습니다.")
                return False
                
        else:
            print("❌ 실행 중인 Milvus 컨테이너가 없습니다.")
            print("\n다음 명령으로 Milvus를 시작하세요:")
            print("  cd 프로젝트_루트")
            print("  docker-compose up -d")
            return False
            
    except Exception as e:
        print(f"❌ 컨테이너 확인 중 오류 발생: {e}")
        return False


def check_python_environment():
    """Python 환경 확인"""
    print_header("Python 환경 확인")
    
    # Python 버전 확인
    print(f"1. Python 버전: {sys.version}")
    
    # 필수 라이브러리 확인
    required_libraries = [
        'pymilvus',
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'sentence_transformers'
    ]
    
    print("\n2. 필수 라이브러리 확인...")
    missing_libraries = []
    
    for lib in required_libraries:
        try:
            __import__(lib)
            print(f"✅ {lib}: 설치됨")
        except ImportError:
            print(f"❌ {lib}: 설치 필요")
            missing_libraries.append(lib)
    
    if missing_libraries:
        print(f"\n⚠️  다음 라이브러리를 설치해야 합니다: {', '.join(missing_libraries)}")
        print("다음 명령으로 설치하세요:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 모든 필수 라이브러리가 설치되어 있습니다!")
        return True


def check_environment_variables():
    """환경 변수 확인"""
    print_header("환경 변수 확인")
    
    # 중요 환경 변수 목록
    env_vars = {
        'MILVUS_HOST': 'localhost',
        'MILVUS_PORT': '19530',
        'DOCKER_VOLUME_DIRECTORY': './volumes'
    }
    
    print("환경 변수 설정 상태:")
    all_set = True
    
    for var, default in env_vars.items():
        value = os.getenv(var, default)
        print(f"  {var}: {value}")
        if value == default:
            print(f"    (기본값 사용)")
    
    # .env 파일 확인
    env_file = project_root / '.env'
    env_example_file = project_root / 'env.example'
    
    print(f"\n.env 파일 확인:")
    if env_file.exists():
        print(f"✅ .env 파일이 존재합니다: {env_file}")
    else:
        print(f"⚠️  .env 파일이 없습니다: {env_file}")
        if env_example_file.exists():
            print(f"💡 {env_example_file}을 복사하여 .env 파일을 만들 수 있습니다:")
            print(f"   cp {env_example_file} {env_file}")
    
    return True


def test_milvus_connection():
    """Milvus 연결 테스트"""
    print_header("Milvus 연결 테스트")
    
    try:
        print("1. Milvus 서버 연결 시도...")
        
        # 연결 객체 생성
        conn = MilvusConnection()
        
        # 연결 시도
        if conn.connect():
            print("✅ Milvus 연결 성공!")
            
            # 서버 정보 확인
            print("\n2. 서버 정보 확인...")
            if conn.check_connection():
                print("✅ 서버 상태 정상!")
                
                # 기존 컬렉션 확인
                print("\n3. 기존 컬렉션 확인...")
                collections = conn.list_collections()
                if collections:
                    print(f"기존 컬렉션: {collections}")
                else:
                    print("기존 컬렉션이 없습니다.")
                
                # 연결 해제
                conn.disconnect()
                print("\n✅ Milvus 연결 테스트 완료!")
                return True
            else:
                print("❌ 서버 상태 확인 실패!")
                return False
        else:
            print("❌ Milvus 연결 실패!")
            print("\n문제 해결 방법:")
            print("1. Milvus 서비스가 실행 중인지 확인: docker-compose ps")
            print("2. 포트가 차단되지 않았는지 확인: netstat -an | grep 19530")
            print("3. 방화벽 설정 확인")
            return False
            
    except Exception as e:
        print(f"❌ 연결 테스트 중 오류 발생: {e}")
        return False


def show_web_interfaces():
    """웹 인터페이스 정보 표시"""
    print_header("웹 인터페이스 접속 정보")
    
    interfaces = [
        {
            'name': 'Attu (Milvus 관리 도구)',
            'url': 'http://localhost:3000',
            'description': 'Milvus 컬렉션 및 데이터 관리'
        },
        {
            'name': 'MinIO 콘솔',
            'url': 'http://localhost:9011',
            'description': '객체 스토리지 관리 (ID: minioadmin, PW: minioadmin)'
        },
        {
            'name': 'Grafana (모니터링)',
            'url': 'http://localhost:3001',
            'description': '시스템 모니터링 대시보드 (ID: admin, PW: admin)'
        },
        {
            'name': 'Prometheus',
            'url': 'http://localhost:9090',
            'description': '메트릭 수집 및 모니터링'
        }
    ]
    
    print("사용 가능한 웹 인터페이스:")
    for interface in interfaces:
        print(f"\n🌐 {interface['name']}")
        print(f"   URL: {interface['url']}")
        print(f"   설명: {interface['description']}")


def show_next_steps():
    """다음 단계 안내"""
    print_header("다음 단계")
    
    print("환경 구축이 완료되었습니다! 다음 실습을 진행하세요:")
    print()
    print("1. 기본 연결 실습:")
    print("   python step01_basics/02_basic_connection.py")
    print()
    print("2. 컬렉션 관리 실습:")
    print("   python step01_basics/03_collection_management.py")
    print()
    print("3. 전체 환경 테스트:")
    print("   python common/test_connection.py")
    print()
    print("💡 각 실습 코드는 주석과 함께 작성되어 있어 학습에 도움이 됩니다.")


def main():
    """메인 함수"""
    print("🚀 Milvus 학습 프로젝트 - 1.1 환경 구축")
    print(f"실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 환경 확인 단계
    checks = [
        ("Docker 환경", check_docker),
        ("Milvus 컨테이너", check_milvus_containers),
        ("Python 환경", check_python_environment),
        ("환경 변수", check_environment_variables),
        ("Milvus 연결", test_milvus_connection)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} 확인 중 오류: {e}")
            results.append((check_name, False))
    
    # 결과 요약
    print_header("환경 구축 결과 요약")
    
    passed = 0
    for check_name, result in results:
        status = "✅ 정상" if result else "❌ 문제"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 검사 항목: {len(results)}개")
    print(f"정상: {passed}개")
    print(f"문제: {len(results) - passed}개")
    
    if passed == len(results):
        print("\n🎉 환경 구축이 성공적으로 완료되었습니다!")
        show_web_interfaces()
        show_next_steps()
    else:
        print(f"\n⚠️  {len(results) - passed}개 항목에 문제가 있습니다.")
        print("문제를 해결한 후 다시 실행해주세요.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 