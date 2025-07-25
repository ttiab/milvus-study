#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus Blue-Green 배포 전략

이 스크립트는 Milvus 서비스의 무중단 배포를 위한 Blue-Green 배포 전략을 구현합니다.
트래픽 스위칭, 롤백 전략, 배포 검증 등 프로덕션 환경에서 필요한 모든 기능을 다룹니다.
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum

# 공통 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class DeploymentColor(Enum):
    """배포 색상"""
    BLUE = "blue"
    GREEN = "green"

class DeploymentStatus(Enum):
    """배포 상태"""
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    TESTING = "testing"
    ACTIVE = "active"
    DRAINING = "draining"
    FAILED = "failed"

class BlueGreenDeploymentManager:
    """Blue-Green 배포 관리자"""
    
    def __init__(self):
        self.namespace = "milvus-production"
        self.blue_service = "milvus-blue"
        self.green_service = "milvus-green"
        self.main_service = "milvus-main"
        
        # 배포 상태 추적
        self.deployments = {
            DeploymentColor.BLUE: {
                'status': DeploymentStatus.ACTIVE,
                'version': 'v1.0.0',
                'replicas': 3,
                'health_score': 100,
                'traffic_weight': 100
            },
            DeploymentColor.GREEN: {
                'status': DeploymentStatus.INACTIVE,
                'version': 'v1.1.0',
                'replicas': 0,
                'health_score': 0,
                'traffic_weight': 0
            }
        }
        
        self.deployment_logs = []
        self.monitoring_active = False
        self.rollback_enabled = True
    
    def log_event(self, event: str, level: str = "INFO"):
        """이벤트 로깅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {event}"
        self.deployment_logs.append(log_entry)
        print(f"  📝 {log_entry}")
    
    def get_inactive_deployment(self) -> DeploymentColor:
        """비활성 배포 환경 반환"""
        for color, deployment in self.deployments.items():
            if deployment['status'] == DeploymentStatus.INACTIVE:
                return color
        return DeploymentColor.GREEN  # 기본값
    
    def get_active_deployment(self) -> DeploymentColor:
        """활성 배포 환경 반환"""
        for color, deployment in self.deployments.items():
            if deployment['status'] == DeploymentStatus.ACTIVE:
                return color
        return DeploymentColor.BLUE  # 기본값
    
    def create_deployment_manifests(self):
        """Blue-Green 배포 매니페스트 생성"""
        print("📋 Blue-Green 배포 매니페스트 생성 중...")
        
        manifests_dir = Path("blue-green-manifests")
        manifests_dir.mkdir(exist_ok=True)
        
        # Blue 배포 매니페스트
        blue_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'milvus-blue',
                'namespace': self.namespace,
                'labels': {
                    'app': 'milvus',
                    'version': 'blue',
                    'environment': 'production'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'milvus',
                        'version': 'blue'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'milvus',
                            'version': 'blue'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'milvus',
                            'image': 'milvusdb/milvus:v2.4.0',
                            'ports': [
                                {'containerPort': 19530, 'name': 'grpc'},
                                {'containerPort': 9091, 'name': 'http'}
                            ],
                            'env': [
                                {'name': 'DEPLOYMENT_COLOR', 'value': 'blue'},
                                {'name': 'ENVIRONMENT', 'value': 'production'}
                            ],
                            'resources': {
                                'requests': {'cpu': '2000m', 'memory': '8Gi'},
                                'limits': {'cpu': '4000m', 'memory': '16Gi'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/healthz', 'port': 9091},
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/readiness', 'port': 9091},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Green 배포 매니페스트 (Blue와 거의 동일하지만 version 라벨이 다름)
        green_deployment = blue_deployment.copy()
        green_deployment['metadata']['name'] = 'milvus-green'
        green_deployment['metadata']['labels']['version'] = 'green'
        green_deployment['spec']['selector']['matchLabels']['version'] = 'green'
        green_deployment['spec']['template']['metadata']['labels']['version'] = 'green'
        green_deployment['spec']['template']['spec']['containers'][0]['env'][0]['value'] = 'green'
        green_deployment['spec']['replicas'] = 0  # 초기에는 비활성
        
        # 서비스 매니페스트들
        services = {
            'blue-service.yaml': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'milvus-blue',
                    'namespace': self.namespace
                },
                'spec': {
                    'selector': {'app': 'milvus', 'version': 'blue'},
                    'ports': [
                        {'name': 'grpc', 'port': 19530, 'targetPort': 19530},
                        {'name': 'http', 'port': 9091, 'targetPort': 9091}
                    ]
                }
            },
            'green-service.yaml': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'milvus-green',
                    'namespace': self.namespace
                },
                'spec': {
                    'selector': {'app': 'milvus', 'version': 'green'},
                    'ports': [
                        {'name': 'grpc', 'port': 19530, 'targetPort': 19530},
                        {'name': 'http', 'port': 9091, 'targetPort': 9091}
                    ]
                }
            },
            'main-service.yaml': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'milvus-main',
                    'namespace': self.namespace,
                    'annotations': {
                        'service.beta.kubernetes.io/aws-load-balancer-type': 'nlb'
                    }
                },
                'spec': {
                    'type': 'LoadBalancer',
                    'selector': {'app': 'milvus', 'version': 'blue'},  # 초기에는 blue
                    'ports': [
                        {'name': 'grpc', 'port': 19530, 'targetPort': 19530}
                    ]
                }
            }
        }
        
        # 매니페스트 파일 저장
        import yaml
        
        with open(manifests_dir / 'blue-deployment.yaml', 'w') as f:
            yaml.dump(blue_deployment, f, default_flow_style=False)
        
        with open(manifests_dir / 'green-deployment.yaml', 'w') as f:
            yaml.dump(green_deployment, f, default_flow_style=False)
        
        for filename, service in services.items():
            with open(manifests_dir / filename, 'w') as f:
                yaml.dump(service, f, default_flow_style=False)
        
        print("  ✅ Blue-Green 매니페스트 생성 완료")
        print(f"  📁 위치: {manifests_dir.absolute()}")
    
    def simulate_health_check(self, color: DeploymentColor) -> Dict[str, Any]:
        """헬스 체크 시뮬레이션"""
        deployment = self.deployments[color]
        
        if deployment['status'] == DeploymentStatus.INACTIVE:
            return {'status': 'unhealthy', 'score': 0, 'details': 'Service not running'}
        
        # 시뮬레이션된 헬스 체크 결과
        base_score = 85 + random.randint(0, 15)
        
        # 새 배포는 초기에 낮은 점수를 가질 수 있음
        if deployment['status'] == DeploymentStatus.DEPLOYING:
            base_score = max(50, base_score - 20)
        elif deployment['status'] == DeploymentStatus.TESTING:
            base_score = max(70, base_score - 10)
        
        health_details = {
            'status': 'healthy' if base_score >= 80 else 'degraded' if base_score >= 60 else 'unhealthy',
            'score': base_score,
            'details': {
                'cpu_usage': f"{random.randint(30, 70)}%",
                'memory_usage': f"{random.randint(40, 80)}%",
                'response_time': f"{random.randint(50, 200)}ms",
                'error_rate': f"{random.uniform(0, 2):.2f}%",
                'active_connections': random.randint(100, 500)
            }
        }
        
        # 헬스 스코어 업데이트
        self.deployments[color]['health_score'] = base_score
        
        return health_details
    
    def deploy_new_version(self, new_version: str, target_color: Optional[DeploymentColor] = None):
        """새 버전 배포"""
        if target_color is None:
            target_color = self.get_inactive_deployment()
        
        self.log_event(f"새 버전 {new_version} 배포 시작 ({target_color.value} 환경)")
        
        # 1. 비활성 환경에 새 버전 배포
        self.deployments[target_color]['status'] = DeploymentStatus.DEPLOYING
        self.deployments[target_color]['version'] = new_version
        self.deployments[target_color]['replicas'] = 3
        
        print(f"\n🚀 {target_color.value.upper()} 환경에 배포 중...")
        
        # 배포 시뮬레이션
        deployment_steps = [
            "Docker 이미지 다운로드",
            "Pod 생성",
            "컨테이너 시작",
            "헬스 체크 대기",
            "Ready 상태 확인"
        ]
        
        for i, step in enumerate(deployment_steps, 1):
            print(f"  {i}/5 {step}...")
            time.sleep(1)
            
        self.deployments[target_color]['status'] = DeploymentStatus.TESTING
        self.log_event(f"{target_color.value} 환경 배포 완료 - 테스트 단계 진입")
        
        return True
    
    def run_deployment_tests(self, color: DeploymentColor) -> bool:
        """배포 테스트 실행"""
        print(f"\n🧪 {color.value.upper()} 환경 테스트 실행 중...")
        
        tests = [
            ("연결성 테스트", True),
            ("기본 CRUD 테스트", True),
            ("성능 테스트", True),
            ("데이터 일관성 테스트", True),
            ("API 호환성 테스트", random.choice([True, True, True, False]))  # 가끔 실패
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, result in tests:
            time.sleep(0.5)
            if result:
                print(f"  ✅ {test_name}")
                passed_tests += 1
            else:
                print(f"  ❌ {test_name}")
                self.log_event(f"테스트 실패: {test_name}", "ERROR")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n  📊 테스트 결과: {passed_tests}/{total_tests} 통과 ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            self.log_event(f"{color.value} 환경 테스트 통과")
            return True
        else:
            self.log_event(f"{color.value} 환경 테스트 실패", "ERROR")
            self.deployments[color]['status'] = DeploymentStatus.FAILED
            return False
    
    def gradual_traffic_switch(self, from_color: DeploymentColor, to_color: DeploymentColor):
        """점진적 트래픽 스위칭"""
        print(f"\n🔄 트래픽 점진적 전환: {from_color.value} → {to_color.value}")
        
        # 트래픽 전환 단계
        traffic_steps = [10, 25, 50, 75, 100]
        
        for step in traffic_steps:
            print(f"  📊 트래픽 {step}% 전환 중...")
            
            # 트래픽 가중치 업데이트
            self.deployments[to_color]['traffic_weight'] = step
            self.deployments[from_color]['traffic_weight'] = 100 - step
            
            # 시뮬레이션된 트래픽 전환 대기
            time.sleep(2)
            
            # 헬스 체크
            new_health = self.simulate_health_check(to_color)
            old_health = self.simulate_health_check(from_color)
            
            print(f"    {to_color.value}: {new_health['score']}점 | {from_color.value}: {old_health['score']}점")
            
            # 문제 감지 시 롤백
            if new_health['score'] < 70:
                print(f"  ⚠️  {to_color.value} 환경 성능 저하 감지!")
                if self.rollback_enabled:
                    print(f"  🔙 자동 롤백 실행...")
                    self.rollback_deployment(from_color)
                    return False
            
            self.log_event(f"트래픽 {step}% 전환 완료")
        
        # 최종 상태 업데이트
        self.deployments[to_color]['status'] = DeploymentStatus.ACTIVE
        self.deployments[from_color]['status'] = DeploymentStatus.DRAINING
        
        return True
    
    def complete_deployment_switch(self, old_color: DeploymentColor, new_color: DeploymentColor):
        """배포 전환 완료"""
        print(f"\n✅ 배포 전환 완료: {new_color.value}가 새로운 활성 환경")
        
        # 구 환경 정리
        print(f"  🧹 {old_color.value} 환경 정리 중...")
        time.sleep(1)
        
        self.deployments[old_color]['status'] = DeploymentStatus.INACTIVE
        self.deployments[old_color]['replicas'] = 0
        self.deployments[old_color]['traffic_weight'] = 0
        
        self.log_event(f"Blue-Green 배포 완료: {new_color.value} 환경 활성화")
        print(f"  🎉 {new_color.value} 환경이 프로덕션 트래픽을 처리합니다")
    
    def rollback_deployment(self, target_color: DeploymentColor):
        """배포 롤백"""
        print(f"\n🔙 {target_color.value} 환경으로 롤백 실행...")
        
        # 즉시 트래픽 전환
        for color in DeploymentColor:
            if color == target_color:
                self.deployments[color]['traffic_weight'] = 100
                self.deployments[color]['status'] = DeploymentStatus.ACTIVE
            else:
                self.deployments[color]['traffic_weight'] = 0
                self.deployments[color]['status'] = DeploymentStatus.FAILED
        
        self.log_event(f"긴급 롤백 완료: {target_color.value} 환경으로 복구", "WARN")
        print(f"  ✅ {target_color.value} 환경으로 즉시 롤백됨")
    
    def monitor_deployment(self, duration_minutes: int = 5):
        """배포 모니터링"""
        print(f"\n📊 배포 모니터링 시작 ({duration_minutes}분간)")
        
        self.monitoring_active = True
        start_time = datetime.now()
        
        def monitoring_loop():
            while self.monitoring_active:
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds() / 60
                
                if elapsed >= duration_minutes:
                    break
                
                print(f"\n⏰ 모니터링 {elapsed:.1f}/{duration_minutes}분")
                
                for color in DeploymentColor:
                    deployment = self.deployments[color]
                    if deployment['status'] != DeploymentStatus.INACTIVE:
                        health = self.simulate_health_check(color)
                        status_emoji = "🟢" if health['score'] >= 80 else "🟡" if health['score'] >= 60 else "🔴"
                        
                        print(f"  {status_emoji} {color.value.upper()}: {deployment['status'].value} | "
                              f"건강도: {health['score']}점 | 트래픽: {deployment['traffic_weight']}%")
                
                time.sleep(30)  # 30초마다 체크
        
        # 모니터링을 별도 스레드에서 실행
        monitor_thread = threading.Thread(target=monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 간단한 시뮬레이션을 위해 잠시 대기
        time.sleep(3)
        self.monitoring_active = False
        
        print("  📊 모니터링 완료")
    
    def create_blue_green_scripts(self):
        """Blue-Green 배포 스크립트 생성"""
        print("📜 Blue-Green 배포 스크립트 생성 중...")
        
        scripts_dir = Path("blue-green-scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        # 배포 스크립트
        deploy_script = '''#!/bin/bash
set -e

# Colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

NAMESPACE="milvus-production"
NEW_VERSION=""
TARGET_COLOR=""
DRY_RUN=false

usage() {
    echo "Usage: $0 -v VERSION [-c COLOR] [-d] [-h]"
    echo "  -v VERSION    New version to deploy"
    echo "  -c COLOR      Target color (blue|green), auto-detect if not specified"
    echo "  -d            Dry run mode"
    echo "  -h            Show this help"
    exit 1
}

while getopts "v:c:dh" opt; do
    case $opt in
        v) NEW_VERSION="$OPTARG" ;;
        c) TARGET_COLOR="$OPTARG" ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}Error: Version is required${NC}"
    usage
fi

# Auto-detect target color if not specified
if [ -z "$TARGET_COLOR" ]; then
    ACTIVE_COLOR=$(kubectl get service milvus-main -n $NAMESPACE -o jsonpath='{.spec.selector.version}')
    if [ "$ACTIVE_COLOR" = "blue" ]; then
        TARGET_COLOR="green"
    else
        TARGET_COLOR="blue"
    fi
    echo -e "${YELLOW}Auto-detected target color: $TARGET_COLOR${NC}"
fi

echo -e "${GREEN}🚀 Starting Blue-Green deployment${NC}"
echo -e "Version: ${YELLOW}$NEW_VERSION${NC}"
echo -e "Target: ${BLUE}$TARGET_COLOR${NC}"
echo -e "Namespace: ${YELLOW}$NAMESPACE${NC}"

# 1. Deploy new version
echo -e "${GREEN}📦 Deploying to $TARGET_COLOR environment...${NC}"
kubectl set image deployment/milvus-$TARGET_COLOR milvus=milvusdb/milvus:$NEW_VERSION -n $NAMESPACE

if [ "$DRY_RUN" = false ]; then
    # 2. Wait for deployment
    echo -e "${GREEN}⏳ Waiting for deployment to be ready...${NC}"
    kubectl rollout status deployment/milvus-$TARGET_COLOR -n $NAMESPACE --timeout=600s
    
    # 3. Health check
    echo -e "${GREEN}🏥 Running health checks...${NC}"
    sleep 30
    
    # 4. Switch traffic (this would be more sophisticated in real scenarios)
    echo -e "${GREEN}🔄 Switching traffic to $TARGET_COLOR...${NC}"
    kubectl patch service milvus-main -n $NAMESPACE -p '{"spec":{"selector":{"version":"'$TARGET_COLOR'"}}}'
    
    echo -e "${GREEN}✅ Blue-Green deployment completed!${NC}"
    echo -e "${GREEN}Active environment: $TARGET_COLOR${NC}"
else
    echo -e "${YELLOW}🧪 Dry run completed${NC}"
fi
'''
        
        # 롤백 스크립트
        rollback_script = '''#!/bin/bash
set -e

GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

NAMESPACE="milvus-production"
TARGET_COLOR=""

usage() {
    echo "Usage: $0 [-c COLOR] [-h]"
    echo "  -c COLOR      Target color to rollback to (blue|green)"
    echo "  -h            Show this help"
    exit 1
}

while getopts "c:h" opt; do
    case $opt in
        c) TARGET_COLOR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Auto-detect previous color if not specified
if [ -z "$TARGET_COLOR" ]; then
    CURRENT_COLOR=$(kubectl get service milvus-main -n $NAMESPACE -o jsonpath='{.spec.selector.version}')
    if [ "$CURRENT_COLOR" = "blue" ]; then
        TARGET_COLOR="green"
    else
        TARGET_COLOR="blue"
    fi
fi

echo -e "${RED}🔙 Emergency rollback to $TARGET_COLOR environment${NC}"

# Immediate traffic switch
kubectl patch service milvus-main -n $NAMESPACE -p '{"spec":{"selector":{"version":"'$TARGET_COLOR'"}}}'

echo -e "${GREEN}✅ Rollback completed!${NC}"
echo -e "${GREEN}Active environment: $TARGET_COLOR${NC}"
'''
        
        # 상태 확인 스크립트
        status_script = '''#!/bin/bash

NAMESPACE="milvus-production"

echo "🔍 Blue-Green Deployment Status"
echo "================================"

# Current active environment
ACTIVE_COLOR=$(kubectl get service milvus-main -n $NAMESPACE -o jsonpath='{.spec.selector.version}')
echo "Active Environment: $ACTIVE_COLOR"

echo ""
echo "🔵 Blue Environment:"
kubectl get deployment milvus-blue -n $NAMESPACE -o custom-columns=READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,UP-TO-DATE:.status.updatedReplicas

echo ""
echo "🟢 Green Environment:"
kubectl get deployment milvus-green -n $NAMESPACE -o custom-columns=READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,UP-TO-DATE:.status.updatedReplicas

echo ""
echo "📊 Pod Status:"
kubectl get pods -n $NAMESPACE -l app=milvus -o wide
'''
        
        # 스크립트 파일 저장
        scripts = [
            ('blue-green-deploy.sh', deploy_script),
            ('blue-green-rollback.sh', rollback_script),
            ('blue-green-status.sh', status_script)
        ]
        
        for filename, content in scripts:
            with open(scripts_dir / filename, 'w') as f:
                f.write(content)
        
        print("  ✅ Blue-Green 스크립트 생성 완료")
        print("  💫 실행 권한 설정 필요:")
        for filename, _ in scripts:
            print(f"    $ chmod +x blue-green-scripts/{filename}")
    
    def demonstrate_blue_green_deployment(self):
        """Blue-Green 배포 시연"""
        print("\n🎭 Blue-Green 배포 시연 시작...")
        
        # 현재 상태 출력
        self.show_current_status()
        
        # 1. 새 버전 배포
        new_version = "v1.1.0"
        target_color = self.get_inactive_deployment()
        
        if not self.deploy_new_version(new_version, target_color):
            return False
        
        # 2. 배포 테스트
        if not self.run_deployment_tests(target_color):
            print("  ❌ 배포 테스트 실패 - 롤백 실행")
            self.rollback_deployment(self.get_active_deployment())
            return False
        
        # 3. 모니터링 시작
        self.monitor_deployment(1)  # 1분간 모니터링
        
        # 4. 트래픽 점진적 전환
        active_color = self.get_active_deployment()
        if not self.gradual_traffic_switch(active_color, target_color):
            return False
        
        # 5. 배포 완료
        self.complete_deployment_switch(active_color, target_color)
        
        # 최종 상태 출력
        self.show_current_status()
        
        return True
    
    def show_current_status(self):
        """현재 배포 상태 출력"""
        print("\n📊 현재 Blue-Green 배포 상태:")
        print("=" * 50)
        
        for color, deployment in self.deployments.items():
            status_emoji = {
                DeploymentStatus.ACTIVE: "🟢",
                DeploymentStatus.INACTIVE: "⚫",
                DeploymentStatus.DEPLOYING: "🟡",
                DeploymentStatus.TESTING: "🔵",
                DeploymentStatus.DRAINING: "🟠",
                DeploymentStatus.FAILED: "🔴"
            }.get(deployment['status'], "❓")
            
            print(f"{status_emoji} {color.value.upper()}: {deployment['status'].value}")
            print(f"   버전: {deployment['version']}")
            print(f"   복제본: {deployment['replicas']}")
            print(f"   건강도: {deployment['health_score']}점")
            print(f"   트래픽: {deployment['traffic_weight']}%")
        
        print("=" * 50)
    
    def show_deployment_logs(self):
        """배포 로그 출력"""
        print("\n📋 배포 로그:")
        print("=" * 60)
        
        if not self.deployment_logs:
            print("  로그가 없습니다.")
        else:
            for log in self.deployment_logs[-10:]:  # 최근 10개만
                print(f"  {log}")
        
        print("=" * 60)

def main():
    """메인 실행 함수"""
    print("🔵 Milvus Blue-Green 배포 전략")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = BlueGreenDeploymentManager()
    
    try:
        # 1. 매니페스트 생성
        print("\n" + "=" * 80)
        print(" 📋 Blue-Green 배포 환경 구축")
        print("=" * 80)
        manager.create_deployment_manifests()
        
        # 2. 스크립트 생성
        print("\n" + "=" * 80)
        print(" 📜 배포 자동화 스크립트")
        print("=" * 80)
        manager.create_blue_green_scripts()
        
        # 3. 배포 시연
        print("\n" + "=" * 80)
        print(" 🎭 Blue-Green 배포 시연")
        print("=" * 80)
        
        success = manager.demonstrate_blue_green_deployment()
        
        # 4. 로그 출력
        print("\n" + "=" * 80)
        print(" 📋 배포 이력")
        print("=" * 80)
        manager.show_deployment_logs()
        
        # 5. 요약
        print("\n" + "=" * 80)
        print(" 📊 Blue-Green 배포 완료")
        print("=" * 80)
        
        if success:
            print("🎉 Blue-Green 배포가 성공적으로 완료되었습니다!")
        else:
            print("⚠️  배포 중 문제가 발생했지만 안전하게 롤백되었습니다.")
        
        print("\n✅ 생성된 리소스:")
        resources = [
            "blue-green-manifests/blue-deployment.yaml",
            "blue-green-manifests/green-deployment.yaml", 
            "blue-green-manifests/blue-service.yaml",
            "blue-green-manifests/green-service.yaml",
            "blue-green-manifests/main-service.yaml",
            "blue-green-scripts/blue-green-deploy.sh",
            "blue-green-scripts/blue-green-rollback.sh",
            "blue-green-scripts/blue-green-status.sh"
        ]
        
        for resource in resources:
            print(f"  📄 {resource}")
        
        print("\n💡 Blue-Green 배포 장점:")
        advantages = [
            "✅ 무중단 서비스 제공",
            "✅ 즉시 롤백 가능",
            "✅ 프로덕션 환경에서 테스트",
            "✅ 위험 최소화",
            "✅ 점진적 트래픽 전환"
        ]
        
        for advantage in advantages:
            print(f"  {advantage}")
        
        print("\n🚀 배포 명령어 예시:")
        commands = [
            "# 새 버전 배포",
            "./blue-green-scripts/blue-green-deploy.sh -v v1.2.0",
            "",
            "# 특정 환경에 배포",
            "./blue-green-scripts/blue-green-deploy.sh -v v1.2.0 -c green",
            "",
            "# 긴급 롤백",
            "./blue-green-scripts/blue-green-rollback.sh -c blue",
            "",
            "# 상태 확인",
            "./blue-green-scripts/blue-green-status.sh"
        ]
        
        for cmd in commands:
            if cmd.startswith('#') or cmd == '':
                print(f"  {cmd}")
            else:
                print(f"  $ {cmd}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Blue-Green 배포 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 무중단 배포 전략 이해")
    print("  • 트래픽 점진적 전환 메커니즘")
    print("  • 자동 롤백 및 장애 복구")
    print("  • 프로덕션 환경 안전성 확보")
    
    print("\n🚀 다음 단계:")
    print("  python step05_production/04_ab_testing.py")

if __name__ == "__main__":
    main() 