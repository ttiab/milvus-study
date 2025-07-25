#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus A/B 테스팅 시스템

이 스크립트는 Milvus 서비스의 A/B 테스팅을 구현합니다.
성능 비교, 기능 테스트, 통계적 유의성 검증 등을 통해 
프로덕션 환경에서 안전하게 새로운 기능을 검증합니다.
"""

import os
import sys
import time
import random
import statistics
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# 공통 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestVariant(Enum):
    """테스트 변형"""
    CONTROL = "A"  # 기존 버전
    TREATMENT = "B"  # 새 버전

@dataclass
class TestMetrics:
    """테스트 메트릭"""
    variant: TestVariant
    response_time_ms: List[float]
    throughput_qps: List[float]
    error_rate: List[float]
    memory_usage_mb: List[float]
    cpu_usage_percent: List[float]
    search_accuracy: List[float]
    user_satisfaction: List[float]
    timestamp: datetime

@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""
    name: str
    description: str
    traffic_split: Dict[TestVariant, int]  # 트래픽 분할 비율
    duration_minutes: int
    sample_size: int
    confidence_level: float
    success_criteria: Dict[str, float]

class ABTestingManager:
    """A/B 테스팅 관리자"""
    
    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.metrics_data: Dict[str, List[TestMetrics]] = {}
        self.active_tests: Dict[str, bool] = {}
        self.results_dir = Path("ab-test-results")
        self.results_dir.mkdir(exist_ok=True)
    
    def create_ab_test_config(self, test_name: str) -> ABTestConfig:
        """A/B 테스트 설정 생성"""
        configs = {
            "search_algorithm_comparison": ABTestConfig(
                name="검색 알고리즘 성능 비교",
                description="기존 IVF_FLAT vs 새로운 HNSW 알고리즘 성능 비교",
                traffic_split={TestVariant.CONTROL: 50, TestVariant.TREATMENT: 50},
                duration_minutes=30,
                sample_size=1000,
                confidence_level=0.95,
                success_criteria={
                    'response_time_improvement': 0.2,  # 20% 개선
                    'accuracy_threshold': 0.95,
                    'error_rate_max': 0.01
                }
            ),
            "memory_optimization": ABTestConfig(
                name="메모리 최적화 테스트",
                description="메모리 사용량 최적화 버전 효과 검증",
                traffic_split={TestVariant.CONTROL: 70, TestVariant.TREATMENT: 30},
                duration_minutes=45,
                sample_size=800,
                confidence_level=0.95,
                success_criteria={
                    'memory_reduction': 0.25,  # 25% 메모리 절약
                    'performance_degradation_max': 0.05  # 5% 이하 성능 저하
                }
            ),
            "new_feature_rollout": ABTestConfig(
                name="신기능 점진적 출시",
                description="새로운 벡터 검색 기능의 사용자 경험 테스트",
                traffic_split={TestVariant.CONTROL: 90, TestVariant.TREATMENT: 10},
                duration_minutes=60,
                sample_size=500,
                confidence_level=0.90,
                success_criteria={
                    'user_satisfaction_min': 4.0,  # 5점 만점 중 4점 이상
                    'adoption_rate_min': 0.30  # 30% 이상 사용률
                }
            )
        }
        
        return configs.get(test_name, configs["search_algorithm_comparison"])
    
    def setup_ab_test_environment(self):
        """A/B 테스트 환경 설정"""
        print("🧪 A/B 테스트 환경 설정 중...")
        
        # Kubernetes 매니페스트 생성
        manifests_dir = Path("ab-test-manifests")
        manifests_dir.mkdir(exist_ok=True)
        
        # Traffic Splitting을 위한 Istio VirtualService
        virtual_service = {
            'apiVersion': 'networking.istio.io/v1beta1',
            'kind': 'VirtualService',
            'metadata': {
                'name': 'milvus-ab-test',
                'namespace': 'milvus-production'
            },
            'spec': {
                'hosts': ['milvus.example.com'],
                'http': [{
                    'match': [{
                        'headers': {
                            'ab-test-group': {'exact': 'treatment'}
                        }
                    }],
                    'route': [{
                        'destination': {
                            'host': 'milvus-treatment',
                            'port': {'number': 19530}
                        }
                    }]
                }, {
                    'route': [{
                        'destination': {
                            'host': 'milvus-control',
                            'port': {'number': 19530}
                        },
                        'weight': 50
                    }, {
                        'destination': {
                            'host': 'milvus-treatment',
                            'port': {'number': 19530}
                        },
                        'weight': 50
                    }]
                }]
            }
        }
        
        # ConfigMap for feature flags
        feature_flags_config = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'ab-test-config',
                'namespace': 'milvus-production'
            },
            'data': {
                'config.json': json.dumps({
                    'ab_tests': {
                        'search_algorithm': {
                            'enabled': True,
                            'treatment_percentage': 50,
                            'feature_flags': {
                                'use_hnsw': True,
                                'optimize_memory': True
                            }
                        }
                    }
                }, indent=2)
            }
        }
        
        # 매니페스트 저장
        import yaml
        
        with open(manifests_dir / 'virtual-service.yaml', 'w') as f:
            yaml.dump(virtual_service, f, default_flow_style=False)
        
        with open(manifests_dir / 'feature-flags-config.yaml', 'w') as f:
            yaml.dump(feature_flags_config, f, default_flow_style=False)
        
        print("  ✅ A/B 테스트 환경 매니페스트 생성됨")
        
        # Monitoring Dashboard 설정
        self.create_ab_test_dashboard()
    
    def create_ab_test_dashboard(self):
        """A/B 테스트 대시보드 생성"""
        print("📊 A/B 테스트 대시보드 설정 중...")
        
        dashboard_config = {
            'dashboard': {
                'title': 'Milvus A/B Testing Dashboard',
                'panels': [
                    {
                        'title': 'Response Time Comparison',
                        'type': 'graph',
                        'metrics': [
                            'avg(milvus_request_duration_seconds{variant="control"})',
                            'avg(milvus_request_duration_seconds{variant="treatment"})'
                        ]
                    },
                    {
                        'title': 'Throughput (QPS)',
                        'type': 'graph',
                        'metrics': [
                            'rate(milvus_requests_total{variant="control"}[5m])',
                            'rate(milvus_requests_total{variant="treatment"}[5m])'
                        ]
                    },
                    {
                        'title': 'Error Rate',
                        'type': 'graph',
                        'metrics': [
                            'rate(milvus_errors_total{variant="control"}[5m])',
                            'rate(milvus_errors_total{variant="treatment"}[5m])'
                        ]
                    },
                    {
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'metrics': [
                            'avg(milvus_memory_usage_bytes{variant="control"})',
                            'avg(milvus_memory_usage_bytes{variant="treatment"})'
                        ]
                    }
                ]
            }
        }
        
        with open(self.results_dir / 'grafana-dashboard.json', 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        print("  ✅ Grafana 대시보드 설정 생성됨")
    
    def simulate_test_traffic(self, config: ABTestConfig, variant: TestVariant) -> TestMetrics:
        """테스트 트래픽 시뮬레이션"""
        
        # 시뮬레이션된 메트릭 생성
        base_response_time = 50 if variant == TestVariant.CONTROL else 40  # Treatment가 더 빠름
        base_throughput = 1000 if variant == TestVariant.CONTROL else 1200  # Treatment가 더 높음
        base_error_rate = 0.005 if variant == TestVariant.CONTROL else 0.003  # Treatment가 더 낮음
        base_memory = 2048 if variant == TestVariant.CONTROL else 1600  # Treatment가 더 적음
        base_cpu = 65 if variant == TestVariant.CONTROL else 60  # Treatment가 더 낮음
        base_accuracy = 0.94 if variant == TestVariant.CONTROL else 0.96  # Treatment가 더 높음
        base_satisfaction = 3.8 if variant == TestVariant.CONTROL else 4.2  # Treatment가 더 높음
        
        # 랜덤 변동 추가
        sample_size = min(config.sample_size, 100)  # 시뮬레이션을 위해 제한
        
        response_times = [
            max(10, base_response_time + random.gauss(0, 10)) 
            for _ in range(sample_size)
        ]
        
        throughputs = [
            max(500, base_throughput + random.gauss(0, 100)) 
            for _ in range(sample_size)
        ]
        
        error_rates = [
            max(0, base_error_rate + random.gauss(0, 0.002)) 
            for _ in range(sample_size)
        ]
        
        memory_usages = [
            max(1000, base_memory + random.gauss(0, 200)) 
            for _ in range(sample_size)
        ]
        
        cpu_usages = [
            max(20, min(100, base_cpu + random.gauss(0, 10))) 
            for _ in range(sample_size)
        ]
        
        accuracies = [
            max(0.8, min(1.0, base_accuracy + random.gauss(0, 0.02))) 
            for _ in range(sample_size)
        ]
        
        satisfactions = [
            max(1.0, min(5.0, base_satisfaction + random.gauss(0, 0.3))) 
            for _ in range(sample_size)
        ]
        
        return TestMetrics(
            variant=variant,
            response_time_ms=response_times,
            throughput_qps=throughputs,
            error_rate=error_rates,
            memory_usage_mb=memory_usages,
            cpu_usage_percent=cpu_usages,
            search_accuracy=accuracies,
            user_satisfaction=satisfactions,
            timestamp=datetime.now()
        )
    
    def run_ab_test(self, test_name: str) -> bool:
        """A/B 테스트 실행"""
        print(f"\n🧪 A/B 테스트 실행: {test_name}")
        
        config = self.create_ab_test_config(test_name)
        self.tests[test_name] = config
        self.metrics_data[test_name] = []
        self.active_tests[test_name] = True
        
        print(f"  📋 테스트 설정:")
        print(f"    이름: {config.name}")
        print(f"    설명: {config.description}")
        print(f"    트래픽 분할: A({config.traffic_split[TestVariant.CONTROL]}%) / B({config.traffic_split[TestVariant.TREATMENT]}%)")
        print(f"    지속 시간: {config.duration_minutes}분")
        print(f"    샘플 크기: {config.sample_size}")
        print(f"    신뢰도: {config.confidence_level * 100}%")
        
        # 병렬로 양쪽 variant 테스트
        control_metrics = self.simulate_test_traffic(config, TestVariant.CONTROL)
        treatment_metrics = self.simulate_test_traffic(config, TestVariant.TREATMENT)
        
        self.metrics_data[test_name] = [control_metrics, treatment_metrics]
        
        print(f"  ✅ 테스트 데이터 수집 완료")
        return True
    
    def calculate_statistical_significance(self, 
                                           control_data: List[float], 
                                           treatment_data: List[float]) -> Tuple[float, bool]:
        """통계적 유의성 계산 (간소화된 t-test)"""
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            return 0.0, False
        
        mean_control = statistics.mean(control_data)
        mean_treatment = statistics.mean(treatment_data)
        
        if len(control_data) == 1 and len(treatment_data) == 1:
            return 0.0, False
        
        # 간소화된 t-test 계산
        try:
            var_control = statistics.variance(control_data) if len(control_data) > 1 else 0
            var_treatment = statistics.variance(treatment_data) if len(treatment_data) > 1 else 0
            
            pooled_std = ((var_control + var_treatment) / 2) ** 0.5
            
            if pooled_std == 0:
                return 0.0, False
            
            t_stat = abs(mean_treatment - mean_control) / (pooled_std * (1/len(control_data) + 1/len(treatment_data)) ** 0.5)
            
            # 간소화된 p-value 계산 (실제로는 더 복잡한 계산 필요)
            p_value = max(0.001, 1 / (1 + t_stat))
            
            is_significant = p_value < 0.05
            
            return p_value, is_significant
            
        except (statistics.StatisticsError, ZeroDivisionError):
            return 0.0, False
    
    def analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """테스트 결과 분석"""
        print(f"\n📊 A/B 테스트 결과 분석: {test_name}")
        
        if test_name not in self.metrics_data or len(self.metrics_data[test_name]) < 2:
            print("  ❌ 충분한 테스트 데이터가 없습니다.")
            return {}
        
        control_metrics = self.metrics_data[test_name][0]
        treatment_metrics = self.metrics_data[test_name][1]
        config = self.tests[test_name]
        
        results = {}
        
        # 각 메트릭별 분석
        metrics_to_analyze = [
            ('response_time_ms', '응답 시간 (ms)', 'lower_is_better'),
            ('throughput_qps', '처리량 (QPS)', 'higher_is_better'),
            ('error_rate', '오류율', 'lower_is_better'),
            ('memory_usage_mb', '메모리 사용량 (MB)', 'lower_is_better'),
            ('cpu_usage_percent', 'CPU 사용률 (%)', 'lower_is_better'),
            ('search_accuracy', '검색 정확도', 'higher_is_better'),
            ('user_satisfaction', '사용자 만족도', 'higher_is_better')
        ]
        
        print("  📈 메트릭별 비교 결과:")
        
        for metric_name, display_name, direction in metrics_to_analyze:
            control_data = getattr(control_metrics, metric_name)
            treatment_data = getattr(treatment_metrics, metric_name)
            
            control_mean = statistics.mean(control_data)
            treatment_mean = statistics.mean(treatment_data)
            
            improvement = ((treatment_mean - control_mean) / control_mean) * 100
            if direction == 'lower_is_better':
                improvement = -improvement
            
            p_value, is_significant = self.calculate_statistical_significance(control_data, treatment_data)
            
            results[metric_name] = {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'improvement_percent': improvement,
                'p_value': p_value,
                'is_significant': is_significant,
                'direction': direction
            }
            
            significance_indicator = "✅" if is_significant else "⚠️"
            improvement_indicator = "📈" if improvement > 0 else "📉" if improvement < 0 else "➖"
            
            print(f"    {significance_indicator} {display_name}:")
            print(f"      A (Control): {control_mean:.2f}")
            print(f"      B (Treatment): {treatment_mean:.2f}")
            print(f"      {improvement_indicator} 변화: {improvement:+.1f}%")
            print(f"      p-value: {p_value:.3f}")
        
        # 성공 기준 체크
        print(f"\n  🎯 성공 기준 검증:")
        success_criteria_met = True
        
        for criterion, threshold in config.success_criteria.items():
            if criterion == 'response_time_improvement':
                actual = results.get('response_time_ms', {}).get('improvement_percent', 0)
                met = actual >= threshold * 100
                print(f"    {'✅' if met else '❌'} 응답시간 개선: {actual:.1f}% (목표: {threshold*100:.1f}%)")
                success_criteria_met = success_criteria_met and met
                
            elif criterion == 'accuracy_threshold':
                actual = results.get('search_accuracy', {}).get('treatment_mean', 0)
                met = actual >= threshold
                print(f"    {'✅' if met else '❌'} 검색 정확도: {actual:.3f} (목표: {threshold:.3f})")
                success_criteria_met = success_criteria_met and met
                
            elif criterion == 'error_rate_max':
                actual = results.get('error_rate', {}).get('treatment_mean', 1)
                met = actual <= threshold
                print(f"    {'✅' if met else '❌'} 오류율: {actual:.3f} (최대: {threshold:.3f})")
                success_criteria_met = success_criteria_met and met
        
        results['success_criteria_met'] = success_criteria_met
        results['recommendation'] = self.generate_recommendation(results, success_criteria_met)
        
        return results
    
    def generate_recommendation(self, results: Dict[str, Any], criteria_met: bool) -> str:
        """테스트 결과 기반 권장사항 생성"""
        
        if criteria_met:
            significant_improvements = []
            for metric, data in results.items():
                if isinstance(data, dict) and data.get('is_significant') and data.get('improvement_percent', 0) > 5:
                    significant_improvements.append(metric)
            
            if significant_improvements:
                return "🚀 PROCEED: Treatment 버전을 프로덕션에 배포할 것을 권장합니다. 통계적으로 유의한 성능 개선이 확인되었습니다."
            else:
                return "🤔 NEUTRAL: 성능 개선이 있지만 큰 차이는 없습니다. 다른 요인을 고려하여 결정하세요."
        else:
            critical_failures = []
            for metric, data in results.items():
                if isinstance(data, dict) and data.get('improvement_percent', 0) < -10:
                    critical_failures.append(metric)
            
            if critical_failures:
                return "🛑 STOP: Treatment 버전에 심각한 성능 저하가 있습니다. 추가 최적화 후 재테스트를 권장합니다."
            else:
                return "⚠️ CAUTION: 일부 성공 기준을 충족하지 못했습니다. 개선 후 재테스트를 고려하세요."
    
    def save_test_results(self, test_name: str, results: Dict[str, Any]):
        """테스트 결과 저장"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        
        report = {
            'test_name': test_name,
            'config': {
                'name': self.tests[test_name].name,
                'description': self.tests[test_name].description,
                'traffic_split': {k.value: v for k, v in self.tests[test_name].traffic_split.items()},
                'duration_minutes': self.tests[test_name].duration_minutes,
                'sample_size': self.tests[test_name].sample_size,
                'confidence_level': self.tests[test_name].confidence_level,
                'success_criteria': self.tests[test_name].success_criteria
            },
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'recommendations': results.get('recommendation', '')
        }
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  💾 테스트 결과 저장: {filename}")
    
    def create_ab_test_scripts(self):
        """A/B 테스트 자동화 스크립트 생성"""
        print("📜 A/B 테스트 스크립트 생성 중...")
        
        scripts_dir = Path("ab-test-scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        # A/B 테스트 시작 스크립트
        start_script = '''#!/bin/bash
set -e

GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m'

TEST_NAME=""
TREATMENT_PERCENTAGE=50
DURATION_MINUTES=30
NAMESPACE="milvus-production"

usage() {
    echo "Usage: $0 -t TEST_NAME [-p PERCENTAGE] [-d DURATION] [-h]"
    echo "  -t TEST_NAME    Name of the A/B test"
    echo "  -p PERCENTAGE   Traffic percentage for treatment (default: 50)"
    echo "  -d DURATION     Test duration in minutes (default: 30)"
    echo "  -h              Show this help"
    exit 1
}

while getopts "t:p:d:h" opt; do
    case $opt in
        t) TEST_NAME="$OPTARG" ;;
        p) TREATMENT_PERCENTAGE="$OPTARG" ;;
        d) DURATION_MINUTES="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$TEST_NAME" ]; then
    echo -e "${RED}Error: Test name is required${NC}"
    usage
fi

echo -e "${GREEN}🧪 Starting A/B Test: $TEST_NAME${NC}"
echo -e "Treatment Traffic: ${YELLOW}$TREATMENT_PERCENTAGE%${NC}"
echo -e "Duration: ${YELLOW}$DURATION_MINUTES minutes${NC}"

# Update traffic splitting
CONTROL_PERCENTAGE=$((100 - TREATMENT_PERCENTAGE))

kubectl patch virtualservice milvus-ab-test -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "http": [{
      "route": [{
        "destination": {"host": "milvus-control"},
        "weight": '$CONTROL_PERCENTAGE'
      }, {
        "destination": {"host": "milvus-treatment"},
        "weight": '$TREATMENT_PERCENTAGE'
      }]
    }]
  }
}'

echo -e "${GREEN}✅ A/B test started successfully${NC}"
echo -e "Monitor progress: ./ab-test-monitor.sh -t $TEST_NAME"
'''
        
        # A/B 테스트 모니터링 스크립트
        monitor_script = '''#!/bin/bash

GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

TEST_NAME=""
NAMESPACE="milvus-production"

usage() {
    echo "Usage: $0 -t TEST_NAME [-h]"
    echo "  -t TEST_NAME    Name of the A/B test to monitor"
    echo "  -h              Show this help"
    exit 1
}

while getopts "t:h" opt; do
    case $opt in
        t) TEST_NAME="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$TEST_NAME" ]; then
    echo -e "${RED}Error: Test name is required${NC}"
    usage
fi

echo -e "${GREEN}📊 Monitoring A/B Test: $TEST_NAME${NC}"
echo -e "${BLUE}Ctrl+C to stop monitoring${NC}"
echo ""

while true; do
    echo -e "${YELLOW}$(date): Checking metrics...${NC}"
    
    # Get pod metrics
    echo "Control Group (A):"
    kubectl top pods -n $NAMESPACE -l variant=control | head -5
    
    echo ""
    echo "Treatment Group (B):"
    kubectl top pods -n $NAMESPACE -l variant=treatment | head -5
    
    echo ""
    echo "Traffic Distribution:"
    kubectl get virtualservice milvus-ab-test -n $NAMESPACE -o jsonpath='{.spec.http[0].route[*].weight}' | tr ' ' '\\n' | paste -d: <(echo -e "Control\\nTreatment") -
    
    echo ""
    echo "----------------------------------------"
    sleep 30
done
'''
        
        # A/B 테스트 종료 스크립트
        stop_script = '''#!/bin/bash
set -e

GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

TEST_NAME=""
WINNER=""
NAMESPACE="milvus-production"

usage() {
    echo "Usage: $0 -t TEST_NAME -w WINNER [-h]"
    echo "  -t TEST_NAME    Name of the A/B test"
    echo "  -w WINNER       Winner variant (control|treatment)"
    echo "  -h              Show this help"
    exit 1
}

while getopts "t:w:h" opt; do
    case $opt in
        t) TEST_NAME="$OPTARG" ;;
        w) WINNER="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$TEST_NAME" ] || [ -z "$WINNER" ]; then
    echo -e "${RED}Error: Test name and winner are required${NC}"
    usage
fi

echo -e "${GREEN}🏁 Stopping A/B Test: $TEST_NAME${NC}"
echo -e "Winner: ${YELLOW}$WINNER${NC}"

if [ "$WINNER" = "control" ]; then
    # Route all traffic to control
    kubectl patch virtualservice milvus-ab-test -n $NAMESPACE --type='merge' -p='{
      "spec": {
        "http": [{
          "route": [{
            "destination": {"host": "milvus-control"},
            "weight": 100
          }]
        }]
      }
    }'
    echo -e "${GREEN}✅ All traffic routed to control group${NC}"
elif [ "$WINNER" = "treatment" ]; then
    # Route all traffic to treatment
    kubectl patch virtualservice milvus-ab-test -n $NAMESPACE --type='merge' -p='{
      "spec": {
        "http": [{
          "route": [{
            "destination": {"host": "milvus-treatment"},
            "weight": 100
          }]
        }]
      }
    }'
    echo -e "${GREEN}✅ All traffic routed to treatment group${NC}"
else
    echo -e "${RED}Error: Winner must be 'control' or 'treatment'${NC}"
    exit 1
fi

echo -e "${GREEN}A/B test completed successfully${NC}"
'''
        
        # 스크립트 저장
        scripts = [
            ('ab-test-start.sh', start_script),
            ('ab-test-monitor.sh', monitor_script),
            ('ab-test-stop.sh', stop_script)
        ]
        
        for filename, content in scripts:
            with open(scripts_dir / filename, 'w') as f:
                f.write(content)
        
        print("  ✅ A/B 테스트 스크립트 생성 완료")
    
    def demonstrate_ab_testing(self):
        """A/B 테스팅 시연"""
        print("\n🎭 A/B 테스팅 시연 시작...")
        
        # 여러 테스트 시나리오 실행
        test_scenarios = [
            "search_algorithm_comparison",
            "memory_optimization", 
            "new_feature_rollout"
        ]
        
        all_results = {}
        
        for scenario in test_scenarios:
            print(f"\n{'='*60}")
            self.run_ab_test(scenario)
            results = self.analyze_test_results(scenario)
            self.save_test_results(scenario, results)
            all_results[scenario] = results
            
            print(f"\n💡 권장사항: {results.get('recommendation', '데이터 부족')}")
        
        # 종합 요약
        print(f"\n{'='*80}")
        print(" 🎯 A/B 테스트 종합 요약")
        print("="*80)
        
        for scenario, results in all_results.items():
            config = self.tests[scenario]
            success = results.get('success_criteria_met', False)
            
            print(f"\n📊 {config.name}:")
            print(f"  결과: {'✅ 성공' if success else '❌ 실패'}")
            
            # 주요 개선사항 표시
            significant_improvements = []
            for metric, data in results.items():
                if isinstance(data, dict) and data.get('is_significant') and data.get('improvement_percent', 0) > 5:
                    improvement = data.get('improvement_percent', 0)
                    significant_improvements.append(f"{metric}: {improvement:+.1f}%")
            
            if significant_improvements:
                print(f"  주요 개선: {', '.join(significant_improvements[:3])}")
            else:
                print(f"  주요 개선: 없음")

def main():
    """메인 실행 함수"""
    print("📊 Milvus A/B 테스팅 시스템")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = ABTestingManager()
    
    try:
        # 1. A/B 테스트 환경 설정
        print("\n" + "=" * 80)
        print(" 🧪 A/B 테스트 환경 설정")
        print("=" * 80)
        manager.setup_ab_test_environment()
        
        # 2. 스크립트 생성
        print("\n" + "=" * 80)
        print(" 📜 자동화 스크립트 생성")
        print("=" * 80)
        manager.create_ab_test_scripts()
        
        # 3. A/B 테스팅 시연
        print("\n" + "=" * 80)
        print(" 🎭 A/B 테스팅 시연")
        print("=" * 80)
        manager.demonstrate_ab_testing()
        
        # 4. 요약
        print("\n" + "=" * 80)
        print(" 📊 A/B 테스팅 완료")
        print("=" * 80)
        
        print("✅ 생성된 리소스:")
        resources = [
            "ab-test-manifests/virtual-service.yaml",
            "ab-test-manifests/feature-flags-config.yaml",
            "ab-test-results/grafana-dashboard.json",
            "ab-test-scripts/ab-test-start.sh",
            "ab-test-scripts/ab-test-monitor.sh", 
            "ab-test-scripts/ab-test-stop.sh"
        ]
        
        for resource in resources:
            print(f"  📄 {resource}")
        
        # 결과 파일들
        result_files = list(manager.results_dir.glob("*.json"))
        if result_files:
            print(f"\n📋 테스트 결과 파일:")
            for file in result_files:
                print(f"  📄 {file.name}")
        
        print("\n💡 A/B 테스팅 모범 사례:")
        best_practices = [
            "✅ 충분한 샘플 크기 확보",
            "✅ 통계적 유의성 확인",
            "✅ 다양한 메트릭 종합 분석",
            "✅ 장기간 모니터링", 
            "✅ 점진적 트래픽 증가",
            "✅ 즉시 롤백 가능한 체계"
        ]
        
        for practice in best_practices:
            print(f"  {practice}")
        
        print("\n🚀 A/B 테스트 명령어 예시:")
        commands = [
            "# A/B 테스트 시작",
            "./ab-test-scripts/ab-test-start.sh -t search_optimization -p 30 -d 60",
            "",
            "# 실시간 모니터링",
            "./ab-test-scripts/ab-test-monitor.sh -t search_optimization",
            "",
            "# 테스트 종료 (승자 선택)",
            "./ab-test-scripts/ab-test-stop.sh -t search_optimization -w treatment"
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
    
    print("\n🎉 A/B 테스팅 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • A/B 테스트 설계 및 실행")
    print("  • 통계적 유의성 검증")
    print("  • 성능 메트릭 비교 분석")
    print("  • 점진적 트래픽 분산")
    
    print("\n🚀 다음 단계:")
    print("  python step05_production/05_security_auth.py")

if __name__ == "__main__":
    main() 