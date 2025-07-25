#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus 프로덕션 모니터링 시스템

이 스크립트는 Milvus 프로덕션 환경의 종합적인 모니터링 시스템을 구현합니다.
SLA 추적, 성능 메트릭, 알림 시스템, 용량 계획, 장애 대응 등 
프로덕션 운영에 필요한 모든 모니터링 기능을 다룹니다.
"""

import os
import sys
import time
import json
import yaml
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

# 공통 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AlertSeverity(Enum):
    """알림 심각도"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class SLA:
    """Service Level Agreement"""
    name: str
    target_percentage: float
    measurement_window: int  # minutes
    metric_name: str
    threshold: float
    comparison: str  # "greater_than", "less_than"

@dataclass
class Alert:
    """알림 정보"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    labels: Dict[str, str]
    resolved: bool = False

class ProductionMonitoringManager:
    """프로덕션 모니터링 관리자"""
    
    def __init__(self):
        self.namespace = "milvus-production"
        self.monitoring_dir = Path("monitoring-configs")
        self.dashboards_dir = self.monitoring_dir / "dashboards"
        self.alerts_dir = self.monitoring_dir / "alerts"
        
        # 디렉토리 생성
        for directory in [self.monitoring_dir, self.dashboards_dir, self.alerts_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # SLA 정의
        self.slas = [
            SLA("availability", 99.9, 60, "up", 1, "equal"),
            SLA("response_time", 95.0, 60, "response_time_ms", 200, "less_than"),
            SLA("error_rate", 99.5, 60, "error_rate", 0.005, "less_than"),
            SLA("throughput", 90.0, 60, "requests_per_second", 1000, "greater_than")
        ]
        
        # 모니터링 데이터
        self.metrics_data: Dict[str, List] = {}
        self.active_alerts: List[Alert] = []
        self.sla_history: Dict[str, List] = {}
        
        # 모니터링 상태
        self.monitoring_active = False
    
    def create_prometheus_config(self):
        """Prometheus 설정 생성"""
        print("📊 Prometheus 모니터링 설정 중...")
        
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s',
                'external_labels': {
                    'cluster': 'milvus-production',
                    'region': 'us-west-2'
                }
            },
            'rule_files': [
                'alerts/*.yml'
            ],
            'alerting': {
                'alertmanagers': [
                    {
                        'static_configs': [
                            {
                                'targets': ['alertmanager:9093']
                            }
                        ]
                    }
                ]
            },
            'scrape_configs': [
                {
                    'job_name': 'milvus-production',
                    'static_configs': [
                        {
                            'targets': ['milvus:9091']
                        }
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s',
                    'scrape_timeout': '10s'
                },
                {
                    'job_name': 'milvus-etcd',
                    'static_configs': [
                        {
                            'targets': ['etcd:2379']
                        }
                    ],
                    'metrics_path': '/metrics'
                },
                {
                    'job_name': 'milvus-minio',
                    'static_configs': [
                        {
                            'targets': ['minio:9000']
                        }
                    ],
                    'metrics_path': '/minio/v2/metrics/cluster'
                },
                {
                    'job_name': 'kubernetes-nodes',
                    'kubernetes_sd_configs': [
                        {
                            'role': 'node'
                        }
                    ],
                    'relabel_configs': [
                        {
                            'source_labels': ['__address__'],
                            'regex': '(.*):10250',
                            'target_label': '__address__',
                            'replacement': '${1}:9100'
                        }
                    ]
                },
                {
                    'job_name': 'kubernetes-pods',
                    'kubernetes_sd_configs': [
                        {
                            'role': 'pod',
                            'namespaces': {
                                'names': ['milvus-production']
                            }
                        }
                    ],
                    'relabel_configs': [
                        {
                            'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_scrape'],
                            'action': 'keep',
                            'regex': True
                        }
                    ]
                }
            ]
        }
        
        with open(self.monitoring_dir / 'prometheus.yml', 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        print("  ✅ Prometheus 설정 생성됨")
    
    def create_alert_rules(self):
        """알림 규칙 생성"""
        print("🚨 알림 규칙 설정 중...")
        
        # Milvus 서비스 알림
        milvus_alerts = {
            'groups': [
                {
                    'name': 'milvus.rules',
                    'rules': [
                        {
                            'alert': 'MilvusDown',
                            'expr': 'up{job="milvus-production"} == 0',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical',
                                'service': 'milvus'
                            },
                            'annotations': {
                                'summary': 'Milvus instance is down',
                                'description': 'Milvus instance {{ $labels.instance }} has been down for more than 1 minute.'
                            }
                        },
                        {
                            'alert': 'MilvusHighResponseTime',
                            'expr': 'histogram_quantile(0.95, milvus_request_duration_seconds_bucket) > 0.2',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning',
                                'service': 'milvus'
                            },
                            'annotations': {
                                'summary': 'Milvus high response time',
                                'description': '95th percentile response time is {{ $value }}s for the last 5 minutes.'
                            }
                        },
                        {
                            'alert': 'MilvusHighErrorRate',
                            'expr': 'rate(milvus_errors_total[5m]) / rate(milvus_requests_total[5m]) > 0.01',
                            'for': '3m',
                            'labels': {
                                'severity': 'critical',
                                'service': 'milvus'
                            },
                            'annotations': {
                                'summary': 'Milvus high error rate',
                                'description': 'Error rate is {{ $value | humanizePercentage }} for the last 5 minutes.'
                            }
                        },
                        {
                            'alert': 'MilvusHighMemoryUsage',
                            'expr': 'milvus_memory_usage_bytes / milvus_memory_limit_bytes > 0.85',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning',
                                'service': 'milvus'
                            },
                            'annotations': {
                                'summary': 'Milvus high memory usage',
                                'description': 'Memory usage is {{ $value | humanizePercentage }} for instance {{ $labels.instance }}.'
                            }
                        },
                        {
                            'alert': 'MilvusLowThroughput',
                            'expr': 'rate(milvus_requests_total[5m]) < 100',
                            'for': '15m',
                            'labels': {
                                'severity': 'warning',
                                'service': 'milvus'
                            },
                            'annotations': {
                                'summary': 'Milvus low throughput',
                                'description': 'Request rate is {{ $value }} requests/second, which is below expected threshold.'
                            }
                        }
                    ]
                },
                {
                    'name': 'infrastructure.rules',
                    'rules': [
                        {
                            'alert': 'HighCPUUsage',
                            'expr': '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[2m])) * 100) > 80',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High CPU usage detected',
                                'description': 'CPU usage is {{ $value }}% on {{ $labels.instance }}.'
                            }
                        },
                        {
                            'alert': 'HighMemoryUsage',
                            'expr': '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High memory usage detected',
                                'description': 'Memory usage is {{ $value }}% on {{ $labels.instance }}.'
                            }
                        },
                        {
                            'alert': 'DiskSpaceLow',
                            'expr': '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 85',
                            'for': '5m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Disk space running low',
                                'description': 'Disk usage is {{ $value }}% on {{ $labels.instance }}:{{ $labels.mountpoint }}.'
                            }
                        }
                    ]
                }
            ]
        }
        
        with open(self.alerts_dir / 'milvus-alerts.yml', 'w') as f:
            yaml.dump(milvus_alerts, f, default_flow_style=False)
        
        # SLA 기반 알림
        sla_alerts = {
            'groups': [
                {
                    'name': 'sla.rules',
                    'rules': [
                        {
                            'alert': 'SLAAvailabilityBreach',
                            'expr': 'avg_over_time(up{job="milvus-production"}[1h]) < 0.999',
                            'for': '0m',
                            'labels': {
                                'severity': 'critical',
                                'sla': 'availability'
                            },
                            'annotations': {
                                'summary': 'SLA availability breach detected',
                                'description': 'Availability SLA (99.9%) has been breached. Current: {{ $value | humanizePercentage }}'
                            }
                        },
                        {
                            'alert': 'SLAResponseTimeBreach',
                            'expr': 'histogram_quantile(0.95, avg_over_time(milvus_request_duration_seconds_bucket[1h])) > 0.2',
                            'for': '0m',
                            'labels': {
                                'severity': 'critical',
                                'sla': 'response_time'
                            },
                            'annotations': {
                                'summary': 'SLA response time breach detected',
                                'description': '95th percentile response time SLA (200ms) has been breached. Current: {{ $value }}s'
                            }
                        }
                    ]
                }
            ]
        }
        
        with open(self.alerts_dir / 'sla-alerts.yml', 'w') as f:
            yaml.dump(sla_alerts, f, default_flow_style=False)
        
        print("  ✅ 알림 규칙 생성됨")
    
    def create_grafana_dashboards(self):
        """Grafana 대시보드 생성"""
        print("📈 Grafana 대시보드 설정 중...")
        
        # 메인 운영 대시보드
        main_dashboard = {
            'dashboard': {
                'title': 'Milvus Production Overview',
                'tags': ['milvus', 'production'],
                'timezone': 'browser',
                'refresh': '30s',
                'time': {
                    'from': 'now-1h',
                    'to': 'now'
                },
                'panels': [
                    {
                        'id': 1,
                        'title': 'Service Status',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'up{job="milvus-production"}',
                                'legendFormat': 'Milvus Status'
                            }
                        ],
                        'fieldConfig': {
                            'defaults': {
                                'color': {
                                    'mode': 'thresholds'
                                },
                                'thresholds': {
                                    'steps': [
                                        {'color': 'red', 'value': 0},
                                        {'color': 'green', 'value': 1}
                                    ]
                                }
                            }
                        },
                        'gridPos': {'h': 4, 'w': 6, 'x': 0, 'y': 0}
                    },
                    {
                        'id': 2,
                        'title': 'Request Rate (QPS)',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(milvus_requests_total[5m])',
                                'legendFormat': 'Requests/sec'
                            }
                        ],
                        'yAxes': [
                            {'label': 'Requests/sec', 'min': 0}
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 6, 'y': 0}
                    },
                    {
                        'id': 3,
                        'title': 'Response Time (95th percentile)',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, milvus_request_duration_seconds_bucket)',
                                'legendFormat': '95th percentile'
                            },
                            {
                                'expr': 'histogram_quantile(0.50, milvus_request_duration_seconds_bucket)', 
                                'legendFormat': '50th percentile'
                            }
                        ],
                        'yAxes': [
                            {'label': 'Seconds', 'min': 0}
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                    },
                    {
                        'id': 4,
                        'title': 'Error Rate',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(milvus_errors_total[5m]) / rate(milvus_requests_total[5m])',
                                'legendFormat': 'Error Rate'
                            }
                        ],
                        'yAxes': [
                            {'label': 'Percentage', 'min': 0, 'max': 1}
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                    },
                    {
                        'id': 5,
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'milvus_memory_usage_bytes / 1024 / 1024 / 1024',
                                'legendFormat': 'Memory Usage (GB)'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}
                    },
                    {
                        'id': 6,
                        'title': 'Active Collections',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'milvus_collections_total',
                                'legendFormat': 'Collections'
                            }
                        ],
                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}
                    }
                ]
            }
        }
        
        with open(self.dashboards_dir / 'main-dashboard.json', 'w') as f:
            json.dump(main_dashboard, f, indent=2)
        
        # SLA 대시보드
        sla_dashboard = {
            'dashboard': {
                'title': 'Milvus SLA Monitoring',
                'tags': ['milvus', 'sla'],
                'panels': [
                    {
                        'id': 1,
                        'title': 'Availability SLA (99.9%)',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'avg_over_time(up{job="milvus-production"}[24h]) * 100',
                                'legendFormat': 'Availability %'
                            }
                        ],
                        'thresholds': [
                            {'color': 'red', 'value': 0},
                            {'color': 'yellow', 'value': 99.0},
                            {'color': 'green', 'value': 99.9}
                        ]
                    },
                    {
                        'id': 2,
                        'title': 'Response Time SLA (95% < 200ms)',
                        'type': 'stat',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, avg_over_time(milvus_request_duration_seconds_bucket[24h])) * 1000',
                                'legendFormat': '95th percentile (ms)'
                            }
                        ],
                        'thresholds': [
                            {'color': 'green', 'value': 0},
                            {'color': 'yellow', 'value': 200},
                            {'color': 'red', 'value': 500}
                        ]
                    }
                ]
            }
        }
        
        with open(self.dashboards_dir / 'sla-dashboard.json', 'w') as f:
            json.dump(sla_dashboard, f, indent=2)
        
        # 인프라 대시보드
        infra_dashboard = {
            'dashboard': {
                'title': 'Milvus Infrastructure Monitoring',
                'tags': ['milvus', 'infrastructure'],
                'panels': [
                    {
                        'id': 1,
                        'title': 'CPU Usage',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[2m])) * 100)',
                                'legendFormat': 'CPU Usage %'
                            }
                        ]
                    },
                    {
                        'id': 2,
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
                                'legendFormat': 'Memory Usage %'
                            }
                        ]
                    },
                    {
                        'id': 3,
                        'title': 'Network I/O',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(node_network_receive_bytes_total[5m])',
                                'legendFormat': 'Receive'
                            },
                            {
                                'expr': 'rate(node_network_transmit_bytes_total[5m])',
                                'legendFormat': 'Transmit'
                            }
                        ]
                    }
                ]
            }
        }
        
        with open(self.dashboards_dir / 'infrastructure-dashboard.json', 'w') as f:
            json.dump(infra_dashboard, f, indent=2)
        
        print("  ✅ Grafana 대시보드 생성됨")
    
    def create_alertmanager_config(self):
        """Alertmanager 설정 생성"""
        print("📢 Alertmanager 알림 설정 중...")
        
        alertmanager_config = {
            'global': {
                'smtp_smarthost': 'smtp.example.com:587',
                'smtp_from': 'alerts@example.com'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'web.hook',
                'routes': [
                    {
                        'match': {
                            'severity': 'critical'
                        },
                        'receiver': 'critical-alerts',
                        'repeat_interval': '5m'
                    },
                    {
                        'match': {
                            'severity': 'warning'
                        },
                        'receiver': 'warning-alerts',
                        'repeat_interval': '30m'
                    }
                ]
            },
            'receivers': [
                {
                    'name': 'web.hook',
                    'webhook_configs': [
                        {
                            'url': 'http://webhook.example.com/alerts',
                            'send_resolved': True
                        }
                    ]
                },
                {
                    'name': 'critical-alerts',
                    'email_configs': [
                        {
                            'to': 'oncall@example.com',
                            'subject': '[CRITICAL] Milvus Alert: {{ .GroupLabels.alertname }}',
                            'body': '''
Alert: {{ .GroupLabels.alertname }}
Severity: {{ .CommonLabels.severity }}
Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
Instance: {{ .CommonLabels.instance }}
Time: {{ .CommonAnnotations.timestamp }}
'''
                        }
                    ],
                    'slack_configs': [
                        {
                            'api_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                            'channel': '#alerts-critical',
                            'title': 'Critical Alert: {{ .GroupLabels.alertname }}',
                            'text': '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
                        }
                    ],
                    'pagerduty_configs': [
                        {
                            'service_key': 'YOUR_PAGERDUTY_KEY',
                            'description': '{{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
                        }
                    ]
                },
                {
                    'name': 'warning-alerts',
                    'email_configs': [
                        {
                            'to': 'team@example.com',
                            'subject': '[WARNING] Milvus Alert: {{ .GroupLabels.alertname }}',
                            'body': '''
Alert: {{ .GroupLabels.alertname }}
Severity: {{ .CommonLabels.severity }}
Description: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
'''
                        }
                    ]
                }
            ]
        }
        
        with open(self.monitoring_dir / 'alertmanager.yml', 'w') as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False)
        
        print("  ✅ Alertmanager 설정 생성됨")
    
    def simulate_metrics_collection(self):
        """메트릭 수집 시뮬레이션"""
        print("\n📊 메트릭 수집 시뮬레이션 시작...")
        
        # 시뮬레이션된 메트릭 데이터 생성
        metrics = {
            'response_time_ms': [],
            'requests_per_second': [],
            'error_rate': [],
            'memory_usage_gb': [],
            'cpu_usage_percent': [],
            'availability': []
        }
        
        # 24시간 데이터 시뮬레이션 (5분 간격)
        time_points = 24 * 12  # 288 data points
        
        for i in range(time_points):
            # 시간대별 패턴 적용
            hour = (i * 5) // 60
            
            # 업무 시간(9-18시)에 트래픽 증가
            if 9 <= hour <= 18:
                traffic_multiplier = 1.5 + random.uniform(-0.2, 0.3)
            else:
                traffic_multiplier = 0.7 + random.uniform(-0.1, 0.2)
            
            # 기본 메트릭 값들
            base_response_time = 80
            base_qps = 1500
            base_error_rate = 0.002
            base_memory = 8.5
            base_cpu = 45
            
            # 트래픽에 따른 성능 영향
            response_time = base_response_time * (1 + (traffic_multiplier - 1) * 0.3)
            qps = base_qps * traffic_multiplier
            error_rate = base_error_rate * (1 + (traffic_multiplier - 1) * 0.5)
            memory_usage = base_memory * (1 + (traffic_multiplier - 1) * 0.2)
            cpu_usage = base_cpu * traffic_multiplier
            
            # 랜덤 노이즈 추가
            metrics['response_time_ms'].append(max(20, response_time + random.gauss(0, 15)))
            metrics['requests_per_second'].append(max(100, qps + random.gauss(0, 200)))
            metrics['error_rate'].append(max(0, error_rate + random.gauss(0, 0.001)))
            metrics['memory_usage_gb'].append(max(4, memory_usage + random.gauss(0, 0.5)))
            metrics['cpu_usage_percent'].append(max(10, min(100, cpu_usage + random.gauss(0, 10))))
            
            # 가용성 (99.95% 정도)
            availability = 1.0 if random.random() > 0.0005 else 0.0
            metrics['availability'].append(availability)
        
        self.metrics_data = metrics
        
        # 현재 상태 요약
        current_metrics = {
            'response_time_ms': metrics['response_time_ms'][-1],
            'requests_per_second': metrics['requests_per_second'][-1],
            'error_rate': metrics['error_rate'][-1],
            'memory_usage_gb': metrics['memory_usage_gb'][-1],
            'cpu_usage_percent': metrics['cpu_usage_percent'][-1],
            'availability': metrics['availability'][-1]
        }
        
        print(f"  📈 현재 메트릭 상태:")
        print(f"    응답시간: {current_metrics['response_time_ms']:.1f}ms")
        print(f"    QPS: {current_metrics['requests_per_second']:.0f}")
        print(f"    오류율: {current_metrics['error_rate']:.3f}%")
        print(f"    메모리: {current_metrics['memory_usage_gb']:.1f}GB")
        print(f"    CPU: {current_metrics['cpu_usage_percent']:.1f}%")
        print(f"    가용성: {'✅ UP' if current_metrics['availability'] else '❌ DOWN'}")
        
        return current_metrics
    
    def check_sla_compliance(self) -> Dict[str, Dict]:
        """SLA 준수 여부 확인"""
        print("\n🎯 SLA 준수 상태 확인 중...")
        
        sla_results = {}
        
        for sla in self.slas:
            if sla.metric_name == "response_time_ms":
                data = self.metrics_data['response_time_ms']
                # 95th percentile 계산
                value = statistics.quantiles(data, n=20)[18]  # 95th percentile
                target_met = value <= sla.threshold
                
            elif sla.metric_name == "error_rate":
                data = self.metrics_data['error_rate']
                value = statistics.mean(data)
                target_met = value <= sla.threshold
                
            elif sla.metric_name == "requests_per_second":
                data = self.metrics_data['requests_per_second']
                value = statistics.mean(data)
                target_met = value >= sla.threshold
                
            elif sla.metric_name == "up":
                data = self.metrics_data['availability']
                value = statistics.mean(data) * 100  # 백분율로 변환
                target_met = value >= sla.target_percentage
            
            else:
                continue
            
            sla_results[sla.name] = {
                'current_value': value,
                'target': sla.threshold if sla.name != 'availability' else sla.target_percentage,
                'target_met': target_met,
                'sla_percentage': sla.target_percentage
            }
            
            status = "✅ 준수" if target_met else "❌ 위반"
            print(f"  {sla.name.upper()}: {status}")
            print(f"    현재값: {value:.2f}")
            print(f"    목표값: {sla.threshold if sla.name != 'availability' else sla.target_percentage}")
            print(f"    SLA: {sla.target_percentage}%")
        
        return sla_results
    
    def generate_alerts(self, current_metrics: Dict[str, float]) -> List[Alert]:
        """알림 생성"""
        alerts = []
        
        # 응답시간 체크
        if current_metrics['response_time_ms'] > 200:
            severity = AlertSeverity.CRITICAL if current_metrics['response_time_ms'] > 500 else AlertSeverity.WARNING
            alerts.append(Alert(
                name="HighResponseTime",
                severity=severity,
                message=f"Response time is {current_metrics['response_time_ms']:.1f}ms",
                timestamp=datetime.now(),
                labels={'service': 'milvus', 'metric': 'response_time'}
            ))
        
        # 오류율 체크
        if current_metrics['error_rate'] > 0.01:
            alerts.append(Alert(
                name="HighErrorRate",
                severity=AlertSeverity.CRITICAL,
                message=f"Error rate is {current_metrics['error_rate']:.3f}%",
                timestamp=datetime.now(),
                labels={'service': 'milvus', 'metric': 'error_rate'}
            ))
        
        # 메모리 사용량 체크
        if current_metrics['memory_usage_gb'] > 12:
            alerts.append(Alert(
                name="HighMemoryUsage",
                severity=AlertSeverity.WARNING,
                message=f"Memory usage is {current_metrics['memory_usage_gb']:.1f}GB",
                timestamp=datetime.now(),
                labels={'service': 'milvus', 'metric': 'memory'}
            ))
        
        # CPU 사용량 체크
        if current_metrics['cpu_usage_percent'] > 80:
            alerts.append(Alert(
                name="HighCPUUsage",
                severity=AlertSeverity.WARNING,
                message=f"CPU usage is {current_metrics['cpu_usage_percent']:.1f}%",
                timestamp=datetime.now(),
                labels={'service': 'milvus', 'metric': 'cpu'}
            ))
        
        # 서비스 다운 체크
        if current_metrics['availability'] == 0:
            alerts.append(Alert(
                name="ServiceDown",
                severity=AlertSeverity.CRITICAL,
                message="Milvus service is not responding",
                timestamp=datetime.now(),
                labels={'service': 'milvus', 'metric': 'availability'}
            ))
        
        return alerts
    
    def capacity_planning_analysis(self):
        """용량 계획 분석"""
        print("\n📊 용량 계획 분석 중...")
        
        # 현재 사용량 분석
        current_usage = {
            'memory_avg': statistics.mean(self.metrics_data['memory_usage_gb']),
            'cpu_avg': statistics.mean(self.metrics_data['cpu_usage_percent']),
            'qps_avg': statistics.mean(self.metrics_data['requests_per_second']),
            'qps_max': max(self.metrics_data['requests_per_second'])
        }
        
        # 성장률 계산 (시뮬레이션)
        growth_rate = 0.15  # 월 15% 성장 가정
        
        # 6개월 후 예상 사용량
        months_ahead = 6
        projected_usage = {
            'memory_gb': current_usage['memory_avg'] * (1 + growth_rate) ** months_ahead,
            'cpu_percent': current_usage['cpu_avg'] * (1 + growth_rate) ** months_ahead,
            'qps': current_usage['qps_avg'] * (1 + growth_rate) ** months_ahead,
            'qps_peak': current_usage['qps_max'] * (1 + growth_rate) ** months_ahead
        }
        
        # 용량 제한 (현재 설정)
        capacity_limits = {
            'memory_gb': 16,
            'cpu_percent': 100,
            'qps_capacity': 5000
        }
        
        # 용량 부족 시점 예측
        capacity_warnings = []
        
        if projected_usage['memory_gb'] > capacity_limits['memory_gb'] * 0.8:
            capacity_warnings.append(f"메모리: {months_ahead}개월 후 {projected_usage['memory_gb']:.1f}GB 필요 (현재 한계: {capacity_limits['memory_gb']}GB)")
        
        if projected_usage['cpu_percent'] > 80:
            capacity_warnings.append(f"CPU: {months_ahead}개월 후 {projected_usage['cpu_percent']:.1f}% 사용 예상")
        
        if projected_usage['qps_peak'] > capacity_limits['qps_capacity'] * 0.8:
            capacity_warnings.append(f"처리량: {months_ahead}개월 후 최대 {projected_usage['qps_peak']:.0f} QPS 필요")
        
        print(f"  📈 현재 리소스 사용량:")
        print(f"    메모리: {current_usage['memory_avg']:.1f}GB (평균)")
        print(f"    CPU: {current_usage['cpu_avg']:.1f}% (평균)")
        print(f"    QPS: {current_usage['qps_avg']:.0f} (평균), {current_usage['qps_max']:.0f} (최대)")
        
        print(f"\n  🔮 {months_ahead}개월 후 예상 사용량:")
        print(f"    메모리: {projected_usage['memory_gb']:.1f}GB")
        print(f"    CPU: {projected_usage['cpu_percent']:.1f}%")
        print(f"    QPS: {projected_usage['qps']:.0f} (평균), {projected_usage['qps_peak']:.0f} (최대)")
        
        if capacity_warnings:
            print(f"\n  ⚠️  용량 계획 권고사항:")
            for warning in capacity_warnings:
                print(f"    • {warning}")
        else:
            print(f"\n  ✅ 현재 용량으로 {months_ahead}개월간 충분함")
        
        # 스케일링 권장사항
        scaling_recommendations = []
        
        if projected_usage['memory_gb'] > capacity_limits['memory_gb'] * 0.8:
            scaling_recommendations.append("메모리: 32GB로 업그레이드 권장")
        
        if projected_usage['qps_peak'] > capacity_limits['qps_capacity'] * 0.8:
            scaling_recommendations.append("인스턴스: 현재 1대에서 2-3대로 확장 권장")
        
        if scaling_recommendations:
            print(f"\n  🚀 스케일링 권장사항:")
            for rec in scaling_recommendations:
                print(f"    • {rec}")
        
        return {
            'current_usage': current_usage,
            'projected_usage': projected_usage,
            'capacity_warnings': capacity_warnings,
            'scaling_recommendations': scaling_recommendations
        }
    
    def create_monitoring_scripts(self):
        """모니터링 운영 스크립트 생성"""
        print("📜 모니터링 운영 스크립트 생성 중...")
        
        scripts_dir = Path("monitoring-scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        # 헬스체크 스크립트
        health_check_script = '''#!/bin/bash
set -e

NAMESPACE="milvus-production"
SERVICE_NAME="milvus-main"

echo "🏥 Milvus Health Check"
echo "===================="

# 1. Pod 상태 확인
echo "📦 Pod Status:"
kubectl get pods -n $NAMESPACE -l app=milvus -o wide

# 2. 서비스 상태 확인
echo ""
echo "🌐 Service Status:"
kubectl get services -n $NAMESPACE

# 3. 엔드포인트 확인
echo ""
echo "🔗 Endpoints:"
kubectl get endpoints -n $NAMESPACE

# 4. 헬스체크 API 호출
echo ""
echo "🩺 Health Check API:"
SERVICE_IP=$(kubectl get svc $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
if curl -s -f http://$SERVICE_IP:9091/healthz > /dev/null; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
fi

# 5. 메트릭 확인
echo ""
echo "📊 Metrics endpoint:"
if curl -s -f http://$SERVICE_IP:9091/metrics | head -5; then
    echo "✅ Metrics endpoint accessible"
else
    echo "❌ Metrics endpoint not accessible"
fi

echo ""
echo "✅ Health check completed"
'''
        
        # 성능 모니터링 스크립트
        performance_script = '''#!/bin/bash

NAMESPACE="milvus-production"
DURATION=${1:-300}  # 기본 5분

echo "📊 Milvus Performance Monitor"
echo "Duration: ${DURATION} seconds"
echo "=========================="

# 1. 리소스 사용량 모니터링
echo "💾 Resource Usage:"
while true; do
    echo "$(date): Checking resource usage..."
    kubectl top pods -n $NAMESPACE -l app=milvus
    echo "---"
    sleep 30
    DURATION=$((DURATION - 30))
    if [ $DURATION -le 0 ]; then
        break
    fi
done

echo "✅ Performance monitoring completed"
'''
        
        # 로그 수집 스크립트
        log_collection_script = '''#!/bin/bash

NAMESPACE="milvus-production"
OUTPUT_DIR="logs/$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

echo "📋 Collecting Milvus logs..."
echo "Output directory: $OUTPUT_DIR"

# 1. Pod logs
echo "Collecting pod logs..."
for pod in $(kubectl get pods -n $NAMESPACE -l app=milvus -o jsonpath='{.items[*].metadata.name}'); do
    echo "  Collecting logs from $pod..."
    kubectl logs $pod -n $NAMESPACE --previous > $OUTPUT_DIR/${pod}_previous.log 2>/dev/null || true
    kubectl logs $pod -n $NAMESPACE > $OUTPUT_DIR/${pod}_current.log
done

# 2. Events
echo "Collecting events..."
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' > $OUTPUT_DIR/events.log

# 3. Describe pods
echo "Collecting pod descriptions..."
kubectl describe pods -n $NAMESPACE -l app=milvus > $OUTPUT_DIR/pod_descriptions.log

echo "✅ Log collection completed"
echo "📁 Logs saved to: $OUTPUT_DIR"
'''
        
        # 알림 테스트 스크립트
        alert_test_script = '''#!/bin/bash

echo "🚨 Testing Milvus Alerting System"
echo "================================"

ALERTMANAGER_URL="http://alertmanager:9093"

# 테스트 알림 생성
TEST_ALERT='{
  "alerts": [
    {
      "labels": {
        "alertname": "TestAlert",
        "severity": "warning",
        "service": "milvus",
        "instance": "test-instance"
      },
      "annotations": {
        "summary": "This is a test alert",
        "description": "Testing the alerting system functionality"
      },
      "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
      "endsAt": "'$(date -u -d '+5 minutes' +%Y-%m-%dT%H:%M:%S.%3NZ)'"
    }
  ]
}'

echo "📤 Sending test alert to Alertmanager..."
curl -X POST $ALERTMANAGER_URL/api/v1/alerts \\
     -H "Content-Type: application/json" \\
     -d "$TEST_ALERT"

if [ $? -eq 0 ]; then
    echo "✅ Test alert sent successfully"
    echo "📧 Check your configured notification channels"
else
    echo "❌ Failed to send test alert"
fi
'''
        
        # 스크립트 파일 저장
        scripts = [
            ('health-check.sh', health_check_script),
            ('performance-monitor.sh', performance_script),
            ('collect-logs.sh', log_collection_script),
            ('test-alerts.sh', alert_test_script)
        ]
        
        for filename, content in scripts:
            with open(scripts_dir / filename, 'w') as f:
                f.write(content)
        
        print("  ✅ 모니터링 운영 스크립트 생성됨")
        print("  💫 실행 권한 설정 필요:")
        for filename, _ in scripts:
            print(f"    $ chmod +x monitoring-scripts/{filename}")
    
    def generate_monitoring_report(self, current_metrics: Dict, sla_results: Dict, capacity_analysis: Dict):
        """모니터링 종합 보고서 생성"""
        print("\n📄 종합 모니터링 보고서 생성 중...")
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'reporting_period': '24 hours',
                'system': 'Milvus Production',
                'version': '2.4.0'
            },
            'executive_summary': {
                'overall_health': 'healthy',
                'sla_compliance': all(result['target_met'] for result in sla_results.values()),
                'critical_alerts': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'warning_alerts': len([a for a in self.active_alerts if a.severity == AlertSeverity.WARNING])
            },
            'current_metrics': current_metrics,
            'sla_compliance': sla_results,
            'capacity_planning': capacity_analysis,
            'active_alerts': [
                {
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'labels': alert.labels
                }
                for alert in self.active_alerts
            ],
            'recommendations': []
        }
        
        # 권장사항 생성
        if not report['sla_compliance']:
            report['recommendations'].append("SLA 위반 항목에 대한 즉시 조치 필요")
        
        if report['executive_summary']['critical_alerts'] > 0:
            report['recommendations'].append("긴급 알림에 대한 즉시 대응 필요")
        
        if current_metrics['memory_usage_gb'] > 12:
            report['recommendations'].append("메모리 사용량 최적화 또는 용량 증설 검토")
        
        if capacity_analysis['scaling_recommendations']:
            report['recommendations'].extend(capacity_analysis['scaling_recommendations'])
        
        # 보고서 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.monitoring_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✅ 모니터링 보고서 저장됨: {report_file.name}")
        
        # 요약 출력
        print(f"\n📊 보고서 요약:")
        print(f"  전체 상태: {'✅ 양호' if report['executive_summary']['overall_health'] == 'healthy' else '⚠️ 주의'}")
        print(f"  SLA 준수: {'✅ 준수' if report['executive_summary']['sla_compliance'] else '❌ 위반'}")
        print(f"  긴급 알림: {report['executive_summary']['critical_alerts']}건")
        print(f"  경고 알림: {report['executive_summary']['warning_alerts']}건")
        
        if report['recommendations']:
            print(f"\n💡 권장사항:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        return report

def main():
    """메인 실행 함수"""
    print("📈 Milvus 프로덕션 모니터링 시스템")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = ProductionMonitoringManager()
    
    try:
        # 1. 모니터링 설정 생성
        print("\n" + "=" * 80)
        print(" 📊 모니터링 시스템 구축")
        print("=" * 80)
        
        manager.create_prometheus_config()
        manager.create_alert_rules()
        manager.create_grafana_dashboards()
        manager.create_alertmanager_config()
        
        # 2. 모니터링 스크립트 생성
        print("\n" + "=" * 80)
        print(" 📜 운영 스크립트 생성")
        print("=" * 80)
        manager.create_monitoring_scripts()
        
        # 3. 메트릭 수집 및 분석
        print("\n" + "=" * 80)
        print(" 📊 실시간 모니터링 시뮬레이션")
        print("=" * 80)
        
        current_metrics = manager.simulate_metrics_collection()
        sla_results = manager.check_sla_compliance()
        
        # 4. 알림 생성
        print("\n" + "=" * 80)
        print(" 🚨 알림 시스템 테스트")
        print("=" * 80)
        
        alerts = manager.generate_alerts(current_metrics)
        manager.active_alerts = alerts
        
        if alerts:
            print(f"  🔔 생성된 알림: {len(alerts)}건")
            for alert in alerts:
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity.value, "⚪")
                print(f"    {severity_emoji} [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
        else:
            print(f"  ✅ 활성 알림 없음 - 시스템 정상")
        
        # 5. 용량 계획 분석
        print("\n" + "=" * 80)
        print(" 📊 용량 계획 분석")
        print("=" * 80)
        
        capacity_analysis = manager.capacity_planning_analysis()
        
        # 6. 종합 보고서 생성
        print("\n" + "=" * 80)
        print(" 📄 종합 모니터링 보고서")
        print("=" * 80)
        
        report = manager.generate_monitoring_report(current_metrics, sla_results, capacity_analysis)
        
        # 7. 요약
        print("\n" + "=" * 80)
        print(" 📈 프로덕션 모니터링 완료")
        print("=" * 80)
        
        print("✅ 생성된 모니터링 리소스:")
        monitoring_resources = [
            "monitoring-configs/prometheus.yml",
            "monitoring-configs/alertmanager.yml",
            "monitoring-configs/alerts/milvus-alerts.yml",
            "monitoring-configs/alerts/sla-alerts.yml",
            "monitoring-configs/dashboards/main-dashboard.json",
            "monitoring-configs/dashboards/sla-dashboard.json",
            "monitoring-configs/dashboards/infrastructure-dashboard.json",
            "monitoring-scripts/health-check.sh",
            "monitoring-scripts/performance-monitor.sh",
            "monitoring-scripts/collect-logs.sh",
            "monitoring-scripts/test-alerts.sh"
        ]
        
        for resource in monitoring_resources:
            print(f"  📄 {resource}")
        
        print("\n📊 모니터링 대시보드 URL:")
        dashboard_urls = [
            "http://grafana.example.com/d/milvus-overview (메인 대시보드)",
            "http://grafana.example.com/d/milvus-sla (SLA 모니터링)",
            "http://grafana.example.com/d/milvus-infra (인프라 모니터링)",
            "http://prometheus.example.com (Prometheus)",
            "http://alertmanager.example.com (Alertmanager)"
        ]
        
        for url in dashboard_urls:
            print(f"  🌐 {url}")
        
        print("\n💡 운영 모니터링 체크리스트:")
        checklist = [
            "✅ 24/7 대시보드 모니터링",
            "✅ 알림 채널 설정 및 테스트",
            "✅ SLA 목표 추적 및 보고",
            "✅ 정기적인 용량 계획 검토",
            "✅ 장애 대응 절차 문서화",
            "✅ 백업 및 복구 계획 수립",
            "✅ 성능 기준선 설정 및 업데이트"
        ]
        
        for item in checklist:
            print(f"  {item}")
        
        print("\n🚀 운영 명령어 예시:")
        commands = [
            "# 헬스체크 실행",
            "./monitoring-scripts/health-check.sh",
            "",
            "# 성능 모니터링 (10분간)",
            "./monitoring-scripts/performance-monitor.sh 600",
            "",
            "# 로그 수집",
            "./monitoring-scripts/collect-logs.sh",
            "",
            "# 알림 시스템 테스트",
            "./monitoring-scripts/test-alerts.sh"
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
    
    print("\n🎉 프로덕션 모니터링 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 종합적인 모니터링 시스템 구축")
    print("  • SLA 기반 성능 관리")
    print("  • 실시간 알림 및 장애 대응")
    print("  • 데이터 기반 용량 계획")
    print("  • 프로덕션 운영 자동화")
    
    print("\n🏆 5단계 프로덕션 배포 및 운영 완료!")
    print("    모든 Milvus 학습 과정을 성공적으로 마쳤습니다! 🎆")

if __name__ == "__main__":
    main() 