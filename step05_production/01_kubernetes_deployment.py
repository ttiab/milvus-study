#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus Kubernetes 배포 및 클러스터 관리

이 스크립트는 Kubernetes 환경에서 Milvus 클러스터를 배포하고 관리하는 방법을 보여줍니다.
실제 프로덕션 환경에서 사용할 수 있는 Helm 차트, 서비스 관리, 오토스케일링 등을 다룹니다.
"""

import os
import sys
import time
import yaml
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 공통 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.connection import MilvusConnection
from common.vector_utils import VectorUtils

class KubernetesManager:
    """Kubernetes 클러스터 관리자"""
    
    def __init__(self):
        self.namespace = "milvus-prod"
        self.release_name = "milvus-cluster"
        self.manifests_dir = Path("k8s-manifests")
        self.manifests_dir.mkdir(exist_ok=True)
        
    def check_prerequisites(self) -> bool:
        """전제 조건 확인"""
        print("🔍 Kubernetes 환경 확인 중...")
        
        try:
            # kubectl 설치 확인
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("  ❌ kubectl이 설치되지 않았습니다.")
                return False
            print("  ✅ kubectl 설치됨")
            
            # 클러스터 연결 확인
            result = subprocess.run(['kubectl', 'cluster-info'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("  ⚠️  Kubernetes 클러스터에 연결할 수 없습니다 (시뮬레이션 모드로 진행)")
                return False
            print("  ✅ Kubernetes 클러스터 연결됨")
            
            # Helm 설치 확인
            result = subprocess.run(['helm', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("  ⚠️  Helm이 설치되지 않았습니다 (매니페스트 모드로 진행)")
                return False
            print("  ✅ Helm 설치됨")
            
            return True
            
        except FileNotFoundError:
            print("  ⚠️  Kubernetes 도구가 설치되지 않았습니다 (시뮬레이션 모드로 진행)")
            return False
    
    def create_namespace(self):
        """네임스페이스 생성"""
        print(f"📁 네임스페이스 '{self.namespace}' 생성 중...")
        
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.namespace,
                'labels': {
                    'app': 'milvus',
                    'environment': 'production'
                }
            }
        }
        
        # 매니페스트 파일 저장
        with open(self.manifests_dir / 'namespace.yaml', 'w') as f:
            yaml.dump(namespace_manifest, f, default_flow_style=False)
        
        print(f"  ✅ 네임스페이스 매니페스트 생성됨")
        print(f"  📄 파일: {self.manifests_dir}/namespace.yaml")
    
    def create_storage_class(self):
        """스토리지 클래스 생성"""
        print("💾 프로덕션용 스토리지 클래스 생성 중...")
        
        storage_manifest = {
            'apiVersion': 'storage.k8s.io/v1',
            'kind': 'StorageClass',
            'metadata': {
                'name': 'milvus-ssd',
                'labels': {
                    'app': 'milvus'
                }
            },
            'provisioner': 'kubernetes.io/aws-ebs',  # AWS 예시
            'parameters': {
                'type': 'gp3',
                'iops': '3000',
                'throughput': '125'
            },
            'volumeBindingMode': 'WaitForFirstConsumer',
            'reclaimPolicy': 'Retain'
        }
        
        with open(self.manifests_dir / 'storage-class.yaml', 'w') as f:
            yaml.dump(storage_manifest, f, default_flow_style=False)
        
        print("  ✅ SSD 스토리지 클래스 생성됨")
        print("  💡 특징: GP3, 3000 IOPS, Retain 정책")
    
    def create_config_maps(self):
        """ConfigMap 생성"""
        print("⚙️  ConfigMap 생성 중...")
        
        # Milvus 설정
        milvus_config = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'milvus-config',
                'namespace': self.namespace
            },
            'data': {
                'milvus.yaml': '''
# Milvus 프로덕션 설정
etcd:
  endpoints:
    - milvus-etcd:2379
  
minio:
  address: milvus-minio
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  useSSL: false
  bucketName: milvus-bucket

common:
  defaultPartitionName: "_default"
  defaultIndexName: "_default_idx"
  retentionDuration: 432000  # 5 days

proxy:
  port: 19530
  http:
    enabled: true
    port: 9091

queryNode:
  cacheSize: "16Gi"
  
dataNode:
  flush:
    size: "512MB"
    interval: 600

indexNode:
  scheduler:
    buildParallel: 8
'''
            }
        }
        
        with open(self.manifests_dir / 'configmap.yaml', 'w') as f:
            yaml.dump(milvus_config, f, default_flow_style=False)
        
        print("  ✅ Milvus ConfigMap 생성됨")
    
    def create_secrets(self):
        """Secrets 생성"""
        print("🔐 보안 정보 생성 중...")
        
        import base64
        
        # 기본 인증 정보 (실제 환경에서는 안전하게 관리)
        username = base64.b64encode(b'admin').decode('utf-8')
        password = base64.b64encode(b'Milvus123!').decode('utf-8')
        api_key = base64.b64encode(b'milvus-prod-api-key-2024').decode('utf-8')
        
        secret_manifest = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'milvus-auth',
                'namespace': self.namespace
            },
            'type': 'Opaque',
            'data': {
                'username': username,
                'password': password,
                'api-key': api_key
            }
        }
        
        with open(self.manifests_dir / 'secrets.yaml', 'w') as f:
            yaml.dump(secret_manifest, f, default_flow_style=False)
        
        print("  ✅ 인증 Secret 생성됨")
        print("  ⚠️  실제 환경에서는 Vault, AWS Secrets Manager 등 사용 권장")
    
    def create_milvus_deployment(self):
        """Milvus 클러스터 배포 매니페스트 생성"""
        print("🚀 Milvus 클러스터 배포 매니페스트 생성 중...")
        
        # etcd 배포
        etcd_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'StatefulSet',
            'metadata': {
                'name': 'milvus-etcd',
                'namespace': self.namespace
            },
            'spec': {
                'serviceName': 'milvus-etcd',
                'replicas': 3,
                'selector': {'matchLabels': {'app': 'milvus-etcd'}},
                'template': {
                    'metadata': {'labels': {'app': 'milvus-etcd'}},
                    'spec': {
                        'containers': [{
                            'name': 'etcd',
                            'image': 'quay.io/coreos/etcd:v3.5.5',
                            'ports': [{'containerPort': 2379}, {'containerPort': 2380}],
                            'env': [
                                {'name': 'ETCD_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}},
                                {'name': 'ETCD_INITIAL_CLUSTER_STATE', 'value': 'new'},
                                {'name': 'ETCD_LISTEN_CLIENT_URLS', 'value': 'http://0.0.0.0:2379'},
                                {'name': 'ETCD_ADVERTISE_CLIENT_URLS', 'value': 'http://0.0.0.0:2379'}
                            ],
                            'resources': {
                                'requests': {'cpu': '100m', 'memory': '256Mi'},
                                'limits': {'cpu': '500m', 'memory': '1Gi'}
                            }
                        }]
                    }
                }
            }
        }
        
        # MinIO 배포
        minio_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'milvus-minio',
                'namespace': self.namespace
            },
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': 'milvus-minio'}},
                'template': {
                    'metadata': {'labels': {'app': 'milvus-minio'}},
                    'spec': {
                        'containers': [{
                            'name': 'minio',
                            'image': 'minio/minio:RELEASE.2023-03-20T20-16-18Z',
                            'args': ['server', '/data', '--console-address', ':9001'],
                            'ports': [{'containerPort': 9000}, {'containerPort': 9001}],
                            'env': [
                                {'name': 'MINIO_ACCESS_KEY', 'value': 'minioadmin'},
                                {'name': 'MINIO_SECRET_KEY', 'value': 'minioadmin'}
                            ],
                            'resources': {
                                'requests': {'cpu': '200m', 'memory': '512Mi'},
                                'limits': {'cpu': '1000m', 'memory': '2Gi'}
                            }
                        }]
                    }
                }
            }
        }
        
        # Milvus Standalone 배포 (프로덕션에서는 분산 모드 권장)
        milvus_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'milvus-standalone',
                'namespace': self.namespace
            },
            'spec': {
                'replicas': 1,
                'selector': {'matchLabels': {'app': 'milvus-standalone'}},
                'template': {
                    'metadata': {'labels': {'app': 'milvus-standalone'}},
                    'spec': {
                        'containers': [{
                            'name': 'milvus',
                            'image': 'milvusdb/milvus:v2.4.0',
                            'command': ['milvus', 'run', 'standalone'],
                            'ports': [{'containerPort': 19530}, {'containerPort': 9091}],
                            'env': [
                                {'name': 'ETCD_ENDPOINTS', 'value': 'milvus-etcd:2379'},
                                {'name': 'MINIO_ADDRESS', 'value': 'milvus-minio:9000'}
                            ],
                            'volumeMounts': [{
                                'name': 'config',
                                'mountPath': '/milvus/configs/milvus.yaml',
                                'subPath': 'milvus.yaml'
                            }],
                            'resources': {
                                'requests': {'cpu': '1000m', 'memory': '4Gi'},
                                'limits': {'cpu': '4000m', 'memory': '8Gi'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/healthz', 'port': 9091},
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/healthz', 'port': 9091},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }],
                        'volumes': [{
                            'name': 'config',
                            'configMap': {'name': 'milvus-config'}
                        }]
                    }
                }
            }
        }
        
        # 모든 배포 매니페스트 저장
        deployments = [
            ('etcd-deployment.yaml', etcd_deployment),
            ('minio-deployment.yaml', minio_deployment),
            ('milvus-deployment.yaml', milvus_deployment)
        ]
        
        for filename, deployment in deployments:
            with open(self.manifests_dir / filename, 'w') as f:
                yaml.dump(deployment, f, default_flow_style=False)
        
        print("  ✅ 배포 매니페스트 생성 완료")
        print("  📦 구성 요소: etcd (3 replicas), MinIO, Milvus")
    
    def create_services(self):
        """서비스 생성"""
        print("🌐 Kubernetes 서비스 생성 중...")
        
        services = []
        
        # etcd 서비스
        etcd_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'milvus-etcd',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {'app': 'milvus-etcd'},
                'ports': [
                    {'name': 'client', 'port': 2379, 'targetPort': 2379},
                    {'name': 'peer', 'port': 2380, 'targetPort': 2380}
                ],
                'clusterIP': 'None'  # Headless service for StatefulSet
            }
        }
        services.append(('etcd-service.yaml', etcd_service))
        
        # MinIO 서비스
        minio_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'milvus-minio',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {'app': 'milvus-minio'},
                'ports': [
                    {'name': 'api', 'port': 9000, 'targetPort': 9000},
                    {'name': 'console', 'port': 9001, 'targetPort': 9001}
                ]
            }
        }
        services.append(('minio-service.yaml', minio_service))
        
        # Milvus 서비스 (내부)
        milvus_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'milvus-cluster',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {'app': 'milvus-standalone'},
                'ports': [
                    {'name': 'grpc', 'port': 19530, 'targetPort': 19530},
                    {'name': 'http', 'port': 9091, 'targetPort': 9091}
                ]
            }
        }
        services.append(('milvus-service.yaml', milvus_service))
        
        # Milvus LoadBalancer 서비스 (외부 접근)
        milvus_lb_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'milvus-loadbalancer',
                'namespace': self.namespace,
                'annotations': {
                    'service.beta.kubernetes.io/aws-load-balancer-type': 'nlb'
                }
            },
            'spec': {
                'type': 'LoadBalancer',
                'selector': {'app': 'milvus-standalone'},
                'ports': [
                    {'name': 'grpc', 'port': 19530, 'targetPort': 19530}
                ]
            }
        }
        services.append(('milvus-loadbalancer.yaml', milvus_lb_service))
        
        # 모든 서비스 매니페스트 저장
        for filename, service in services:
            with open(self.manifests_dir / filename, 'w') as f:
                yaml.dump(service, f, default_flow_style=False)
        
        print("  ✅ 서비스 매니페스트 생성 완료")
        print("  🌐 구성: etcd, MinIO, Milvus (Internal + LoadBalancer)")
    
    def create_hpa(self):
        """Horizontal Pod Autoscaler 생성"""
        print("📊 자동 스케일링 설정 중...")
        
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'milvus-hpa',
                'namespace': self.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'milvus-standalone'
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
        
        with open(self.manifests_dir / 'hpa.yaml', 'w') as f:
            yaml.dump(hpa_manifest, f, default_flow_style=False)
        
        print("  ✅ HPA 설정 완료")
        print("  📊 설정: CPU 70%, Memory 80%, 2-10 replicas")
    
    def create_network_policies(self):
        """네트워크 정책 생성"""
        print("🔒 네트워크 보안 정책 생성 중...")
        
        # Milvus 네트워크 정책
        network_policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'milvus-network-policy',
                'namespace': self.namespace
            },
            'spec': {
                'podSelector': {'matchLabels': {'app': 'milvus-standalone'}},
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    {
                        'from': [
                            {'namespaceSelector': {'matchLabels': {'name': 'app-namespace'}}},
                            {'podSelector': {'matchLabels': {'app': 'client-app'}}}
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 19530},
                            {'protocol': 'TCP', 'port': 9091}
                        ]
                    }
                ],
                'egress': [
                    {
                        'to': [{'podSelector': {'matchLabels': {'app': 'milvus-etcd'}}}],
                        'ports': [{'protocol': 'TCP', 'port': 2379}]
                    },
                    {
                        'to': [{'podSelector': {'matchLabels': {'app': 'milvus-minio'}}}],
                        'ports': [{'protocol': 'TCP', 'port': 9000}]
                    }
                ]
            }
        }
        
        with open(self.manifests_dir / 'network-policy.yaml', 'w') as f:
            yaml.dump(network_policy, f, default_flow_style=False)
        
        print("  ✅ 네트워크 정책 생성 완료")
        print("  🔒 설정: Ingress/Egress 제한, 포트별 접근 제어")
    
    def create_monitoring_resources(self):
        """모니터링 리소스 생성"""
        print("📈 모니터링 리소스 생성 중...")
        
        # ServiceMonitor (Prometheus Operator용)
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'milvus-metrics',
                'namespace': self.namespace,
                'labels': {'app': 'milvus'}
            },
            'spec': {
                'selector': {
                    'matchLabels': {'app': 'milvus-standalone'}
                },
                'endpoints': [{
                    'port': 'http',
                    'path': '/metrics',
                    'interval': '30s'
                }]
            }
        }
        
        with open(self.manifests_dir / 'service-monitor.yaml', 'w') as f:
            yaml.dump(service_monitor, f, default_flow_style=False)
        
        print("  ✅ ServiceMonitor 생성 완료")
        print("  📊 설정: 30초 간격 메트릭 수집")
    
    def generate_helm_values(self):
        """Helm Values 파일 생성"""
        print("⚙️  Helm Values 파일 생성 중...")
        
        helm_values = {
            'cluster': {
                'enabled': True
            },
            'etcd': {
                'replicaCount': 3,
                'persistence': {
                    'enabled': True,
                    'storageClass': 'milvus-ssd',
                    'size': '50Gi'
                },
                'resources': {
                    'limits': {'cpu': '1000m', 'memory': '2Gi'},
                    'requests': {'cpu': '500m', 'memory': '1Gi'}
                }
            },
            'minio': {
                'mode': 'standalone',
                'persistence': {
                    'enabled': True,
                    'storageClass': 'milvus-ssd',
                    'size': '200Gi'
                },
                'resources': {
                    'limits': {'cpu': '2000m', 'memory': '4Gi'},
                    'requests': {'cpu': '1000m', 'memory': '2Gi'}
                }
            },
            'proxy': {
                'replicaCount': 2,
                'resources': {
                    'limits': {'cpu': '2000m', 'memory': '4Gi'},
                    'requests': {'cpu': '1000m', 'memory': '2Gi'}
                },
                'service': {
                    'type': 'LoadBalancer',
                    'annotations': {
                        'service.beta.kubernetes.io/aws-load-balancer-type': 'nlb'
                    }
                }
            },
            'queryNode': {
                'replicaCount': 2,
                'resources': {
                    'limits': {'cpu': '4000m', 'memory': '16Gi'},
                    'requests': {'cpu': '2000m', 'memory': '8Gi'}
                }
            },
            'dataNode': {
                'replicaCount': 2,
                'resources': {
                    'limits': {'cpu': '2000m', 'memory': '8Gi'},
                    'requests': {'cpu': '1000m', 'memory': '4Gi'}
                }
            },
            'indexNode': {
                'replicaCount': 1,
                'resources': {
                    'limits': {'cpu': '4000m', 'memory': '16Gi'},
                    'requests': {'cpu': '2000m', 'memory': '8Gi'}
                }
            },
            'metrics': {
                'serviceMonitor': {
                    'enabled': True
                }
            },
            'log': {
                'level': 'info'
            },
            'image': {
                'repository': 'milvusdb/milvus',
                'tag': 'v2.4.0',
                'pullPolicy': 'IfNotPresent'
            }
        }
        
        with open(self.manifests_dir / 'helm-values.yaml', 'w') as f:
            yaml.dump(helm_values, f, default_flow_style=False)
        
        print("  ✅ Helm Values 파일 생성 완료")
        print("  🎯 설정: 고가용성, 리소스 최적화, 모니터링 활성화")
    
    def simulate_cluster_operations(self):
        """클러스터 운영 시뮬레이션"""
        print("\n🎮 클러스터 운영 시뮬레이션...")
        
        operations = [
            ("네임스페이스 생성", "kubectl apply -f k8s-manifests/namespace.yaml"),
            ("스토리지 클래스 생성", "kubectl apply -f k8s-manifests/storage-class.yaml"),
            ("ConfigMap 적용", "kubectl apply -f k8s-manifests/configmap.yaml"),
            ("Secrets 적용", "kubectl apply -f k8s-manifests/secrets.yaml"),
            ("etcd 배포", "kubectl apply -f k8s-manifests/etcd-deployment.yaml"),
            ("MinIO 배포", "kubectl apply -f k8s-manifests/minio-deployment.yaml"),
            ("Milvus 배포", "kubectl apply -f k8s-manifests/milvus-deployment.yaml"),
            ("서비스 생성", "kubectl apply -f k8s-manifests/"),
            ("HPA 설정", "kubectl apply -f k8s-manifests/hpa.yaml"),
            ("네트워크 정책", "kubectl apply -f k8s-manifests/network-policy.yaml")
        ]
        
        for i, (operation, command) in enumerate(operations, 1):
            print(f"  {i:2d}. {operation}")
            print(f"      $ {command}")
            time.sleep(0.5)
        
        print("\n  📊 클러스터 상태 확인:")
        monitoring_commands = [
            "kubectl get pods -n milvus-prod",
            "kubectl get services -n milvus-prod",
            "kubectl get hpa -n milvus-prod",
            "kubectl logs -f deployment/milvus-standalone -n milvus-prod"
        ]
        
        for cmd in monitoring_commands:
            print(f"    $ {cmd}")
        
        print("\n  🚀 Helm을 이용한 배포 (권장):")
        helm_commands = [
            "helm repo add milvus https://zilliztech.github.io/milvus-helm/",
            "helm repo update",
            f"helm install {self.release_name} milvus/milvus -n {self.namespace} --create-namespace -f k8s-manifests/helm-values.yaml",
            f"helm status {self.release_name} -n {self.namespace}",
            f"helm upgrade {self.release_name} milvus/milvus -n {self.namespace} -f k8s-manifests/helm-values.yaml"
        ]
        
        for cmd in helm_commands:
            print(f"    $ {cmd}")
    
    def test_cluster_connectivity(self):
        """클러스터 연결 테스트"""
        print("\n🔗 클러스터 연결 테스트...")
        
        try:
            # 시뮬레이션된 연결 정보
            cluster_endpoints = [
                "milvus-loadbalancer.milvus-prod.svc.cluster.local:19530",
                "external-lb-12345.us-west-2.elb.amazonaws.com:19530",
                "localhost:19530"
            ]
            
            print("  📡 사용 가능한 엔드포인트:")
            for i, endpoint in enumerate(cluster_endpoints, 1):
                print(f"    {i}. {endpoint}")
            
            # 가상 연결 테스트
            print("\n  🧪 연결 테스트 시뮬레이션:")
            test_results = [
                ("Health Check", "✅ 정상"),
                ("gRPC 연결", "✅ 성공"),
                ("HTTP API", "✅ 응답"),
                ("인증 확인", "✅ 인증됨"),
                ("권한 확인", "✅ 읽기/쓰기 권한")
            ]
            
            for test_name, result in test_results:
                print(f"    {test_name}: {result}")
                time.sleep(0.3)
            
            print("\n  📊 클러스터 메트릭:")
            metrics = {
                "Pod 상태": "3/3 Running",
                "Memory 사용률": "45%",
                "CPU 사용률": "32%",
                "Active Connections": "127",
                "QPS": "2,450",
                "Storage 사용량": "156GB / 500GB"
            }
            
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")
            
        except Exception as e:
            print(f"  ❌ 연결 테스트 실패: {e}")

def main():
    """메인 실행 함수"""
    print("🐳 Milvus Kubernetes 배포 및 클러스터 관리")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = KubernetesManager()
    
    try:
        # 1. 전제 조건 확인
        print("\n" + "=" * 80)
        print(" 🔍 환경 확인")
        print("=" * 80)
        has_k8s = manager.check_prerequisites()
        
        # 2. Kubernetes 매니페스트 생성
        print("\n" + "=" * 80)
        print(" 📋 Kubernetes 매니페스트 생성")
        print("=" * 80)
        
        manager.create_namespace()
        manager.create_storage_class()
        manager.create_config_maps()
        manager.create_secrets()
        manager.create_milvus_deployment()
        manager.create_services()
        manager.create_hpa()
        manager.create_network_policies()
        manager.create_monitoring_resources()
        
        # 3. Helm Values 생성
        print("\n" + "=" * 80)
        print(" ⚙️  Helm 차트 설정")
        print("=" * 80)
        manager.generate_helm_values()
        
        # 4. 배포 시뮬레이션
        print("\n" + "=" * 80)
        print(" 🚀 배포 및 운영 가이드")
        print("=" * 80)
        manager.simulate_cluster_operations()
        
        # 5. 연결 테스트
        print("\n" + "=" * 80)
        print(" 🧪 클러스터 테스트")
        print("=" * 80)
        manager.test_cluster_connectivity()
        
        # 6. 요약 및 다음 단계
        print("\n" + "=" * 80)
        print(" 📊 배포 완료 요약")
        print("=" * 80)
        
        print("✅ 생성된 리소스:")
        manifests = list(manager.manifests_dir.glob("*.yaml"))
        for manifest in sorted(manifests):
            print(f"  📄 {manifest.name}")
        
        print(f"\n📁 매니페스트 위치: {manager.manifests_dir.absolute()}")
        print(f"📦 네임스페이스: {manager.namespace}")
        print(f"🏷️  릴리스명: {manager.release_name}")
        
        print("\n💡 운영 팁:")
        tips = [
            "매니페스트 파일을 버전 관리 시스템에 저장하세요",
            "Helm을 사용하면 업그레이드와 롤백이 쉬워집니다",
            "프로덕션에서는 리소스 제한을 적절히 설정하세요",
            "모니터링과 로깅을 반드시 설정하세요",
            "정기적인 백업과 재해 복구 계획을 수립하세요"
        ]
        
        for tip in tips:
            print(f"  • {tip}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Kubernetes 배포 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • Kubernetes 매니페스트 작성 및 관리")
    print("  • Helm 차트를 이용한 패키지 관리")
    print("  • 프로덕션급 리소스 설정 및 보안")
    print("  • 자동 스케일링 및 네트워크 정책")
    
    print("\n🚀 다음 단계:")
    print("  python step05_production/02_cicd_pipeline.py")

if __name__ == "__main__":
    main() 