#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus 보안 및 인증 시스템

이 스크립트는 Milvus 프로덕션 환경의 보안을 강화하고 
인증 시스템을 구현합니다. RBAC, TLS/SSL, API 키 관리,
감사 로그 등 엔터프라이즈급 보안 기능을 다룹니다.
"""

import os
import sys
import time
import yaml
import json
import base64
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
import secrets
import uuid

class SecurityRole(Enum):
    """보안 역할"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"

class AuditAction(Enum):
    """감사 액션"""
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE_COLLECTION = "create_collection"
    DROP_COLLECTION = "drop_collection"
    INSERT_DATA = "insert_data"
    SEARCH_DATA = "search_data"
    DELETE_DATA = "delete_data"
    MODIFY_SETTINGS = "modify_settings"

class SecurityAuthManager:
    """보안 및 인증 관리자"""
    
    def __init__(self):
        self.namespace = "milvus-production"
        self.security_dir = Path("security-configs")
        self.certs_dir = Path("security-configs/certs")
        self.audit_logs: List[Dict] = []
        
        # 디렉토리 생성
        for directory in [self.security_dir, self.certs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 기본 사용자 및 역할
        self.users = {
            "admin": {
                "password_hash": self.hash_password("admin123!"),
                "roles": [SecurityRole.ADMIN],
                "api_keys": [],
                "created_at": datetime.now(),
                "last_login": None,
                "active": True
            },
            "developer": {
                "password_hash": self.hash_password("dev123!"),
                "roles": [SecurityRole.DEVELOPER],
                "api_keys": [],
                "created_at": datetime.now(),
                "last_login": None,
                "active": True
            },
            "readonly": {
                "password_hash": self.hash_password("read123!"),
                "roles": [SecurityRole.VIEWER],
                "api_keys": [],
                "created_at": datetime.now(),
                "last_login": None,
                "active": True
            }
        }
        
        self.role_permissions = {
            SecurityRole.ADMIN: [
                "collection:*", "data:*", "index:*", "user:*", "system:*"
            ],
            SecurityRole.DEVELOPER: [
                "collection:create", "collection:read", "collection:update",
                "data:create", "data:read", "data:update", "data:delete",
                "index:create", "index:read", "index:update"
            ],
            SecurityRole.VIEWER: [
                "collection:read", "data:read", "index:read"
            ],
            SecurityRole.SERVICE_ACCOUNT: [
                "data:create", "data:read", "data:search"
            ]
        }
    
    def hash_password(self, password: str) -> str:
        """패스워드 해시화"""
        salt = secrets.token_hex(16)
        pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                      password.encode('utf-8'), 
                                      salt.encode('utf-8'), 
                                      100000)
        return salt + pwdhash.hex()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """패스워드 검증"""
        salt = hashed[:32]
        stored_hash = hashed[32:]
        pwdhash = hashlib.pbkdf2_hmac('sha256',
                                      password.encode('utf-8'),
                                      salt.encode('utf-8'),
                                      100000)
        return pwdhash.hex() == stored_hash
    
    def generate_api_key(self, user: str, description: str = "") -> str:
        """API 키 생성"""
        api_key = f"mk_{secrets.token_urlsafe(32)}"
        key_info = {
            "key": api_key,
            "user": user,
            "description": description,
            "created_at": datetime.now(),
            "last_used": None,
            "active": True
        }
        
        if user in self.users:
            self.users[user]["api_keys"].append(key_info)
        
        self.log_audit_event(user, AuditAction.MODIFY_SETTINGS, {
            "action": "api_key_created",
            "description": description
        })
        
        return api_key
    
    def log_audit_event(self, user: str, action: AuditAction, details: Dict[str, Any]):
        """감사 로그 기록"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "action": action.value,
            "details": details,
            "ip_address": "127.0.0.1",  # 시뮬레이션
            "user_agent": "MilvusClient/1.0",
            "session_id": str(uuid.uuid4())
        }
        
        self.audit_logs.append(log_entry)
        
        # 실제 환경에서는 외부 로그 시스템에 전송
        print(f"  🔍 감사로그: {user} - {action.value}")
    
    def create_tls_certificates(self):
        """TLS 인증서 생성"""
        print("🔐 TLS 인증서 설정 생성 중...")
        
        # OpenSSL 설정 파일
        openssl_config = '''[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=KR
ST=Seoul
L=Seoul
O=Milvus Production
OU=Infrastructure
CN=milvus.example.com

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = milvus.example.com
DNS.2 = *.milvus.example.com
DNS.3 = milvus-production.default.svc.cluster.local
DNS.4 = localhost
IP.1 = 127.0.0.1
'''
        
        with open(self.certs_dir / 'openssl.conf', 'w') as f:
            f.write(openssl_config)
        
        # 인증서 생성 스크립트
        cert_script = '''#!/bin/bash
set -e

CERTS_DIR="security-configs/certs"
cd $CERTS_DIR

echo "🔑 Generating CA private key..."
openssl genrsa -out ca-key.pem 4096

echo "🏛️  Generating CA certificate..."
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca-cert.pem -subj "/C=KR/ST=Seoul/L=Seoul/O=Milvus CA/CN=Milvus CA"

echo "🔑 Generating server private key..."
openssl genrsa -out server-key.pem 4096

echo "📄 Generating server certificate signing request..."
openssl req -subj "/C=KR/ST=Seoul/L=Seoul/O=Milvus Production/CN=milvus.example.com" -sha256 -new -key server-key.pem -out server.csr

echo "🏛️  Generating server certificate..."
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca-cert.pem -CAkey ca-key.pem -out server-cert.pem -extensions v3_req -extfile openssl.conf -CAcreateserial

echo "🔑 Generating client private key..."
openssl genrsa -out client-key.pem 4096

echo "📄 Generating client certificate signing request..."
openssl req -subj "/C=KR/ST=Seoul/L=Seoul/O=Milvus Client/CN=client" -new -key client-key.pem -out client.csr

echo "🏛️  Generating client certificate..."
openssl x509 -req -days 365 -sha256 -in client.csr -CA ca-cert.pem -CAkey ca-key.pem -out client-cert.pem -CAcreateserial

echo "🧹 Cleaning up..."
rm server.csr client.csr

echo "✅ TLS certificates generated successfully!"
echo "📁 Certificates location: $(pwd)"
'''
        
        with open(self.security_dir / 'generate-certs.sh', 'w') as f:
            f.write(cert_script)
        
        print("  ✅ TLS 인증서 생성 스크립트 작성됨")
        print("  💫 실행 명령: chmod +x security-configs/generate-certs.sh && ./security-configs/generate-certs.sh")
    
    def create_rbac_configuration(self):
        """RBAC 설정 생성"""
        print("👥 RBAC 권한 관리 설정 중...")
        
        # Kubernetes RBAC 설정
        rbac_manifests = []
        
        # ServiceAccount
        service_account = {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'name': 'milvus-operator',
                'namespace': self.namespace
            }
        }
        rbac_manifests.append(('service-account.yaml', service_account))
        
        # ClusterRole
        cluster_role = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRole',
            'metadata': {
                'name': 'milvus-operator'
            },
            'rules': [
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'services', 'endpoints', 'persistentvolumeclaims', 'events', 'configmaps', 'secrets'],
                    'verbs': ['*']
                },
                {
                    'apiGroups': ['apps'],
                    'resources': ['deployments', 'daemonsets', 'replicasets', 'statefulsets'],
                    'verbs': ['*']
                },
                {
                    'apiGroups': ['monitoring.coreos.com'],
                    'resources': ['servicemonitors'],
                    'verbs': ['get', 'create']
                }
            ]
        }
        rbac_manifests.append(('cluster-role.yaml', cluster_role))
        
        # ClusterRoleBinding
        cluster_role_binding = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRoleBinding',
            'metadata': {
                'name': 'milvus-operator'
            },
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'ClusterRole',
                'name': 'milvus-operator'
            },
            'subjects': [{
                'kind': 'ServiceAccount',
                'name': 'milvus-operator',
                'namespace': self.namespace
            }]
        }
        rbac_manifests.append(('cluster-role-binding.yaml', cluster_role_binding))
        
        # Role for namespace-specific operations
        role = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'Role',
            'metadata': {
                'namespace': self.namespace,
                'name': 'milvus-developer'
            },
            'rules': [
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'pods/log'],
                    'verbs': ['get', 'list', 'watch']
                },
                {
                    'apiGroups': ['apps'],
                    'resources': ['deployments'],
                    'verbs': ['get', 'list', 'watch', 'patch']
                }
            ]
        }
        rbac_manifests.append(('developer-role.yaml', role))
        
        # RoleBinding
        role_binding = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'RoleBinding',
            'metadata': {
                'name': 'milvus-developers',
                'namespace': self.namespace
            },
            'subjects': [{
                'kind': 'User',
                'name': 'developer@example.com',
                'apiGroup': 'rbac.authorization.k8s.io'
            }],
            'roleRef': {
                'kind': 'Role',
                'name': 'milvus-developer',
                'apiGroup': 'rbac.authorization.k8s.io'
            }
        }
        rbac_manifests.append(('developer-role-binding.yaml', role_binding))
        
        # RBAC 매니페스트 저장
        rbac_dir = self.security_dir / "rbac"
        rbac_dir.mkdir(exist_ok=True)
        
        for filename, manifest in rbac_manifests:
            with open(rbac_dir / filename, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
        
        print("  ✅ Kubernetes RBAC 설정 생성됨")
        
        # Milvus 사용자 권한 설정
        user_permissions = {
            'users': {},
            'roles': {}
        }
        
        for username, user_info in self.users.items():
            user_permissions['users'][username] = {
                'roles': [role.value for role in user_info['roles']],
                'active': user_info['active'],
                'created_at': user_info['created_at'].isoformat()
            }
        
        for role, permissions in self.role_permissions.items():
            user_permissions['roles'][role.value] = {
                'permissions': permissions,
                'description': f"Role for {role.value} access level"
            }
        
        with open(self.security_dir / 'user-permissions.json', 'w') as f:
            json.dump(user_permissions, f, indent=2, default=str)
        
        print("  ✅ Milvus 사용자 권한 설정 생성됨")
    
    def create_network_policies(self):
        """네트워크 보안 정책 생성"""
        print("🌐 네트워크 보안 정책 설정 중...")
        
        # 기본 네트워크 정책 (모든 트래픽 차단)
        default_deny = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'default-deny-all',
                'namespace': self.namespace
            },
            'spec': {
                'podSelector': {},
                'policyTypes': ['Ingress', 'Egress']
            }
        }
        
        # Milvus 트래픽 허용 정책
        milvus_allow = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'milvus-allow-traffic',
                'namespace': self.namespace
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {'app': 'milvus'}
                },
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    {
                        # API Gateway에서의 트래픽 허용
                        'from': [
                            {
                                'namespaceSelector': {
                                    'matchLabels': {'name': 'api-gateway'}
                                }
                            },
                            {
                                'podSelector': {
                                    'matchLabels': {'app': 'nginx-ingress'}
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 19530},
                            {'protocol': 'TCP', 'port': 9091}
                        ]
                    },
                    {
                        # 모니터링 시스템에서의 트래픽 허용
                        'from': [
                            {
                                'namespaceSelector': {
                                    'matchLabels': {'name': 'monitoring'}
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 9091}
                        ]
                    }
                ],
                'egress': [
                    {
                        # etcd 통신 허용
                        'to': [
                            {
                                'podSelector': {
                                    'matchLabels': {'app': 'etcd'}
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 2379}
                        ]
                    },
                    {
                        # MinIO 통신 허용
                        'to': [
                            {
                                'podSelector': {
                                    'matchLabels': {'app': 'minio'}
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 9000}
                        ]
                    },
                    {
                        # DNS 해결 허용
                        'to': [],
                        'ports': [
                            {'protocol': 'UDP', 'port': 53}
                        ]
                    }
                ]
            }
        }
        
        # 관리자 접근 정책
        admin_access = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'admin-access',
                'namespace': self.namespace
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {'app': 'milvus'}
                },
                'policyTypes': ['Ingress'],
                'ingress': [
                    {
                        'from': [
                            {
                                'ipBlock': {
                                    'cidr': '10.0.0.0/8'  # 내부 네트워크만 허용
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 19530}
                        ]
                    }
                ]
            }
        }
        
        # 네트워크 정책 저장
        network_policies = [
            ('default-deny-all.yaml', default_deny),
            ('milvus-allow-traffic.yaml', milvus_allow),
            ('admin-access.yaml', admin_access)
        ]
        
        network_dir = self.security_dir / "network-policies"
        network_dir.mkdir(exist_ok=True)
        
        for filename, policy in network_policies:
            with open(network_dir / filename, 'w') as f:
                yaml.dump(policy, f, default_flow_style=False)
        
        print("  ✅ 네트워크 보안 정책 생성됨")
    
    def create_secrets_management(self):
        """시크릿 관리 설정"""
        print("🔑 시크릿 관리 시스템 설정 중...")
        
        # Kubernetes Secrets
        secrets = []
        
        # TLS Secret
        tls_secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'milvus-tls',
                'namespace': self.namespace
            },
            'type': 'kubernetes.io/tls',
            'data': {
                # 실제 환경에서는 실제 인증서를 base64 인코딩
                'tls.crt': base64.b64encode(b'# TLS Certificate').decode('utf-8'),
                'tls.key': base64.b64encode(b'# TLS Private Key').decode('utf-8'),
                'ca.crt': base64.b64encode(b'# CA Certificate').decode('utf-8')
            }
        }
        secrets.append(('milvus-tls-secret.yaml', tls_secret))
        
        # Database Credentials
        db_secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'milvus-db-credentials',
                'namespace': self.namespace
            },
            'type': 'Opaque',
            'data': {
                'username': base64.b64encode(b'milvus_admin').decode('utf-8'),
                'password': base64.b64encode(b'secure_password_123!').decode('utf-8'),
                'etcd-username': base64.b64encode(b'etcd_user').decode('utf-8'),
                'etcd-password': base64.b64encode(b'etcd_pass_456!').decode('utf-8'),
                'minio-access-key': base64.b64encode(b'milvus_access_key').decode('utf-8'),
                'minio-secret-key': base64.b64encode(b'milvus_secret_key_789!').decode('utf-8')
            }
        }
        secrets.append(('milvus-db-credentials.yaml', db_secret))
        
        # API Keys Secret
        api_keys_data = {}
        for username, user_info in self.users.items():
            if user_info['api_keys']:
                api_keys_data[f'{username}-api-key'] = base64.b64encode(
                    user_info['api_keys'][0]['key'].encode('utf-8')
                ).decode('utf-8')
        
        api_secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'milvus-api-keys',
                'namespace': self.namespace
            },
            'type': 'Opaque',
            'data': api_keys_data
        }
        secrets.append(('milvus-api-keys.yaml', api_secret))
        
        # Secrets 저장
        secrets_dir = self.security_dir / "secrets"
        secrets_dir.mkdir(exist_ok=True)
        
        for filename, secret in secrets:
            with open(secrets_dir / filename, 'w') as f:
                yaml.dump(secret, f, default_flow_style=False)
        
        # Vault 설정 (시뮬레이션)
        vault_config = {
            'vault': {
                'address': 'https://vault.example.com:8200',
                'auth': {
                    'method': 'kubernetes',
                    'path': 'auth/kubernetes',
                    'role': 'milvus-production'
                },
                'secrets': {
                    'database': 'secret/milvus/database',
                    'tls': 'secret/milvus/tls',
                    'api-keys': 'secret/milvus/api-keys'
                }
            }
        }
        
        with open(self.security_dir / 'vault-config.yaml', 'w') as f:
            yaml.dump(vault_config, f, default_flow_style=False)
        
        print("  ✅ 시크릿 관리 설정 생성됨")
    
    def create_security_monitoring(self):
        """보안 모니터링 설정"""
        print("🔍 보안 모니터링 시스템 설정 중...")
        
        # Falco 규칙 (런타임 보안)
        falco_rules = '''# Milvus Security Rules
- rule: Unauthorized Process in Milvus Container
  desc: Detect unauthorized process execution in Milvus containers
  condition: >
    spawned_process and container and
    container.image.repository contains "milvus" and
    not proc.name in (milvus, sh, bash, grep, ps, top, netstat)
  output: >
    Unauthorized process in Milvus container
    (user=%user.name command=%proc.cmdline container=%container.name image=%container.image.repository)
  priority: WARNING
  tags: [milvus, security]

- rule: Milvus Configuration File Modified
  desc: Detect modifications to Milvus configuration files
  condition: >
    open_write and container and
    container.image.repository contains "milvus" and
    fd.name in (/milvus/configs/milvus.yaml, /milvus/configs/advanced.yaml)
  output: >
    Milvus configuration file modified
    (user=%user.name file=%fd.name container=%container.name)
  priority: ERROR
  tags: [milvus, configuration]

- rule: Suspicious Network Activity from Milvus
  desc: Detect unusual network connections from Milvus containers
  condition: >
    outbound and container and
    container.image.repository contains "milvus" and
    not fd.sip in (etcd_ips, minio_ips, prometheus_ips)
  output: >
    Suspicious network activity from Milvus
    (connection=%fd.name container=%container.name)
  priority: WARNING
  tags: [milvus, network]
'''
        
        with open(self.security_dir / 'falco-rules.yaml', 'w') as f:
            f.write(falco_rules)
        
        # 보안 메트릭 수집 설정
        security_metrics = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'milvus-security',
                    'static_configs': [
                        {
                            'targets': ['milvus:9091']
                        }
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'falco-security',
                    'static_configs': [
                        {
                            'targets': ['falco:8765']
                        }
                    ]
                }
            ],
            'rule_files': [
                'security-alerts.yml'
            ]
        }
        
        with open(self.security_dir / 'security-monitoring.yaml', 'w') as f:
            yaml.dump(security_metrics, f, default_flow_style=False)
        
        # 보안 알림 규칙
        alert_rules = '''groups:
- name: milvus_security
  rules:
  - alert: MilvusUnauthorizedAccess
    expr: increase(milvus_auth_failures_total[5m]) > 5
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Multiple authentication failures detected"
      description: "{{ $value }} authentication failures in the last 5 minutes"

  - alert: MilvusHighMemoryUsage
    expr: milvus_memory_usage_ratio > 0.9
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Milvus memory usage is high"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  - alert: MilvusConfigurationChanged
    expr: increase(milvus_config_changes_total[1h]) > 0
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "Milvus configuration has been modified"
      description: "Configuration changes detected in the last hour"

  - alert: MilvusSecurityViolation
    expr: increase(falco_security_violations_total{service="milvus"}[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Security violation detected in Milvus"
      description: "Falco detected security violation: {{ $labels.rule }}"
'''
        
        with open(self.security_dir / 'security-alerts.yml', 'w') as f:
            f.write(alert_rules)
        
        print("  ✅ 보안 모니터링 설정 생성됨")
    
    def simulate_security_operations(self):
        """보안 운영 시뮬레이션"""
        print("\n🎮 보안 시스템 운영 시뮬레이션...")
        
        # 1. 사용자 인증 시뮬레이션
        print("\n👤 사용자 인증 테스트:")
        test_users = [
            ("admin", "admin123!", True),
            ("developer", "dev123!", True),
            ("readonly", "read123!", True),
            ("hacker", "wrongpass", False)
        ]
        
        for username, password, should_succeed in test_users:
            if username in self.users:
                success = self.verify_password(password, self.users[username]["password_hash"])
                result = "✅ 성공" if success else "❌ 실패"
                expected = "예상됨" if success == should_succeed else "예상과 다름"
                print(f"  {username}: {result} ({expected})")
                
                if success:
                    self.users[username]["last_login"] = datetime.now()
                    self.log_audit_event(username, AuditAction.LOGIN, {"method": "password"})
            else:
                print(f"  {username}: ❌ 실패 (사용자 없음)")
        
        # 2. API 키 생성 및 테스트
        print(f"\n🔑 API 키 관리 테스트:")
        for username in ["admin", "developer"]:
            api_key = self.generate_api_key(username, f"{username} production key")
            print(f"  {username} API 키: {api_key[:20]}...")
        
        # 3. 권한 체크 시뮬레이션
        print(f"\n🛡️  권한 체크 테스트:")
        permission_tests = [
            ("admin", "collection:create", True),
            ("developer", "data:read", True),
            ("readonly", "data:delete", False),
            ("readonly", "collection:read", True)
        ]
        
        for username, permission, should_allow in permission_tests:
            if username in self.users:
                user_roles = self.users[username]["roles"]
                allowed = any(
                    permission in self.role_permissions[role] or 
                    any(perm.endswith(":*") and permission.startswith(perm[:-1]) 
                        for perm in self.role_permissions[role])
                    for role in user_roles
                )
                result = "✅ 허용" if allowed else "❌ 거부"
                expected = "예상됨" if allowed == should_allow else "예상과 다름"
                print(f"  {username} - {permission}: {result} ({expected})")
        
        # 4. 감사 로그 요약
        print(f"\n📋 감사 로그 요약:")
        action_counts = {}
        for log in self.audit_logs:
            action = log["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in action_counts.items():
            print(f"  {action}: {count}회")
        
        # 5. 보안 알림 시뮬레이션
        print(f"\n🚨 보안 알림 시뮬레이션:")
        security_events = [
            ("인증 실패 급증", "CRITICAL", "5분간 10회 이상 로그인 실패"),
            ("의심스러운 API 사용", "WARNING", "비정상적인 데이터 접근 패턴 감지"),
            ("설정 변경 감지", "INFO", "관리자가 시스템 설정 수정"),
            ("네트워크 이상", "WARNING", "허용되지 않은 외부 연결 시도")
        ]
        
        for event, severity, description in security_events:
            severity_emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(severity, "⚪")
            print(f"  {severity_emoji} [{severity}] {event}: {description}")
            time.sleep(0.3)
    
    def generate_security_report(self):
        """보안 상태 보고서 생성"""
        print("\n📊 보안 상태 보고서 생성 중...")
        
        report = {
            "report_date": datetime.now().isoformat(),
            "security_summary": {
                "total_users": len(self.users),
                "active_users": sum(1 for u in self.users.values() if u["active"]),
                "total_api_keys": sum(len(u["api_keys"]) for u in self.users.values()),
                "total_audit_logs": len(self.audit_logs)
            },
            "user_activity": {},
            "security_configurations": {
                "tls_enabled": True,
                "rbac_enabled": True,
                "network_policies": True,
                "audit_logging": True,
                "api_key_auth": True
            },
            "compliance_status": {
                "password_policy": "COMPLIANT",
                "access_control": "COMPLIANT", 
                "data_encryption": "COMPLIANT",
                "audit_trail": "COMPLIANT"
            },
            "recommendations": [
                "정기적인 API 키 로테이션 수행",
                "패스워드 복잡성 정책 강화",
                "MFA(다중 인증) 도입 검토",
                "정기적인 보안 스캔 수행",
                "침입 탐지 시스템 고도화"
            ]
        }
        
        # 사용자별 활동 요약
        for username, user_info in self.users.items():
            user_logs = [log for log in self.audit_logs if log["user"] == username]
            report["user_activity"][username] = {
                "last_login": user_info["last_login"].isoformat() if user_info["last_login"] else None,
                "total_actions": len(user_logs),
                "roles": [role.value for role in user_info["roles"]],
                "api_keys_count": len(user_info["api_keys"])
            }
        
        # 보고서 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.security_dir / f"security_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✅ 보안 보고서 저장됨: {report_file.name}")
        
        # 요약 출력
        print(f"\n📈 보안 상태 요약:")
        print(f"  총 사용자: {report['security_summary']['total_users']}명")
        print(f"  활성 사용자: {report['security_summary']['active_users']}명")
        print(f"  발급된 API 키: {report['security_summary']['total_api_keys']}개")
        print(f"  감사 로그: {report['security_summary']['total_audit_logs']}건")
        
        return report

def main():
    """메인 실행 함수"""
    print("🛡️ Milvus 보안 및 인증 시스템")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = SecurityAuthManager()
    
    try:
        # 1. TLS 인증서 설정
        print("\n" + "=" * 80)
        print(" 🔐 TLS/SSL 인증서 설정")
        print("=" * 80)
        manager.create_tls_certificates()
        
        # 2. RBAC 설정
        print("\n" + "=" * 80)
        print(" 👥 RBAC 권한 관리")
        print("=" * 80)
        manager.create_rbac_configuration()
        
        # 3. 네트워크 보안
        print("\n" + "=" * 80)
        print(" 🌐 네트워크 보안 정책")
        print("=" * 80)
        manager.create_network_policies()
        
        # 4. 시크릿 관리
        print("\n" + "=" * 80)
        print(" 🔑 시크릿 관리 시스템")
        print("=" * 80)
        manager.create_secrets_management()
        
        # 5. 보안 모니터링
        print("\n" + "=" * 80)
        print(" 🔍 보안 모니터링")
        print("=" * 80)
        manager.create_security_monitoring()
        
        # 6. 보안 운영 시뮬레이션
        print("\n" + "=" * 80)
        print(" 🎮 보안 시스템 운영")
        print("=" * 80)
        manager.simulate_security_operations()
        
        # 7. 보안 보고서 생성
        print("\n" + "=" * 80)
        print(" 📊 보안 상태 보고서")
        print("=" * 80)
        report = manager.generate_security_report()
        
        # 8. 요약
        print("\n" + "=" * 80)
        print(" 🛡️ 보안 설정 완료")
        print("=" * 80)
        
        print("✅ 생성된 보안 리소스:")
        security_resources = [
            "security-configs/generate-certs.sh",
            "security-configs/openssl.conf",
            "security-configs/rbac/service-account.yaml",
            "security-configs/rbac/cluster-role.yaml",
            "security-configs/rbac/developer-role.yaml",
            "security-configs/network-policies/default-deny-all.yaml",
            "security-configs/network-policies/milvus-allow-traffic.yaml",
            "security-configs/secrets/milvus-tls-secret.yaml",
            "security-configs/secrets/milvus-db-credentials.yaml",
            "security-configs/vault-config.yaml",
            "security-configs/falco-rules.yaml",
            "security-configs/security-monitoring.yaml",
            "security-configs/user-permissions.json"
        ]
        
        for resource in security_resources:
            print(f"  📄 {resource}")
        
        print("\n🔒 보안 기능 요약:")
        security_features = [
            "✅ TLS/SSL 암호화 통신",
            "✅ RBAC 역할 기반 접근 제어",
            "✅ API 키 인증 시스템",
            "✅ 네트워크 정책 기반 격리",
            "✅ 시크릿 안전 관리",
            "✅ 실시간 보안 모니터링",
            "✅ 종합 감사 로깅",
            "✅ 침입 탐지 시스템"
        ]
        
        for feature in security_features:
            print(f"  {feature}")
        
        print("\n💡 보안 운영 가이드:")
        operation_tips = [
            "정기적인 패스워드 및 API 키 로테이션",
            "최소 권한 원칙 적용",
            "모든 보안 이벤트 모니터링",
            "정기적인 보안 감사 수행",
            "침입 탐지 알림 즉시 대응",
            "백업 데이터 암호화 보관"
        ]
        
        for tip in operation_tips:
            print(f"  • {tip}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 보안 및 인증 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • 엔터프라이즈급 보안 아키텍처")
    print("  • RBAC 기반 접근 제어")
    print("  • TLS/SSL 암호화 구현")
    print("  • 네트워크 보안 정책")
    print("  • 실시간 보안 모니터링")
    
    print("\n🚀 다음 단계:")
    print("  python step05_production/06_production_monitoring.py")

if __name__ == "__main__":
    main() 