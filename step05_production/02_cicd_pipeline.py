#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus CI/CD 파이프라인 구축

이 스크립트는 Milvus 애플리케이션의 CI/CD 파이프라인을 구축하고 관리하는 방법을 보여줍니다.
GitHub Actions, GitLab CI, Docker 이미지 관리, 자동화된 테스트 및 배포를 다룹니다.
"""

import os
import sys
import yaml
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class CICDPipelineManager:
    """CI/CD 파이프라인 관리자"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.cicd_dir = Path(".github/workflows")
        self.gitlab_dir = Path(".gitlab-ci")
        self.docker_dir = Path("docker")
        self.scripts_dir = Path("scripts")
        
        # 디렉토리 생성
        for directory in [self.cicd_dir, self.gitlab_dir, self.docker_dir, self.scripts_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_github_actions_workflow(self):
        """GitHub Actions 워크플로우 생성"""
        print("🐙 GitHub Actions 워크플로우 생성 중...")
        
        # Main CI/CD workflow
        main_workflow = {
            'name': 'Milvus CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                },
                'release': {
                    'types': ['published']
                }
            },
            'env': {
                'REGISTRY': 'ghcr.io',
                'IMAGE_NAME': '${{ github.repository }}/milvus-app',
                'KUBECONFIG_DATA': '${{ secrets.KUBECONFIG_DATA }}',
                'DOCKER_BUILDKIT': '1'
            },
            'jobs': {
                'test': {
                    'name': 'Test Suite',
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.9', '3.10', '3.11']
                        }
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Cache dependencies',
                            'uses': 'actions/cache@v3',
                            'with': {
                                'path': '~/.cache/pip',
                                'key': '${{ runner.os }}-pip-${{ hashFiles(\'**/requirements.txt\') }}',
                                'restore-keys': '${{ runner.os }}-pip-'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
'''
                        },
                        {
                            'name': 'Lint code',
                            'run': '''
black --check .
flake8 .
mypy . --ignore-missing-imports
'''
                        },
                        {
                            'name': 'Run tests',
                            'run': '''
pytest --cov=. --cov-report=xml --cov-report=html
'''
                        },
                        {
                            'name': 'Upload coverage to Codecov',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'flags': 'unittests',
                                'name': 'codecov-umbrella'
                            }
                        }
                    ]
                },
                'security-scan': {
                    'name': 'Security Scan',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Run Bandit Security Scan',
                            'run': '''
pip install bandit
bandit -r . -f json -o bandit-report.json || true
'''
                        },
                        {
                            'name': 'Run Safety Check',
                            'run': '''
pip install safety
safety check --json --output safety-report.json || true
'''
                        },
                        {
                            'name': 'Upload security reports',
                            'uses': 'actions/upload-artifact@v3',
                            'with': {
                                'name': 'security-reports',
                                'path': '*-report.json'
                            }
                        }
                    ]
                },
                'build': {
                    'name': 'Build Docker Image',
                    'runs-on': 'ubuntu-latest',
                    'needs': ['test', 'security-scan'],
                    'outputs': {
                        'image-tag': '${{ steps.meta.outputs.tags }}',
                        'image-digest': '${{ steps.build.outputs.digest }}'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v3'
                        },
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v3',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v5',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}',
                                'tags': '''
type=ref,event=branch
type=ref,event=pr
type=sha,prefix={{branch}}-
type=raw,value=latest,enable={{is_default_branch}}
'''
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'id': 'build',
                            'uses': 'docker/build-push-action@v5',
                            'with': {
                                'context': '.',
                                'file': './docker/Dockerfile',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}',
                                'cache-from': 'type=gha',
                                'cache-to': 'type=gha,mode=max'
                            }
                        }
                    ]
                },
                'deploy-staging': {
                    'name': 'Deploy to Staging',
                    'runs-on': 'ubuntu-latest',
                    'needs': 'build',
                    'if': 'github.ref == \'refs/heads/develop\'',
                    'environment': 'staging',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up kubectl',
                            'uses': 'azure/setup-kubectl@v3',
                            'with': {
                                'version': 'v1.28.0'
                            }
                        },
                        {
                            'name': 'Set up Helm',
                            'uses': 'azure/setup-helm@v3',
                            'with': {
                                'version': 'v3.12.0'
                            }
                        },
                        {
                            'name': 'Configure kubectl',
                            'run': '''
echo "${{ secrets.KUBECONFIG_DATA }}" | base64 -d > kubeconfig
export KUBECONFIG=kubeconfig
'''
                        },
                        {
                            'name': 'Deploy to staging',
                            'run': '''
export KUBECONFIG=kubeconfig
helm upgrade --install milvus-staging ./helm/milvus \\
  --namespace milvus-staging \\
  --create-namespace \\
  --set image.tag=${{ needs.build.outputs.image-tag }} \\
  --set environment=staging \\
  --values helm/values-staging.yaml
'''
                        }
                    ]
                },
                'deploy-production': {
                    'name': 'Deploy to Production',
                    'runs-on': 'ubuntu-latest',
                    'needs': 'build',
                    'if': 'github.ref == \'refs/heads/main\'',
                    'environment': 'production',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up kubectl',
                            'uses': 'azure/setup-kubectl@v3'
                        },
                        {
                            'name': 'Set up Helm',
                            'uses': 'azure/setup-helm@v3'
                        },
                        {
                            'name': 'Deploy to production',
                            'run': '''
export KUBECONFIG=kubeconfig
helm upgrade --install milvus-production ./helm/milvus \\
  --namespace milvus-production \\
  --create-namespace \\
  --set image.tag=${{ needs.build.outputs.image-tag }} \\
  --set environment=production \\
  --values helm/values-production.yaml
'''
                        }
                    ]
                }
            }
        }
        
        # 워크플로우 파일 저장
        with open(self.cicd_dir / 'ci-cd.yml', 'w') as f:
            yaml.dump(main_workflow, f, default_flow_style=False, sort_keys=False)
        
        print("  ✅ GitHub Actions 메인 워크플로우 생성됨")
        
        # 추가 워크플로우들 생성
        self.create_additional_workflows()
    
    def create_additional_workflows(self):
        """추가 GitHub Actions 워크플로우 생성"""
        
        # 1. 릴리스 워크플로우
        release_workflow = {
            'name': 'Release',
            'on': {
                'push': {
                    'tags': ['v*']
                }
            },
            'jobs': {
                'release': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Create Release',
                            'uses': 'actions/create-release@v1',
                            'env': {
                                'GITHUB_TOKEN': '${{ secrets.GITHUB_TOKEN }}'
                            },
                            'with': {
                                'tag_name': '${{ github.ref }}',
                                'release_name': 'Release ${{ github.ref }}',
                                'draft': False,
                                'prerelease': False
                            }
                        }
                    ]
                }
            }
        }
        
        with open(self.cicd_dir / 'release.yml', 'w') as f:
            yaml.dump(release_workflow, f, default_flow_style=False)
        
        # 2. 보안 스캔 워크플로우
        security_workflow = {
            'name': 'Security Scan',
            'on': {
                'schedule': [{'cron': '0 6 * * 1'}],  # 매주 월요일 6AM
                'push': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'security': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Run Snyk to check for vulnerabilities',
                            'uses': 'snyk/actions/python@master',
                            'env': {
                                'SNYK_TOKEN': '${{ secrets.SNYK_TOKEN }}'
                            }
                        }
                    ]
                }
            }
        }
        
        with open(self.cicd_dir / 'security.yml', 'w') as f:
            yaml.dump(security_workflow, f, default_flow_style=False)
        
        print("  ✅ 추가 워크플로우 생성됨 (릴리스, 보안)")
    
    def create_gitlab_ci_pipeline(self):
        """GitLab CI 파이프라인 생성"""
        print("🦊 GitLab CI 파이프라인 생성 중...")
        
        gitlab_ci = {
            'image': 'python:3.11',
            'variables': {
                'DOCKER_HOST': 'tcp://docker:2376',
                'DOCKER_TLS_CERTDIR': '/certs',
                'DOCKER_TLS_VERIFY': '1',
                'DOCKER_CERT_PATH': '/certs/client',
                'PIP_CACHE_DIR': '$CI_PROJECT_DIR/.cache/pip'
            },
            'cache': {
                'paths': ['.cache/pip/', 'venv/']
            },
            'before_script': [
                'python -V',
                'pip install virtualenv',
                'virtualenv venv',
                'source venv/bin/activate',
                'pip install -r requirements.txt'
            ],
            'stages': ['test', 'security', 'build', 'deploy'],
            'test': {
                'stage': 'test',
                'script': [
                    'source venv/bin/activate',
                    'pip install pytest pytest-cov black flake8',
                    'black --check .',
                    'flake8 .',
                    'pytest --cov=. --cov-report=xml'
                ],
                'artifacts': {
                    'reports': {
                        'coverage_report': {
                            'coverage_format': 'cobertura',
                            'path': 'coverage.xml'
                        }
                    }
                },
                'coverage': '/TOTAL.+?(\\d+\\%)$/'
            },
            'security_scan': {
                'stage': 'security',
                'script': [
                    'pip install bandit safety',
                    'bandit -r . -f json -o bandit-report.json || true',
                    'safety check --json --output safety-report.json || true'
                ],
                'artifacts': {
                    'paths': ['*-report.json'],
                    'expire_in': '1 week'
                },
                'allow_failure': True
            },
            'build_image': {
                'stage': 'build',
                'image': 'docker:24.0.5',
                'services': ['docker:24.0.5-dind'],
                'variables': {
                    'IMAGE_TAG': '$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA'
                },
                'before_script': [
                    'docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY'
                ],
                'script': [
                    'docker build -t $IMAGE_TAG -f docker/Dockerfile .',
                    'docker push $IMAGE_TAG',
                    'docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest',
                    'docker push $CI_REGISTRY_IMAGE:latest'
                ],
                'only': ['main', 'develop']
            },
            'deploy_staging': {
                'stage': 'deploy',
                'image': 'bitnami/kubectl:latest',
                'before_script': [
                    'echo $KUBECONFIG_STAGING | base64 -d > kubeconfig',
                    'export KUBECONFIG=kubeconfig'
                ],
                'script': [
                    'kubectl set image deployment/milvus-app milvus-app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n milvus-staging',
                    'kubectl rollout status deployment/milvus-app -n milvus-staging'
                ],
                'environment': {
                    'name': 'staging',
                    'url': 'https://milvus-staging.example.com'
                },
                'only': ['develop']
            },
            'deploy_production': {
                'stage': 'deploy',
                'image': 'bitnami/kubectl:latest',
                'before_script': [
                    'echo $KUBECONFIG_PRODUCTION | base64 -d > kubeconfig',
                    'export KUBECONFIG=kubeconfig'
                ],
                'script': [
                    'kubectl set image deployment/milvus-app milvus-app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n milvus-production',
                    'kubectl rollout status deployment/milvus-app -n milvus-production'
                ],
                'environment': {
                    'name': 'production',
                    'url': 'https://milvus.example.com'
                },
                'when': 'manual',
                'only': ['main']
            }
        }
        
        with open('.gitlab-ci.yml', 'w') as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False)
        
        print("  ✅ GitLab CI 파이프라인 생성됨")
    
    def create_docker_files(self):
        """Docker 파일들 생성"""
        print("🐳 Docker 설정 파일 생성 중...")
        
        # Dockerfile
        dockerfile_content = '''# Multi-stage build for production
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r milvus && useradd -r -g milvus milvus

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/milvus/.local

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R milvus:milvus /app
USER milvus

# Make sure scripts in .local are usable
ENV PATH=/home/milvus/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(self.docker_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose for development
        docker_compose = {
            'version': '3.8',
            'services': {
                'app': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'docker/Dockerfile'
                    },
                    'ports': ['8000:8000'],
                    'environment': [
                        'MILVUS_HOST=milvus',
                        'ENVIRONMENT=development'
                    ],
                    'volumes': ['.:/app'],
                    'depends_on': ['milvus']
                },
                'milvus': {
                    'image': 'milvusdb/milvus:v2.4.0',
                    'command': 'milvus run standalone',
                    'ports': ['19530:19530', '9091:9091'],
                    'environment': [
                        'ETCD_ENDPOINTS=etcd:2379',
                        'MINIO_ADDRESS=minio:9000'
                    ],
                    'volumes': ['milvus_data:/var/lib/milvus'],
                    'depends_on': ['etcd', 'minio']
                },
                'etcd': {
                    'image': 'quay.io/coreos/etcd:v3.5.5',
                    'environment': [
                        'ETCD_AUTO_COMPACTION_MODE=revision',
                        'ETCD_AUTO_COMPACTION_RETENTION=1000',
                        'ETCD_QUOTA_BACKEND_BYTES=4294967296',
                        'ETCD_SNAPSHOT_COUNT=50000'
                    ],
                    'volumes': ['etcd_data:/etcd'],
                    'command': '''etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd'''
                },
                'minio': {
                    'image': 'minio/minio:RELEASE.2023-03-20T20-16-18Z',
                    'environment': [
                        'MINIO_ACCESS_KEY=minioadmin',
                        'MINIO_SECRET_KEY=minioadmin'
                    ],
                    'ports': ['9000:9000', '9001:9001'],
                    'volumes': ['minio_data:/data'],
                    'command': 'minio server /data --console-address ":9001"',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:9000/minio/health/live'],
                        'interval': '30s',
                        'timeout': '20s',
                        'retries': 3
                    }
                }
            },
            'volumes': {
                'milvus_data': None,
                'etcd_data': None,
                'minio_data': None
            }
        }
        
        with open(self.docker_dir / 'docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        # .dockerignore
        dockerignore_content = '''# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Documentation
docs/
*.md
README.md

# Development
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
docker-compose.yml
Dockerfile
.dockerignore

# CI/CD
.github/
.gitlab-ci.yml
'''
        
        with open(self.docker_dir / '.dockerignore', 'w') as f:
            f.write(dockerignore_content)
        
        print("  ✅ Docker 파일들 생성됨")
    
    def create_deployment_scripts(self):
        """배포 스크립트 생성"""
        print("📜 배포 스크립트 생성 중...")
        
        # 배포 스크립트
        deploy_script = '''#!/bin/bash
set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Default values
ENVIRONMENT="staging"
NAMESPACE="milvus-staging"
IMAGE_TAG="latest"
HELM_CHART="./helm/milvus"
DRY_RUN=false

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --environment ENVIRONMENT  Target environment (staging|production)"
    echo "  -n, --namespace NAMESPACE      Kubernetes namespace"
    echo "  -t, --tag TAG                  Docker image tag"
    echo "  -c, --chart CHART             Helm chart path"
    echo "  -d, --dry-run                  Perform a dry run"
    echo "  -h, --help                     Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -c|--chart)
            HELM_CHART="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option $1"
            usage
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo -e "${RED}Error: Environment must be 'staging' or 'production'${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 Starting deployment...${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Namespace: ${YELLOW}$NAMESPACE${NC}"
echo -e "Image Tag: ${YELLOW}$IMAGE_TAG${NC}"
echo -e "Helm Chart: ${YELLOW}$HELM_CHART${NC}"

# Check prerequisites
echo -e "${GREEN}🔍 Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found${NC}"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo -e "${RED}❌ helm not found${NC}"
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create namespace if it doesn't exist
echo -e "${GREEN}📁 Ensuring namespace exists...${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy with Helm
echo -e "${GREEN}🎯 Deploying application...${NC}"

HELM_COMMAND="helm upgrade --install milvus-$ENVIRONMENT $HELM_CHART \\
    --namespace $NAMESPACE \\
    --set image.tag=$IMAGE_TAG \\
    --set environment=$ENVIRONMENT \\
    --values helm/values-$ENVIRONMENT.yaml"

if [ "$DRY_RUN" = true ]; then
    HELM_COMMAND="$HELM_COMMAND --dry-run"
    echo -e "${YELLOW}🧪 Dry run mode enabled${NC}"
fi

echo "Executing: $HELM_COMMAND"
eval $HELM_COMMAND

if [ "$DRY_RUN" = false ]; then
    # Wait for deployment to be ready
    echo -e "${GREEN}⏳ Waiting for deployment to be ready...${NC}"
    kubectl rollout status deployment/milvus-standalone -n $NAMESPACE --timeout=300s
    
    # Run health checks
    echo -e "${GREEN}🏥 Running health checks...${NC}"
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc milvus-loadbalancer -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$SERVICE_IP" != "pending" ] && [ "$SERVICE_IP" != "" ]; then
        echo -e "${GREEN}✅ Service available at: $SERVICE_IP:19530${NC}"
    else
        echo -e "${YELLOW}⏳ LoadBalancer IP pending...${NC}"
    fi
    
    # Check pods
    echo -e "${GREEN}📦 Pod status:${NC}"
    kubectl get pods -n $NAMESPACE -l app=milvus-standalone
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
else
    echo -e "${GREEN}🧪 Dry run completed successfully!${NC}"
fi
'''
        
        with open(self.scripts_dir / 'deploy.sh', 'w') as f:
            f.write(deploy_script)
        
        # 롤백 스크립트
        rollback_script = '''#!/bin/bash
set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

ENVIRONMENT="staging"
NAMESPACE="milvus-staging"
REVISION=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --environment ENVIRONMENT  Target environment (staging|production)"
    echo "  -n, --namespace NAMESPACE      Kubernetes namespace"
    echo "  -r, --revision REVISION        Helm revision number (optional)"
    echo "  -h, --help                     Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--revision)
            REVISION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option $1"
            usage
            ;;
    esac
done

echo -e "${GREEN}🔄 Starting rollback...${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Namespace: ${YELLOW}$NAMESPACE${NC}"

# Show deployment history
echo -e "${GREEN}📜 Deployment history:${NC}"
helm history milvus-$ENVIRONMENT -n $NAMESPACE

if [ -z "$REVISION" ]; then
    echo -e "${YELLOW}No revision specified. Rolling back to previous version...${NC}"
    helm rollback milvus-$ENVIRONMENT -n $NAMESPACE
else
    echo -e "${YELLOW}Rolling back to revision $REVISION...${NC}"
    helm rollback milvus-$ENVIRONMENT $REVISION -n $NAMESPACE
fi

# Wait for rollback to complete
echo -e "${GREEN}⏳ Waiting for rollback to complete...${NC}"
kubectl rollout status deployment/milvus-standalone -n $NAMESPACE --timeout=300s

echo -e "${GREEN}🎉 Rollback completed successfully!${NC}"
'''
        
        with open(self.scripts_dir / 'rollback.sh', 'w') as f:
            f.write(rollback_script)
        
        # 스크립트 실행 권한 부여 (시뮬레이션)
        print("  💫 스크립트 실행 권한 설정 중...")
        print("    $ chmod +x scripts/deploy.sh")
        print("    $ chmod +x scripts/rollback.sh")
        
        print("  ✅ 배포 스크립트 생성됨")
    
    def create_helm_values(self):
        """Helm Values 파일들 생성"""
        print("⚙️  Helm Values 파일 생성 중...")
        
        helm_dir = Path("helm")
        helm_dir.mkdir(exist_ok=True)
        
        # Staging values
        staging_values = {
            'environment': 'staging',
            'replicaCount': 1,
            'image': {
                'repository': 'ghcr.io/your-org/milvus-app',
                'tag': 'latest',
                'pullPolicy': 'Always'
            },
            'service': {
                'type': 'ClusterIP',
                'port': 19530
            },
            'ingress': {
                'enabled': True,
                'className': 'nginx',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/'
                },
                'hosts': [{
                    'host': 'milvus-staging.example.com',
                    'paths': [{
                        'path': '/',
                        'pathType': 'Prefix'
                    }]
                }]
            },
            'resources': {
                'limits': {
                    'cpu': '2000m',
                    'memory': '4Gi'
                },
                'requests': {
                    'cpu': '1000m',
                    'memory': '2Gi'
                }
            },
            'autoscaling': {
                'enabled': True,
                'minReplicas': 1,
                'maxReplicas': 3,
                'targetCPUUtilizationPercentage': 70
            },
            'milvus': {
                'storage': {
                    'size': '100Gi'
                },
                'config': {
                    'log': {
                        'level': 'debug'
                    }
                }
            }
        }
        
        # Production values
        production_values = {
            'environment': 'production',
            'replicaCount': 3,
            'image': {
                'repository': 'ghcr.io/your-org/milvus-app',
                'tag': 'stable',
                'pullPolicy': 'IfNotPresent'
            },
            'service': {
                'type': 'LoadBalancer',
                'port': 19530,
                'annotations': {
                    'service.beta.kubernetes.io/aws-load-balancer-type': 'nlb'
                }
            },
            'ingress': {
                'enabled': True,
                'className': 'nginx',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                },
                'hosts': [{
                    'host': 'milvus.example.com',
                    'paths': [{
                        'path': '/',
                        'pathType': 'Prefix'
                    }]
                }],
                'tls': [{
                    'secretName': 'milvus-tls',
                    'hosts': ['milvus.example.com']
                }]
            },
            'resources': {
                'limits': {
                    'cpu': '4000m',
                    'memory': '16Gi'
                },
                'requests': {
                    'cpu': '2000m',
                    'memory': '8Gi'
                }
            },
            'autoscaling': {
                'enabled': True,
                'minReplicas': 3,
                'maxReplicas': 10,
                'targetCPUUtilizationPercentage': 70,
                'targetMemoryUtilizationPercentage': 80
            },
            'persistence': {
                'enabled': True,
                'storageClass': 'gp3-ssd',
                'size': '500Gi'
            },
            'milvus': {
                'storage': {
                    'size': '1Ti'
                },
                'backup': {
                    'enabled': True,
                    'schedule': '0 2 * * *'
                },
                'config': {
                    'log': {
                        'level': 'info'
                    }
                }
            },
            'monitoring': {
                'enabled': True,
                'serviceMonitor': {
                    'enabled': True
                }
            },
            'security': {
                'networkPolicy': {
                    'enabled': True
                },
                'podSecurityContext': {
                    'fsGroup': 1000,
                    'runAsNonRoot': True,
                    'runAsUser': 1000
                }
            }
        }
        
        # Values 파일 저장
        with open(helm_dir / 'values-staging.yaml', 'w') as f:
            yaml.dump(staging_values, f, default_flow_style=False)
        
        with open(helm_dir / 'values-production.yaml', 'w') as f:
            yaml.dump(production_values, f, default_flow_style=False)
        
        print("  ✅ Helm Values 파일 생성됨 (staging, production)")
    
    def create_testing_pipeline(self):
        """테스팅 파이프라인 설정"""
        print("🧪 테스팅 파이프라인 설정 중...")
        
        # pytest configuration
        pytest_config = '''[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "e2e: marks tests as end-to-end tests"
]

[tool.coverage.run]
source = ["."]
omit = [
    "*/venv/*",
    "*/tests/*",
    "*/test_*",
    "setup.py",
    "*/migrations/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]

[tool.black]
line-length = 88
target-version = ['py39']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''
'''
        
        with open('pyproject.toml', 'w') as f:
            f.write(pytest_config)
        
        # GitHub Actions test workflow
        test_workflow = {
            'name': 'Tests',
            'on': ['push', 'pull_request'],
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.9', '3.10', '3.11']
                        }
                    },
                    'services': {
                        'milvus': {
                            'image': 'milvusdb/milvus:v2.4.0',
                            'ports': ['19530:19530'],
                            'options': '--health-cmd "curl -f http://localhost:9091/healthz" --health-interval 30s --health-timeout 20s --health-retries 3'
                        }
                    },
                    'steps': [
                        {
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python ${{ matrix.python-version }}',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov pytest-xdist
'''
                        },
                        {
                            'name': 'Run unit tests',
                            'run': 'pytest tests/unit -v --cov=. --cov-report=xml'
                        },
                        {
                            'name': 'Run integration tests',
                            'run': 'pytest tests/integration -v',
                            'env': {
                                'MILVUS_HOST': 'localhost',
                                'MILVUS_PORT': '19530'
                            }
                        }
                    ]
                }
            }
        }
        
        with open(self.cicd_dir / 'tests.yml', 'w') as f:
            yaml.dump(test_workflow, f, default_flow_style=False)
        
        print("  ✅ 테스팅 파이프라인 설정 완료")
    
    def demonstrate_pipeline_operations(self):
        """파이프라인 운영 시뮬레이션"""
        print("\n🎮 CI/CD 파이프라인 운영 시뮬레이션...")
        
        # Git 워크플로우 시뮬레이션
        print("\n📋 Git 워크플로우:")
        git_commands = [
            "git checkout -b feature/new-search-algorithm",
            "git add .",
            "git commit -m 'feat: implement new vector search algorithm'",
            "git push origin feature/new-search-algorithm",
            "# Pull Request 생성 → CI 파이프라인 자동 실행",
            "git checkout main",
            "git merge feature/new-search-algorithm",
            "git tag v1.2.0",
            "git push origin v1.2.0 # 프로덕션 배포 트리거"
        ]
        
        for cmd in git_commands:
            if cmd.startswith('#'):
                print(f"  {cmd}")
            else:
                print(f"  $ {cmd}")
            time.sleep(0.3)
        
        # CI/CD 실행 시뮬레이션
        print(f"\n🔄 CI/CD 파이프라인 실행 결과:")
        pipeline_steps = [
            ("✅ 코드 체크아웃", "0.5초"),
            ("✅ 의존성 설치", "45초"),
            ("✅ 린팅 검사", "15초"),
            ("✅ 단위 테스트", "120초"),
            ("✅ 통합 테스트", "180초"),
            ("✅ 보안 스캔", "90초"),
            ("✅ Docker 이미지 빌드", "300초"),
            ("✅ 스테이징 배포", "120초"),
            ("⏳ 프로덕션 배포 대기", "수동 승인 필요")
        ]
        
        for step, duration in pipeline_steps:
            print(f"  {step} ({duration})")
            time.sleep(0.2)
        
        # 배포 명령어 예시
        print(f"\n🚀 배포 명령어 예시:")
        deploy_commands = [
            "# 스테이징 배포",
            "./scripts/deploy.sh --environment staging --tag v1.2.0",
            "",
            "# 프로덕션 배포",
            "./scripts/deploy.sh --environment production --tag v1.2.0",
            "",
            "# 롤백",
            "./scripts/rollback.sh --environment production --revision 5"
        ]
        
        for cmd in deploy_commands:
            if cmd.startswith('#') or cmd == '':
                print(f"  {cmd}")
            else:
                print(f"  $ {cmd}")

def main():
    """메인 실행 함수"""
    print("🔄 Milvus CI/CD 파이프라인 구축")
    print("=" * 80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = CICDPipelineManager()
    
    try:
        # 1. GitHub Actions 워크플로우 생성
        print("\n" + "=" * 80)
        print(" 🐙 GitHub Actions 설정")
        print("=" * 80)
        manager.create_github_actions_workflow()
        
        # 2. GitLab CI 파이프라인 생성
        print("\n" + "=" * 80)
        print(" 🦊 GitLab CI 설정")
        print("=" * 80)
        manager.create_gitlab_ci_pipeline()
        
        # 3. Docker 파일 생성
        print("\n" + "=" * 80)
        print(" 🐳 Docker 설정")
        print("=" * 80)
        manager.create_docker_files()
        
        # 4. 배포 스크립트 생성
        print("\n" + "=" * 80)
        print(" 📜 배포 스크립트")
        print("=" * 80)
        manager.create_deployment_scripts()
        
        # 5. Helm Values 생성
        print("\n" + "=" * 80)
        print(" ⚙️  Helm 설정")
        print("=" * 80)
        manager.create_helm_values()
        
        # 6. 테스팅 파이프라인
        print("\n" + "=" * 80)
        print(" 🧪 테스팅 설정")
        print("=" * 80)
        manager.create_testing_pipeline()
        
        # 7. 파이프라인 운영 시뮬레이션
        print("\n" + "=" * 80)
        print(" 🎮 파이프라인 운영")
        print("=" * 80)
        manager.demonstrate_pipeline_operations()
        
        # 8. 요약
        print("\n" + "=" * 80)
        print(" 📊 CI/CD 설정 완료")
        print("=" * 80)
        
        print("✅ 생성된 파일들:")
        created_files = [
            ".github/workflows/ci-cd.yml",
            ".github/workflows/release.yml",
            ".github/workflows/security.yml",
            ".github/workflows/tests.yml",
            ".gitlab-ci.yml",
            "docker/Dockerfile",
            "docker/docker-compose.yml",
            "docker/.dockerignore",
            "scripts/deploy.sh",
            "scripts/rollback.sh",
            "helm/values-staging.yaml",
            "helm/values-production.yaml",
            "pyproject.toml"
        ]
        
        for file in created_files:
            print(f"  📄 {file}")
        
        print("\n💡 주요 기능:")
        features = [
            "✅ 자동화된 테스트 (단위, 통합, E2E)",
            "✅ 코드 품질 검사 (린팅, 포맷팅)",
            "✅ 보안 스캔 (Bandit, Safety, Snyk)",
            "✅ Docker 이미지 빌드 및 푸시",
            "✅ 멀티 환경 배포 (스테이징, 프로덕션)",
            "✅ 롤백 및 버전 관리",
            "✅ 모니터링 및 알림"
        ]
        
        for feature in features:
            print(f"  {feature}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 CI/CD 파이프라인 실습 완료!")
    print("\n💡 학습 포인트:")
    print("  • GitHub Actions/GitLab CI 워크플로우 작성")
    print("  • Docker 이미지 빌드 및 배포 자동화")
    print("  • 멀티 환경 배포 전략")
    print("  • 자동화된 테스트 및 보안 검사")
    
    print("\n🚀 다음 단계:")
    print("  python step05_production/03_blue_green_deployment.py")

if __name__ == "__main__":
    main() 