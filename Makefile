PROJECT_NAME ?= llm-d-rl
REGISTRY ?= ghcr.io/llm-d
IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)
PY_IS_IMAGE ?= $(IMAGE)-py-is
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS ?= -s -w -X main.version=$(VERSION)

GOLANGCI_LINT_VERSION ?= v2.8.0

.PHONY: all
all: build

##@ Build

.PHONY: build
build: ## Build the rollout controller (no k8s discovery)
	go build -ldflags "$(LDFLAGS)" -o bin/rollout-controller ./cmd/rollout-controller

.PHONY: build-k8s
build-k8s: ## Build the rollout controller with Kubernetes pod discovery
	go build -tags k8s -ldflags "$(LDFLAGS)" -o bin/rollout-controller-k8s ./cmd/rollout-controller

.PHONY: build-all
build-all: build build-k8s ## Build both variants

##@ Development

.PHONY: test
test: ## Run tests
	go test ./... -v -race -count=1

.PHONY: lint
lint: ## Run linter
	golangci-lint run ./...

.PHONY: fmt
fmt: ## Format code
	go fmt ./...

.PHONY: generate
generate: ## Generate protobuf and deepcopy code
	protoc --go_out=. --go-grpc_out=. api/v1alpha1/rollout.proto

##@ Container

.PHONY: docker-build
docker-build: ## Build rollout controller container image
	docker build -t $(IMAGE):$(VERSION) .

.PHONY: docker-push
docker-push: ## Push rollout controller container image
	docker push $(IMAGE):$(VERSION)

.PHONY: docker-build-py-is
docker-build-py-is: ## Build py-inference-scheduler proxy image
	docker build -f integration/py-is/Dockerfile -t $(PY_IS_IMAGE):$(VERSION) .

.PHONY: docker-push-py-is
docker-push-py-is: ## Push py-inference-scheduler proxy image
	docker push $(PY_IS_IMAGE):$(VERSION)

.PHONY: docker-build-all
docker-build-all: docker-build docker-build-py-is ## Build all container images

.PHONY: docker-push-all
docker-push-all: docker-push docker-push-py-is ## Push all container images

##@ Deploy

CONFIGMAP_OUT ?= deploy/cks/nccl-trainer-configmap.yaml
PROMPTS_CONFIGMAP_OUT ?= deploy/cks/nccl-trainer-prompts-configmap.yaml

.PHONY: generate-configmaps
generate-configmaps: ## Generate ConfigMap YAMLs from python/ sources (requires kubectl)
	kubectl create configmap nccl-trainer-script \
		--from-file=nccl_weight_trainer.py=python/nccl_weight_trainer.py \
		--namespace=llm-d-rl \
		--dry-run=client -o yaml > $(CONFIGMAP_OUT)
	kubectl create configmap nccl-trainer-text-prompts \
		--from-file=prompts.txt=python/prompts.txt \
		--namespace=llm-d-rl \
		--dry-run=client -o yaml > $(PROMPTS_CONFIGMAP_OUT)

##@ Help

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
