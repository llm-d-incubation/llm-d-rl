PROJECT_NAME ?= llm-d-rl
REGISTRY ?= ghcr.io/llm-d
IMAGE ?= $(REGISTRY)/$(PROJECT_NAME)
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
docker-build: ## Build container image
	docker build -t $(IMAGE):$(VERSION) .

.PHONY: docker-push
docker-push: ## Push container image
	docker push $(IMAGE):$(VERSION)

##@ Deploy

CONFIGMAP_OUT ?= deploy/cks/nccl-trainer-configmap.yaml

.PHONY: generate-configmaps
generate-configmaps: ## Generate ConfigMap YAML from python/nccl_weight_trainer.py (requires kubectl)
	kubectl create configmap nccl-trainer-script \
		--from-file=nccl_weight_trainer.py=python/nccl_weight_trainer.py \
		--namespace=llm-d-rl \
		--dry-run=client -o yaml > $(CONFIGMAP_OUT)

##@ Help

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
