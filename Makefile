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
build: build-rollout-controller ## Build lightweight binary (no k8s deps)

.PHONY: build-rollout-controller
build-rollout-controller: ## Build the rollout controller (no k8s discovery)
	go build -ldflags "$(LDFLAGS)" -o bin/rollout-controller ./cmd/rollout-controller

.PHONY: build-rollout-controller-k8s
build-rollout-controller-k8s: ## Build the rollout controller with Kubernetes pod discovery
	go build -tags k8s -ldflags "$(LDFLAGS)" -o bin/rollout-controller-k8s ./cmd/rollout-controller

.PHONY: build-all
build-all: build-rollout-controller build-rollout-controller-k8s ## Build both variants

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

##@ Help

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
