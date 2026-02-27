FROM golang:1.23-alpine AS builder
RUN apk add --no-cache git
WORKDIR /src
COPY go.mod ./
COPY go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags "-s -w -X main.version=$(git describe --tags --always --dirty 2>/dev/null || echo dev)" \
    -o /rollout-controller ./cmd/rollout-controller

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /rollout-controller /rollout-controller
EXPOSE 8090
ENTRYPOINT ["/rollout-controller"]
