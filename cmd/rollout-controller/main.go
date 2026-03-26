// Command rollout-controller starts the llm-d RL rollout controller.
//
// The rollout controller exposes an HTTP/gRPC API that RL training frameworks
// call to manage inference engine pools for generation rollouts. It orchestrates
// weight synchronization, engine lifecycle, and request routing through the
// llm-d inference scheduler.
//
// Engine discovery is driven by a Kubernetes pod label selector. Pods matching
// the selector are automatically added to the weight-sync pool when they become
// Ready, and removed when they are deleted or become NotReady. This removes the
// need to pass engine IPs as flags — just label your vLLM pods at deploy time.
//
// Usage (Kubernetes):
//
//	rollout-controller \
//	  --engine-selector llm-d-role=rollout-engine \
//	  --engine-port 8000 \
//	  --router-url http://envoy-gateway.llm-d.svc.cluster.local:80
//
// Usage (local/GPU-free demo):
//
//	rollout-controller \
//	  --engines http://localhost:8000 \
//	  --simulate-lifecycle
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/llm-d/llm-d-rl/pkg/lifecycle"
	"github.com/llm-d/llm-d-rl/pkg/rollout"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

var version = "dev"

func main() {
	var (
		port                = flag.Int("port", 8090, "HTTP server port")
		healthCheckInterval = flag.Duration("health-check-interval", 30*time.Second, "Interval between engine health checks")
		simulateLifecycle   = flag.Bool("simulate-lifecycle", false, "No-op lifecycle operations (pause, sleep, weight sync) for GPU-free demos")
		showVersion         = flag.Bool("version", false, "Print version and exit")

		// Static engine list (local development / GPU-free demos).
		engines = flag.String("engines", "", "Comma-separated engine URLs (e.g., http://localhost:8000). Used when --engine-selector is not set.")

		// Inference routing via router/gateway (optional).
		routerURL = flag.String("router-url", "", "HTTP URL of the inference router gateway for prefix-cache-aware routing (e.g., http://llm-d-inference-gateway:80). If unset, requests are dispatched directly to engines.")
	)

	// Kubernetes discovery flags are registered only when built with -tags k8s.
	cfg := newDiscoveryConfig()
	bindDiscoveryFlags(cfg, flag.CommandLine)

	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		os.Exit(0)
	}

	log.Printf("llm-d rollout controller %s starting on port %d", version, *port)

	coordinator := weightsync.NewCoordinator()
	poolManager := lifecycle.NewPoolManager(lifecycle.PoolManagerConfig{
		HealthCheckInterval: *healthCheckInterval,
		OnEngineUnhealthy: func(id string, err error) {
			log.Printf("WARNING: engine %s is unhealthy: %v", id, err)
		},
	})

	// newEngineClient is a helper used by both discovery paths to build the
	// right client based on --simulate-lifecycle.
	newEngineClient := func(address string) weightsync.EngineClient {
		if *simulateLifecycle {
			return weightsync.NewSimulatedEngineClient(address)
		}
		return weightsync.NewVLLMClient(address)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if discoveryEnabled(cfg) {
		// --- Kubernetes pod-selector-based discovery ---
		err := startDiscovery(ctx, cfg, engineCallbacks{
			OnReady: func(id, address string) {
				client := newEngineClient(address)
				poolManager.AddEngine(lifecycle.EngineInfo{ID: id, Address: address, Client: client})
				coordinator.RegisterEngine(id, client)
				log.Printf("registered engine id=%s addr=%s", id, address)
			},
			OnGone: func(id string) {
				poolManager.RemoveEngine(id)
				coordinator.UnregisterEngine(id)
				log.Printf("deregistered engine id=%s", id)
			},
		})
		if err != nil {
			log.Fatalf("engine discovery: %v", err)
		}

	} else if *engines != "" {
		// --- Static engine list (local dev / GPU-free demo) ---
		for i, rawURL := range strings.Split(*engines, ",") {
			url := strings.TrimSpace(rawURL)
			if url == "" {
				continue
			}
			id := fmt.Sprintf("engine-%d", i)
			client := newEngineClient(url)
			poolManager.AddEngine(lifecycle.EngineInfo{ID: id, Address: url, Client: client})
			coordinator.RegisterEngine(id, client)
			log.Printf("registered engine %s at %s", id, url)
		}

	} else {
		log.Printf("WARNING: no engine source configured — set --engine-selector or --engines")
	}

	if *routerURL != "" {
		log.Printf("inference routing: via router gateway at %s", *routerURL)
	} else {
		log.Printf("inference routing: direct engine dispatch (no router)")
	}

	server := rollout.NewServer(poolManager, coordinator, *routerURL)

	go poolManager.StartHealthLoop(ctx)
	poolManager.RunHealthChecks(ctx)

	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *port),
		Handler: server.Handler(),
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("shutting down...")
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		httpServer.Shutdown(shutdownCtx)
	}()

	log.Printf("rollout controller listening on :%d", *port)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server error: %v", err)
	}
}
