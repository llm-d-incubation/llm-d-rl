// Command rollout-controller starts the llm-d RL rollout controller.
//
// The rollout controller exposes an HTTP/gRPC API that RL training frameworks
// call to manage inference engine pools for generation rollouts. It orchestrates
// weight synchronization, engine lifecycle, and request routing through the
// llm-d inference scheduler.
//
// Usage:
//
//	rollout-controller \
//	  --port=8090 \
//	  --engines=http://localhost:8000 \
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
		engines             = flag.String("engines", "", "Comma-separated list of engine URLs (e.g., http://localhost:8000,http://localhost:8001)")
		simulateLifecycle   = flag.Bool("simulate-lifecycle", false, "No-op lifecycle operations (pause, sleep, weight sync) for GPU-free demos with llm-d-inference-sim")
		showVersion         = flag.Bool("version", false, "Print version and exit")
	)
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		os.Exit(0)
	}

	log.Printf("llm-d rollout controller %s starting on port %d", version, *port)

	// Initialize components
	coordinator := weightsync.NewCoordinator()

	poolManager := lifecycle.NewPoolManager(lifecycle.PoolManagerConfig{
		HealthCheckInterval: *healthCheckInterval,
		OnEngineUnhealthy: func(id string, err error) {
			log.Printf("WARNING: engine %s is unhealthy: %v", id, err)
		},
	})

	// Register engines from --engines flag
	if *engines != "" {
		for i, rawURL := range strings.Split(*engines, ",") {
			url := strings.TrimSpace(rawURL)
			if url == "" {
				continue
			}
			id := fmt.Sprintf("engine-%d", i)

			var client weightsync.EngineClient
			if *simulateLifecycle {
				client = weightsync.NewSimulatedEngineClient(url)
				log.Printf("Registered engine %s at %s (simulated lifecycle)", id, url)
			} else {
				client = weightsync.NewVLLMClient(url)
				log.Printf("Registered engine %s at %s", id, url)
			}

			poolManager.AddEngine(lifecycle.EngineInfo{
				ID:      id,
				Address: url,
				Client:  client,
			})
			coordinator.RegisterEngine(id, client)
		}
	}

	server := rollout.NewServer(poolManager, coordinator)

	// Start health check loop
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go poolManager.StartHealthLoop(ctx)

	// Run an initial health check so engines are marked ready before
	// the first request arrives.
	poolManager.RunHealthChecks(ctx)

	// Start HTTP server
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *port),
		Handler: server.Handler(),
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down...")
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		httpServer.Shutdown(shutdownCtx)
	}()

	log.Printf("Rollout controller listening on :%d", *port)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("HTTP server error: %v", err)
	}
}
