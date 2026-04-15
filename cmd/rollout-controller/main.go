// Command rollout-controller starts the llm-d RL rollout controller.
//
// The rollout controller exposes an HTTP API that RL training frameworks call
// to manage inference engine pools for generation rollouts. It orchestrates
// weight synchronization, engine lifecycle, and request routing.
//
// Pool types (select via --router-url):
//
//	RouterPool (--router-url set): generation requests go through the inference
//	  router gateway (EPP/Envoy). Engine pods are still discovered for weight sync.
//
//	DirectPool (no --router-url): generation requests are dispatched directly to
//	  a ready engine picked from the pool.
//
// Pool population (select via --engine-selector or --engines):
//
//	K8s discovery (--engine-selector): pods matching the label selector are
//	  added/removed automatically as they become Ready or are deleted.
//
//	Static list (--engines): a fixed comma-separated list of engine URLs.
//
// Usage (llm-d inference stack):
//
//	rollout-controller \
//	  --engine-selector llm-d.ai/inference-serving=true \
//	  --router-url http://llm-d-inference-gateway-istio.llm-d-rl.svc.cluster.local:80
//
// Usage (local/GPU-free demo):
//
//	rollout-controller --engines http://localhost:8000 --simulate-lifecycle
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

	"github.com/llm-d/llm-d-rl/pkg/pool"
	"github.com/llm-d/llm-d-rl/pkg/rollout"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

var version = "dev"

func main() {
	var (
		port                = flag.Int("port", 8090, "HTTP server port")
		healthCheckInterval = flag.Duration("health-check-interval", 30*time.Second, "Interval between engine health checks")
		simulateLifecycle   = flag.Bool("simulate-lifecycle", false, "No-op lifecycle operations for GPU-free demos")
		showVersion         = flag.Bool("version", false, "Print version and exit")

		// Pool population flags.
		engines = flag.String("engines", "", "Comma-separated static engine URLs (e.g., http://localhost:8000). Used when --engine-selector is not set.")

		// Pool type flags.
		routerURL = flag.String("router-url", "", "HTTP URL of the inference router/EPP gateway. When set, generation requests go through this URL. When empty, requests go directly to a ready engine.")
		tokensIn  = flag.Bool("tokens-in", false, "Pass token-ID arrays as prompt input. Default (false) sends text strings, required when routing via EPP/gateway.")
	)

	// Kubernetes discovery flags are registered only when built with -tags k8s.
	discoveryCfg := newDiscoveryConfig()
	bindDiscoveryFlags(discoveryCfg, flag.CommandLine)

	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		os.Exit(0)
	}

	log.Printf("llm-d rollout controller %s starting on port %d", version, *port)

	// --- Build engine client factory ---
	newClient := func(address string) weightsync.EngineClient {
		if *simulateLifecycle {
			return weightsync.NewSimulatedEngineClient(address)
		}
		return weightsync.NewVLLMClient(address)
	}

	// --- Create pool ---
	p := pool.NewEnginePool(pool.Config{
		BaseConfig: pool.BaseConfig{
			ClientFactory:       newClient,
			HealthCheckInterval: *healthCheckInterval,
			OnEngineUnhealthy: func(id string, err error) {
				log.Printf("WARNING: engine %s is unhealthy: %v", id, err)
			},
		},
		RouterURL: *routerURL,
		TokensIn:  *tokensIn,
	})
	log.Printf("pool: router-url=%q tokens-in=%v", *routerURL, *tokensIn)

	// --- Create and start populator ---
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if discoveryEnabled(discoveryCfg) {
		// Kubernetes pod-selector discovery.
		k8sPop := &dynamicDiscoveryAdaptor{cfg: discoveryCfg}
		if err := k8sPop.Start(ctx, p); err != nil {
			log.Fatalf("engine discovery: %v", err)
		}
	} else if *engines != "" {
		// Static engine list.
		for i, raw := range strings.Split(*engines, ",") {
			url := strings.TrimSpace(raw)
			if url == "" {
				continue
			}
			p.AddEngine(fmt.Sprintf("engine-%d", i), url)
		}
	} else {
		log.Printf("WARNING: no engine source configured — set --engine-selector or --engines")
	}

	// --- Start health loop and server ---
	go p.StartHealthLoop(ctx)

	server := rollout.NewServer(p)
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

// dynamicDiscoveryAdaptor bridges the build-tag-gated startDiscovery function
// to the pool. The actual discovery logic lives in discovery_k8s.go (built with
// -tags k8s) or discovery_nok8s.go (stub).
type dynamicDiscoveryAdaptor struct {
	cfg *discoveryConfig
}

func (k *dynamicDiscoveryAdaptor) Start(ctx context.Context, p pool.Pool) error {
	return startDiscovery(ctx, k.cfg, engineCallbacks{
		OnReady: func(id, address string) { p.AddEngine(id, address) },
		OnGone:  func(id string) { p.RemoveEngine(id) },
	})
}

