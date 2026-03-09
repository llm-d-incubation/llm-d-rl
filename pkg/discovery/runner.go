package discovery

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// Config holds all Kubernetes-specific configuration for engine pod discovery.
// Call BindFlags to register CLI flags onto a FlagSet, then Start to begin
// watching. All k8s imports are contained here so callers (e.g. main.go) stay
// free of k8s dependencies.
type Config struct {
	Selector   string
	Namespace  string
	Port       int
	Kubeconfig string
}

// BindFlags registers the four discovery flags on fs.
func (c *Config) BindFlags(fs *flag.FlagSet) {
	fs.StringVar(&c.Selector, "engine-selector", "",
		"Kubernetes label selector for vLLM engine pods (e.g., llm-d-role=rollout-engine). Takes precedence over --engines.")
	fs.IntVar(&c.Port, "engine-port", 8000,
		"vLLM HTTP port on engine pods (used with --engine-selector)")
	fs.StringVar(&c.Namespace, "namespace", "",
		"Kubernetes namespace to watch for engine pods. Defaults to NAMESPACE env var, then 'default'.")
	fs.StringVar(&c.Kubeconfig, "kubeconfig", "",
		"Path to kubeconfig file. Defaults to in-cluster config when empty.")
}

// Enabled reports whether pod discovery is configured (i.e. --engine-selector was set).
func (c *Config) Enabled() bool {
	return c.Selector != ""
}

// Start builds a Kubernetes client, creates a PodWatcher for the configured
// selector, and runs it in a background goroutine until ctx is cancelled.
// The initial cache sync happens inside the goroutine; OnReady will fire for
// all already-running matching pods before the watcher blocks on ctx.Done().
func (c *Config) Start(ctx context.Context, callbacks EngineCallbacks) error {
	client, err := buildKubeClient(c.Kubeconfig)
	if err != nil {
		return fmt.Errorf("build kubernetes client: %w", err)
	}

	ns := resolveNamespace(c.Namespace)
	log.Printf("engine discovery: selector=%q namespace=%s port=%d", c.Selector, namespaceLabel(ns), c.Port)

	watcher, err := NewPodWatcher(client, ns, c.Selector, c.Port, callbacks)
	if err != nil {
		return fmt.Errorf("create pod watcher: %w", err)
	}

	go watcher.Run(ctx)
	return nil
}

// buildKubeClient returns an in-cluster client when kubeconfigPath is empty,
// falling back to ~/.kube/config for local development.
func buildKubeClient(kubeconfigPath string) (kubernetes.Interface, error) {
	var cfg *rest.Config
	var err error

	if kubeconfigPath != "" {
		cfg, err = clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	} else {
		cfg, err = rest.InClusterConfig()
		if err != nil {
			cfg, err = clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("build kube config: %w", err)
	}

	return kubernetes.NewForConfig(cfg)
}

// resolveNamespace resolves the effective namespace: flag value → NAMESPACE env → "default".
func resolveNamespace(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	if ns := os.Getenv("NAMESPACE"); ns != "" {
		return ns
	}
	return "default"
}

// namespaceLabel returns a human-readable label for the namespace (used in log lines).
func namespaceLabel(ns string) string {
	if ns == "" {
		return "<all>"
	}
	return ns
}
