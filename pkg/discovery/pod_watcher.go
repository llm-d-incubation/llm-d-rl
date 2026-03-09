// Package discovery implements dynamic engine discovery via Kubernetes pod selectors.
// Rather than requiring a static list of vLLM engine addresses, the PodWatcher
// watches Kubernetes pods matching a label selector and calls the provided
// callbacks as pods become ready or are removed.
//
// This allows the rollout controller to discover engines automatically:
//
//  1. Label vLLM pods at deploy time (e.g., llm-d-role=rollout-engine)
//  2. Pass that selector to the rollout controller via --engine-selector
//  3. The watcher tracks pod readiness and keeps the engine pool in sync
//
// Weight sync operations use the pod's IP address directly. The EPP continues
// to handle inference routing independently via its own pod watch.
package discovery

import (
	"context"
	"fmt"
	"log"
	"sync"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

// EngineCallbacks are invoked as pods enter and leave the ready state.
// Both callbacks must be safe to call concurrently.
type EngineCallbacks struct {
	// OnReady is called when a pod becomes Ready and has a PodIP.
	// id is the pod's UID (stable across restarts of the informer).
	// address is the HTTP base URL: "http://{podIP}:{port}".
	OnReady func(id, address string)

	// OnGone is called when a pod is deleted or transitions to NotReady.
	OnGone func(id string)
}

// PodWatcher watches Kubernetes pods matching a label selector and fires
// EngineCallbacks as pods become ready or are removed.
type PodWatcher struct {
	client    kubernetes.Interface
	namespace string
	selector  labels.Selector
	port      int
	callbacks EngineCallbacks

	// mu protects registered to prevent duplicate OnReady / spurious OnGone calls.
	mu         sync.Mutex
	registered map[string]string // podUID → address
}

// NewPodWatcher creates a PodWatcher for pods matching selectorStr in namespace.
// Use an empty namespace to watch across all namespaces.
// port is the vLLM HTTP port on each pod (typically 8000).
func NewPodWatcher(
	client kubernetes.Interface,
	namespace string,
	selectorStr string,
	port int,
	callbacks EngineCallbacks,
) (*PodWatcher, error) {
	sel, err := labels.Parse(selectorStr)
	if err != nil {
		return nil, fmt.Errorf("parse engine selector %q: %w", selectorStr, err)
	}
	return &PodWatcher{
		client:     client,
		namespace:  namespace,
		selector:   sel,
		port:       port,
		callbacks:  callbacks,
		registered: make(map[string]string),
	}, nil
}

// Run starts the informer and blocks until ctx is cancelled.
// It performs an initial list-and-watch, so pods that are already running
// and ready when Run is called will trigger OnReady before returning.
func (pw *PodWatcher) Run(ctx context.Context) {
	factory := informers.NewSharedInformerFactoryWithOptions(
		pw.client,
		0, // no resync period; rely on watch events
		informers.WithNamespace(pw.namespace),
		informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
			opts.LabelSelector = pw.selector.String()
		}),
	)

	podInformer := factory.Core().V1().Pods().Informer()

	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if pod, ok := obj.(*corev1.Pod); ok {
				pw.sync(pod)
			}
		},
		UpdateFunc: func(_, newObj interface{}) {
			if pod, ok := newObj.(*corev1.Pod); ok {
				pw.sync(pod)
			}
		},
		DeleteFunc: func(obj interface{}) {
			pod, ok := obj.(*corev1.Pod)
			if !ok {
				// Handle tombstone from the informer cache.
				if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
					pod, ok = d.Obj.(*corev1.Pod)
					if !ok {
						return
					}
				} else {
					return
				}
			}
			pw.markGone(string(pod.UID))
		},
	})

	factory.Start(ctx.Done())

	log.Printf("discovery: waiting for pod cache sync (selector=%s, namespace=%s)", pw.selector, namespaceName(pw.namespace))
	if !cache.WaitForCacheSync(ctx.Done(), podInformer.HasSynced) {
		log.Printf("discovery: cache sync interrupted")
		return
	}
	log.Printf("discovery: pod cache synced")

	<-ctx.Done()
}

// sync is called on every Add or Update event. It registers the pod if it
// became ready, or deregisters it if it is no longer ready.
func (pw *PodWatcher) sync(pod *corev1.Pod) {
	id := string(pod.UID)
	ready := isPodReady(pod) && pod.Status.PodIP != ""

	pw.mu.Lock()
	defer pw.mu.Unlock()

	existing, wasRegistered := pw.registered[id]

	if ready {
		addr := fmt.Sprintf("http://%s:%d", pod.Status.PodIP, pw.port)
		if wasRegistered && existing == addr {
			return // no change
		}
		if wasRegistered {
			// Address changed (e.g., pod restarted with a new IP); remove old first.
			pw.callbacks.OnGone(id)
		}
		pw.registered[id] = addr
		log.Printf("discovery: engine ready pod=%s/%s id=%s addr=%s", pod.Namespace, pod.Name, id, addr)
		pw.callbacks.OnReady(id, addr)
		return
	}

	if wasRegistered {
		delete(pw.registered, id)
		log.Printf("discovery: engine gone pod=%s/%s id=%s", pod.Namespace, pod.Name, id)
		pw.callbacks.OnGone(id)
	}
}

// markGone deregisters a pod by UID if it was registered.
func (pw *PodWatcher) markGone(id string) {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	if _, ok := pw.registered[id]; !ok {
		return
	}
	delete(pw.registered, id)
	log.Printf("discovery: engine deleted id=%s", id)
	pw.callbacks.OnGone(id)
}

// isPodReady returns true when the pod has a Ready condition with status True.
func isPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady {
			return cond.Status == corev1.ConditionTrue
		}
	}
	return false
}

func namespaceName(ns string) string {
	if ns == "" {
		return "<all>"
	}
	return ns
}
