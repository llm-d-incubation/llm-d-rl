package discovery

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// makePod builds a minimal Pod for use in tests.
func makePod(uid, ip string, ready bool) *corev1.Pod {
	pod := &corev1.Pod{}
	pod.UID = types.UID(uid)
	pod.Namespace = "default"
	pod.Name = uid
	pod.Status.PodIP = ip
	if ready {
		pod.Status.Conditions = []corev1.PodCondition{{
			Type:   corev1.PodReady,
			Status: corev1.ConditionTrue,
		}}
	}
	return pod
}

// newTestWatcher builds a PodWatcher directly (no kubernetes client needed)
// for unit-testing sync/markGone in isolation.
func newTestWatcher(port int, initial map[string]string, cb EngineCallbacks) *PodWatcher {
	if initial == nil {
		initial = make(map[string]string)
	}
	return &PodWatcher{
		port:       port,
		callbacks:  cb,
		registered: initial,
	}
}

// TestPodWatcherSync covers the idempotency and state-transition logic of sync().
func TestPodWatcherSync(t *testing.T) {
	const port = 8000

	tests := []struct {
		name              string
		initialRegistered map[string]string // pre-state of the registered map
		pod               *corev1.Pod
		wantOnReady       int
		wantOnGone        int
		wantReadyAddr     string // non-empty → verify the address passed to OnReady
	}{
		{
			name:          "new ready pod triggers OnReady",
			pod:           makePod("uid-1", "1.2.3.4", true),
			wantOnReady:   1,
			wantReadyAddr: "http://1.2.3.4:8000",
		},
		{
			// Same pod, same IP re-synced: no duplicate callback.
			name:              "ready pod idempotent: no duplicate OnReady",
			initialRegistered: map[string]string{"uid-1": "http://1.2.3.4:8000"},
			pod:               makePod("uid-1", "1.2.3.4", true),
			wantOnReady:       0,
		},
		{
			// Pod was ready, now transitions to not-ready.
			name:              "ready → not ready triggers OnGone",
			initialRegistered: map[string]string{"uid-1": "http://1.2.3.4:8000"},
			pod:               makePod("uid-1", "", false),
			wantOnGone:        1,
		},
		{
			// Pod restarts and gets a new IP: OnGone for the old address then OnReady for the new.
			name:              "IP change: OnGone then OnReady",
			initialRegistered: map[string]string{"uid-1": "http://1.2.3.1:8000"},
			pod:               makePod("uid-1", "1.2.3.4", true),
			wantOnReady:       1,
			wantOnGone:        1,
			wantReadyAddr:     "http://1.2.3.4:8000",
		},
		{
			// Pod never became ready: no callbacks expected.
			name:        "not ready and not registered: no callbacks",
			pod:         makePod("uid-1", "", false),
			wantOnReady: 0,
			wantOnGone:  0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var onReadyCalls, onGoneCalls int
			var lastReadyAddr string

			pw := newTestWatcher(port, tc.initialRegistered, EngineCallbacks{
				OnReady: func(_, addr string) {
					onReadyCalls++
					lastReadyAddr = addr
				},
				OnGone: func(_ string) { onGoneCalls++ },
			})

			pw.sync(tc.pod)

			if onReadyCalls != tc.wantOnReady {
				t.Errorf("OnReady calls = %d, want %d", onReadyCalls, tc.wantOnReady)
			}
			if onGoneCalls != tc.wantOnGone {
				t.Errorf("OnGone calls = %d, want %d", onGoneCalls, tc.wantOnGone)
			}
			if tc.wantReadyAddr != "" && lastReadyAddr != tc.wantReadyAddr {
				t.Errorf("OnReady addr = %q, want %q", lastReadyAddr, tc.wantReadyAddr)
			}
		})
	}
}

// TestMarkGone verifies that markGone fires OnGone only for registered pods.
func TestMarkGone(t *testing.T) {
	tests := []struct {
		name              string
		initialRegistered map[string]string
		id                string
		wantOnGone        int
	}{
		{
			name:              "registered pod: OnGone called and entry removed",
			initialRegistered: map[string]string{"uid-1": "http://1.2.3.4:8000"},
			id:                "uid-1",
			wantOnGone:        1,
		},
		{
			name:       "unknown pod: no callback",
			id:         "uid-unknown",
			wantOnGone: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var onGoneCalls int

			pw := newTestWatcher(8000, tc.initialRegistered, EngineCallbacks{
				OnReady: func(_, _ string) {},
				OnGone:  func(_ string) { onGoneCalls++ },
			})

			pw.markGone(tc.id)

			if onGoneCalls != tc.wantOnGone {
				t.Errorf("OnGone calls = %d, want %d", onGoneCalls, tc.wantOnGone)
			}
			if tc.wantOnGone > 0 {
				if _, stillPresent := pw.registered[tc.id]; stillPresent {
					t.Errorf("engine %q still in registered map after markGone", tc.id)
				}
			}
		})
	}
}

// TestIsPodReady covers the ready-condition lookup.
func TestIsPodReady(t *testing.T) {
	tests := []struct {
		name       string
		conditions []corev1.PodCondition
		want       bool
	}{
		{
			name: "no conditions",
			want: false,
		},
		{
			name: "Ready=True",
			conditions: []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: corev1.ConditionTrue,
			}},
			want: true,
		},
		{
			name: "Ready=False",
			conditions: []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: corev1.ConditionFalse,
			}},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod := &corev1.Pod{}
			pod.Status.Conditions = tc.conditions
			if got := isPodReady(pod); got != tc.want {
				t.Errorf("isPodReady = %v, want %v", got, tc.want)
			}
		})
	}
}
