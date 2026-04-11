package main

import "testing"

func TestGetLimiter(t *testing.T) {
	lim := getLimiter("test-session")
	if lim == nil {
		t.Fatal("Limiter should not be nil")
	}
}
