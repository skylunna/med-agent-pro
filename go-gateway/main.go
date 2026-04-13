package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// 基于 session_id 的独立限流器 (生产环境可换 Redis)
var limiters sync.Map

// 限流
func getLimiter(sessionID string) *rate.Limiter {
	//给我拿到这个用户的专属限流器。是第一次来，就给他创建一个 “每秒 2 个请求” 的限流器，以后再来，都用同一个限流器。
	limiter, _ := limiters.LoadOrStore(sessionID, rate.NewLimiter(2, 5)) // 2 req/s peek = 5
	return limiter.(*rate.Limiter)
}

// 计时 + 打印日志
func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next(w, r)
		log.Printf("[%s] %s %s %v", r.Method, r.URL.Path, r.RemoteAddr, time.Since(start))
	}
}

// 限流中间件
func rateLimitMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sessionID := r.Header.Get("X-Session-ID")
		if sessionID == "" {
			http.Error(w, `{"error": "missing session_id"}`, http.StatusBadRequest)
			return
		}

		if !getLimiter(sessionID).Allow() {
			http.Error(w, `{"error": "rate limit exceeded"}`, http.StatusTooManyRequests)
			return
		}
		next(w, r)
	}
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	// w 用来给前端返回数据
	// r 是前端发来的请求
	// 只允许 post 请求
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// 接收前端传的 json
	var req struct {
		Question  string `json:"question"`
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}

	stream := r.URL.Query().Get("stream") == "true"

	// 拿到 Python AI 服务的地址
	pythonURL := os.Getenv("PYTHON_SERVICE_URL")
	if pythonURL == "" {
		pythonURL = "http://localhost:8001"
	}

	payload := fmt.Sprintf(`{"question": "%s", "session_id": "%s", "stream": %t}`, req.Question, req.SessionID, stream)
	pythonReq, _ := http.NewRequest("POST", pythonURL+"/agent/rag_query", strings.NewReader(payload))
	pythonReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(pythonReq)

	if err != nil {
		http.Error(w, `{"error":"ai service unavailable"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, `{"error":"streaming not supported"}`, http.StatusInternalServerError)
			return
		}

		buf := make([]byte, 4096)
		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				w.Write(buf[:n])
				flusher.Flush()
			}
			if readErr != nil {
				if readErr == io.EOF {
					break
				}
				log.Printf("⚠️ Stream read error: %v", readErr)
				break
			}
		}
	} else {
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	}

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/chat", loggingMiddleware(rateLimitMiddleware(chatHandler)))

	log.Println("🚀 Go Gateway running on : 8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
