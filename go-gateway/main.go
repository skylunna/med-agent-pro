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
// 防止一个人刷爆请求
var limiters sync.Map

/*
每个用户一个限流器
*/
func getLimiter(sessionID string) *rate.Limiter {
	//给我拿到这个用户的专属限流器。是第一次来，就给他创建一个 “每秒 2 个请求” 的限流器，以后再来，都用同一个限流器。
	// sync.map 自带的方法 LoadOrStore, 有就取出来，没有就存一个
	limiter, _ := limiters.LoadOrStore(sessionID, rate.NewLimiter(2, 5)) // 2 req/s peek = 5
	return limiter.(*rate.Limiter)
}

func maskSession(id string) string {
	if len(id) < 6 {
		return id
	}
	return id[:3] + "***" + id[len(id)-3:]
}

// 计时 + 打印日志
func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		session := r.Header.Get("X-Session-ID")
		next(w, r)
		log.Printf("[%s] %s %s session=%s latency=%v",
			r.Method, r.URL.Path, r.RemoteAddr, maskSession(session), time.Since(start))
	}
}

// 限流中间件
// 必须传 X-Session-ID
// 超过限制直接返回429限流
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
		Stream    bool   `json:"stream"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}

	// 判断是否流式输出
	// stream := r.URL.Query().Get("stream") == "true"

	// 拿到 Python AI 服务的地址
	pythonURL := os.Getenv("PYTHON_SERVICE_URL")
	if pythonURL == "" {
		pythonURL = "http://localhost:8001"
	}

	targetURL := fmt.Sprintf("%s/agent/rag_query", pythonURL)
	payload := fmt.Sprintf(`{"question":"%s","session_id":"%s","stream":%t}`,
		req.Question, req.SessionID, req.Stream)

	httpReq, err := http.NewRequest("POST", targetURL, strings.NewReader(payload))
	if err != nil {
		http.Error(w, `{"error":"ai_service_unavailable"}`, http.StatusBadGateway)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(httpReq)

	if err != nil {
		http.Error(w, `{"error":"ai service unavailable"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if req.Stream {
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

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok","service":"go-gateway","mode":"cloud-first"}`))
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/chat", loggingMiddleware(rateLimitMiddleware(chatHandler)))
	mux.HandleFunc("/health", healthHandler)

	log.Println("🚀 Go Gateway running on :8080 (Cloud-First Mode)")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
