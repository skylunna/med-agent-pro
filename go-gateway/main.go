package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type ChatRequest struct {
	Question  string `json:"question"`
	SessionID string `json:"session_id"`
}

func main() {
	http.HandleFunc("/api/v1/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method Not Allowed", 405)
			return
		}

		var req ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		r.Body.Close()

		// TODO: 后续加 JWT / 限流 / 日志中间件

		// 转发给 Python 服务
		pythonURL := "http://localhost:8001/agent/query"
		resp, err := http.Post(pythonURL, "application/json", strings.NewReader(fmt.Sprintf(`{"question":"%s","session_id":"%s"}`, req.Question, req.SessionID)))
		if err != nil {
			http.Error(w, "AI Service Unavailable", 502)
			return
		}
		defer resp.Body.Close()

		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	})

	fmt.Println("Go Gateway running on :8080")
	http.ListenAndServe(":8080", nil)
}
