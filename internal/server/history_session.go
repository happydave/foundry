package server

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/happydave/foundry/internal/history"
)

// historySession holds per-request state for a persistent chat session.
type historySession struct {
	sessionID   string
	userContent string
}

// chatMessage is the OpenAI-compatible message object used when reading/rewriting
// the messages array for history injection.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// prepareSession reads the session headers from r, validates or generates a session ID,
// loads prior turns, and rewrites buf with prior turns prepended to the messages array.
//
// Returns (ses, newBuf, abort):
//   - ses nil, abort false: non-persistent request; newBuf == buf unchanged.
//   - ses non-nil, abort false: persistent session; newBuf is the rewritten body.
//   - abort true: an error response has already been written to w; caller should return.
func (s *Server) prepareSession(w http.ResponseWriter, r *http.Request, buf []byte) (*historySession, []byte, bool) {
	if r.Header.Get("X-Foundry-Persist") != "true" {
		return nil, buf, false
	}

	sessionID := r.Header.Get("X-Foundry-Session-Id")
	if sessionID == "" {
		id, err := generateSessionID()
		if err != nil {
			writeOAIError(w, http.StatusInternalServerError, "server_error", "session_error",
				"failed to generate session ID")
			return nil, nil, true
		}
		sessionID = id
		// Emit the generated ID so the client can reuse it for subsequent requests.
		w.Header().Set("X-Foundry-Session-Id", sessionID)
	}

	priorTurns, err := s.historyStore.Read(sessionID)
	if err != nil {
		if errors.Is(err, history.ErrInvalidSessionID) {
			writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_session_id",
				fmt.Sprintf("invalid session ID: %v", err))
			return nil, nil, true
		}
		writeOAIError(w, http.StatusInternalServerError, "server_error", "history_error",
			fmt.Sprintf("failed to read session history: %v", err))
		return nil, nil, true
	}

	userContent, newBuf, ok := prependHistory(w, buf, priorTurns)
	if !ok {
		return nil, nil, true
	}

	return &historySession{sessionID: sessionID, userContent: userContent}, newBuf, false
}

// prependHistory prepends priorTurns to the messages array inside buf (a JSON chat
// completions body). Returns the user content of the last user message in the original
// messages array, the rewritten body, and true on success.
func prependHistory(w http.ResponseWriter, buf []byte, priorTurns []history.Turn) (string, []byte, bool) {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(buf, &body); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_body",
			fmt.Sprintf("cannot parse request body: %v", err))
		return "", nil, false
	}

	rawMsgs, hasMsgs := body["messages"]
	if !hasMsgs || string(rawMsgs) == "null" {
		// Not a chat completions request or no messages field; pass through unchanged.
		return "", buf, true
	}

	var msgs []chatMessage
	if err := json.Unmarshal(rawMsgs, &msgs); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_body",
			fmt.Sprintf("cannot parse messages: %v", err))
		return "", nil, false
	}

	// Extract the content of the last user message to record after inference.
	userContent := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			userContent = msgs[i].Content
			break
		}
	}

	if len(priorTurns) > 0 {
		prepended := make([]chatMessage, 0, len(priorTurns)+len(msgs))
		for _, t := range priorTurns {
			prepended = append(prepended, chatMessage{Role: t.Role, Content: t.Content})
		}
		prepended = append(prepended, msgs...)

		enc, err := json.Marshal(prepended)
		if err != nil {
			writeOAIError(w, http.StatusInternalServerError, "server_error", "history_error",
				"failed to encode history")
			return "", nil, false
		}
		body["messages"] = enc
	}

	newBuf, err := json.Marshal(body)
	if err != nil {
		writeOAIError(w, http.StatusInternalServerError, "server_error", "history_error",
			"failed to re-encode request body")
		return "", nil, false
	}

	return userContent, newBuf, true
}

// recordTurns writes the user and assistant turns to the history store after a
// successful inference response. Disk errors are logged but not surfaced to the client
// (inference has already succeeded).
func (s *Server) recordTurns(ses *historySession, crw *capturingResponseWriter) {
	if crw.status() < 200 || crw.status() >= 300 {
		return
	}

	assistantContent := extractAssistantContent(crw)
	if assistantContent == "" {
		return
	}

	now := time.Now().UTC()

	if ses.userContent != "" {
		if err := s.historyStore.Append(ses.sessionID, "user", ses.userContent, now); err != nil {
			if s.logger != nil {
				s.logger.Warn("history: failed to record user turn",
					"session", ses.sessionID, "error", err)
			}
			return
		}
	}

	if err := s.historyStore.Append(ses.sessionID, "assistant", assistantContent, now.Add(time.Millisecond)); err != nil {
		if s.logger != nil {
			s.logger.Warn("history: failed to record assistant turn",
				"session", ses.sessionID, "error", err)
		}
	}
}

// extractAssistantContent reads the captured response and returns the assistant
// message content, handling both SSE streaming and plain JSON responses.
func extractAssistantContent(crw *capturingResponseWriter) string {
	if strings.Contains(crw.Header().Get("Content-Type"), "text/event-stream") {
		return sseAssistantContent(crw.buf.Bytes())
	}
	return jsonAssistantContent(crw.buf.Bytes())
}

// sseAssistantContent assembles assistant content from SSE delta chunks.
// Returns an empty string if [DONE] was never received (partial/aborted stream).
func sseAssistantContent(data []byte) string {
	type sseChunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	var sb strings.Builder
	done := false
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			done = true
			break
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			sb.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if !done {
		return "" // partial stream — discard, do not write a turn
	}
	return sb.String()
}

// jsonAssistantContent extracts assistant content from a non-streaming JSON response.
func jsonAssistantContent(data []byte) string {
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &resp); err != nil || len(resp.Choices) == 0 {
		return ""
	}
	return resp.Choices[0].Message.Content
}

// capturingResponseWriter wraps an http.ResponseWriter, teeing all writes to an
// internal buffer while passing them through to the client. Implements http.Flusher
// so SSE streaming is not blocked.
type capturingResponseWriter struct {
	inner      http.ResponseWriter
	statusCode int
	buf        bytes.Buffer
	headerSent bool
}

func (c *capturingResponseWriter) Header() http.Header { return c.inner.Header() }

func (c *capturingResponseWriter) WriteHeader(code int) {
	c.statusCode = code
	c.headerSent = true
	c.inner.WriteHeader(code)
}

func (c *capturingResponseWriter) Write(b []byte) (int, error) {
	c.buf.Write(b)
	return c.inner.Write(b)
}

func (c *capturingResponseWriter) Flush() {
	if f, ok := c.inner.(http.Flusher); ok {
		f.Flush()
	}
}

func (c *capturingResponseWriter) status() int {
	if !c.headerSent {
		return http.StatusOK
	}
	return c.statusCode
}

// generateSessionID returns a 32-character lowercase hex string backed by 16 crypto-random bytes.
func generateSessionID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
