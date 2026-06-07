package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/happydave/foundry/internal/processmanager"
)

// InferenceHook is called before each inference request is proxied, after model
// validation and body parsing. It may modify r (e.g., replace r.Body with
// rewritten content). Return false to abort proxying; the hook is responsible
// for writing the error response in that case.
type InferenceHook func(w http.ResponseWriter, r *http.Request) bool

// oaiErrorBody is the OpenAI-compatible JSON error response envelope.
type oaiErrorBody struct {
	Error oaiErrorDetail `json:"error"`
}

type oaiErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// anthropicErrorBody is the Anthropic-compatible JSON error response envelope.
type anthropicErrorBody struct {
	Type  string               `json:"type"`
	Error anthropicErrorDetail `json:"error"`
}

type anthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// oaiModelsResponse is the OpenAI-format response for GET /v1/models.
type oaiModelsResponse struct {
	Object string     `json:"object"`
	Data   []oaiModel `json:"data"`
}

type oaiModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// inferenceModelField is the minimal struct used to extract the model field from
// an OpenAI-compatible request body without unmarshalling the full payload.
type inferenceModelField struct {
	Model string `json:"model"`
}

func writeOAIError(w http.ResponseWriter, status int, errType, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(oaiErrorBody{
		Error: oaiErrorDetail{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(anthropicErrorBody{
		Type: "error",
		Error: anthropicErrorDetail{
			Type:    errType,
			Message: message,
		},
	})
}

// handleOAIModels handles GET /v1/models. Returns only loaded, healthy models in
// OpenAI list format. Models whose subprocesses have crashed are excluded.
func (s *Server) handleOAIModels(w http.ResponseWriter, _ *http.Request) {
	loaded := s.procMgr.List()
	data := make([]oaiModel, 0, len(loaded))
	for _, lm := range loaded {
		if lm.Health() != processmanager.HealthStatusHealthy {
			continue
		}
		m, found := s.registry.Get(lm.ModelID)
		if !found {
			continue
		}
		data = append(data, oaiModel{
			ID:      m.DisplayName,
			Object:  "model",
			Created: lm.LoadTime.Unix(),
			OwnedBy: "foundry",
		})
	}
	writeJSON(w, http.StatusOK, oaiModelsResponse{
		Object: "list",
		Data:   data,
	})
}

// handleInferenceProxy handles POST /v1/chat/completions and POST /v1/completions.
// It reads the request body to extract the model field, validates the model is loaded
// and healthy, then reverse-proxies the full request to the target llama-server port.
// Streaming responses (SSE) are passed through without buffering.
func (s *Server) handleInferenceProxy(w http.ResponseWriter, r *http.Request) {
	// Buffer the body to extract the model field; reconstruct it for forwarding.
	// For a local automation tool, request bodies are small in practice. The
	// Remaining Unknowns in plan.md acknowledges this trade-off.
	buf, err := io.ReadAll(r.Body)
	if err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "request_error",
			fmt.Sprintf("failed to read request body: %v", err))
		return
	}

	if len(buf) == 0 {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request_error",
			"request body is required")
		return
	}

	var field inferenceModelField
	if err := json.Unmarshal(buf, &field); err != nil {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request_error",
			fmt.Sprintf("invalid JSON in request body: %v", err))
		return
	}
	if field.Model == "" {
		writeOAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request_error",
			"missing required field: model")
		return
	}

	m, found := s.registry.GetByName(field.Model)
	if !found {
		writeOAIError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model %q is not available", field.Model))
		return
	}

	lm, loaded := s.procMgr.Get(m.ID)
	if !loaded {
		writeOAIError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model %q is not loaded", field.Model))
		return
	}
	if lm.Health() != processmanager.HealthStatusHealthy {
		writeOAIError(w, http.StatusServiceUnavailable, "server_error", "model_unavailable",
			fmt.Sprintf("model %q is currently unavailable", field.Model))
		return
	}

	// Apply the inference hook if set.
	if s.inferenceHook != nil {
		r.Body = io.NopCloser(bytes.NewReader(buf))
		r.ContentLength = int64(len(buf))
		if !s.inferenceHook(w, r) {
			return
		}
		// Re-read buf in case the hook modified r.Body.
		var rerr error
		buf, rerr = io.ReadAll(r.Body)
		if rerr != nil {
			writeOAIError(w, http.StatusInternalServerError, "server_error", "request_error",
				fmt.Sprintf("failed to re-read request body after hook: %v", rerr))
			return
		}
	}

	// History session handling: validate/generate session ID, load prior turns,
	// and rewrite the request body with history prepended.
	var ses *historySession
	if s.historyStore != nil {
		var abort bool
		ses, buf, abort = s.prepareSession(w, r, buf)
		if abort {
			return
		}
	}

	// Reconstruct the body for the proxy (after any hook or history rewrite).
	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = int64(len(buf))

	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", lm.Port),
	}
	proxy := newReverseProxy(target, func(w http.ResponseWriter, req *http.Request, err error) {
		if s.logger != nil {
			s.logger.Warn("upstream proxy error",
				"path", req.URL.Path,
				"error", err,
			)
		}
		writeOAIError(w, http.StatusBadGateway, "server_error", "upstream_error",
			fmt.Sprintf("upstream error: %v", err))
	})

	// For persistent sessions, wrap w to capture the response for turn recording.
	proxyW := http.ResponseWriter(w)
	var crw *capturingResponseWriter
	if ses != nil {
		crw = &capturingResponseWriter{inner: w}
		proxyW = crw
	}
	proxy.ServeHTTP(proxyW, r)

	if ses != nil {
		s.recordTurns(ses, crw)
	}
}

// newReverseProxy creates a reverse proxy targeting the given URL with
// immediate flushing (required for SSE pass-through) and a custom error handler.
func newReverseProxy(target *url.URL, errHandler func(http.ResponseWriter, *http.Request, error)) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
		FlushInterval: -1,
		ErrorHandler:  errHandler,
	}
}

// handleMessagesProxy handles POST /v1/messages by proxying the request to the
// target llama-server's native /v1/messages endpoint. Errors are returned in
// the Anthropic error envelope format. The inference hook and history session
// machinery are intentionally not applied to this path.
func (s *Server) handleMessagesProxy(w http.ResponseWriter, r *http.Request) {
	buf, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("failed to read request body: %v", err))
		return
	}

	if len(buf) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			"request body is required")
		return
	}

	var field inferenceModelField
	if err := json.Unmarshal(buf, &field); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("invalid JSON in request body: %v", err))
		return
	}
	if field.Model == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error",
			"missing required field: model")
		return
	}

	m, found := s.registry.GetByName(field.Model)
	if !found {
		writeAnthropicError(w, http.StatusNotFound, "not_found_error",
			fmt.Sprintf("model %q is not available", field.Model))
		return
	}

	lm, loaded := s.procMgr.Get(m.ID)
	if !loaded {
		writeAnthropicError(w, http.StatusNotFound, "not_found_error",
			fmt.Sprintf("model %q is not loaded", field.Model))
		return
	}
	if lm.Health() != processmanager.HealthStatusHealthy {
		writeAnthropicError(w, http.StatusServiceUnavailable, "overloaded_error",
			fmt.Sprintf("model %q is currently unavailable", field.Model))
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = int64(len(buf))

	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", lm.Port),
	}
	proxy := newReverseProxy(target, func(w http.ResponseWriter, req *http.Request, err error) {
		if s.logger != nil {
			s.logger.Warn("upstream proxy error",
				"path", req.URL.Path,
				"error", err,
			)
		}
		writeAnthropicError(w, http.StatusBadGateway, "api_error",
			fmt.Sprintf("upstream error: %v", err))
	})
	proxy.ServeHTTP(w, r)
}
