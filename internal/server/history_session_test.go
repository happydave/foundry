package server

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/happydave/foundry/internal/history"
	"github.com/happydave/foundry/internal/processmanager"
	"github.com/happydave/foundry/internal/registry"
)

// readTurns polls the store until at least n turns are present or the deadline passes.
// The client receives the HTTP response before the server finishes recordTurns (since
// data is flushed to TCP before the handler returns), so a brief retry is necessary.
func readTurns(t *testing.T, store *history.JSONLStore, sessionID string, wantAtLeast int) []history.Turn {
	t.Helper()
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		turns, err := store.Read(sessionID)
		if err != nil {
			t.Fatalf("store.Read(%q): %v", sessionID, err)
		}
		if len(turns) >= wantAtLeast || time.Now().After(deadline) {
			return turns
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// --- unit tests for content extraction helpers ---

func TestSSEAssistantContent_AssemblesChunks(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"data: [DONE]\n\n"
	got := sseAssistantContent([]byte(sse))
	if got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

func TestSSEAssistantContent_PartialStream_ReturnsEmpty(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n"
	// No [DONE] — stream cut off
	got := sseAssistantContent([]byte(sse))
	if got != "" {
		t.Errorf("expected empty for partial stream, got %q", got)
	}
}

func TestSSEAssistantContent_EmptyDeltaSkipped(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	got := sseAssistantContent([]byte(sse))
	if got != "hi" {
		t.Errorf("got %q, want %q", got, "hi")
	}
}

func TestJSONAssistantContent_ExtractsContent(t *testing.T) {
	body := `{"choices":[{"message":{"role":"assistant","content":"hello world"}}]}`
	got := jsonAssistantContent([]byte(body))
	if got != "hello world" {
		t.Errorf("got %q, want %q", got, "hello world")
	}
}

func TestJSONAssistantContent_EmptyChoices_ReturnsEmpty(t *testing.T) {
	body := `{"choices":[]}`
	got := jsonAssistantContent([]byte(body))
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

// --- integration tests via HTTP ---

// historyFixture extends serverFixture with a backend server and loaded model.
type historyFixture struct {
	*serverFixture
	store   *history.JSONLStore
	backend *httptest.Server
}

func newHistoryFixture(t *testing.T, backendHandler http.HandlerFunc) *historyFixture {
	t.Helper()

	// Register store dir FIRST so its t.TempDir cleanup runs LAST (LIFO).
	// This ensures the HTTP server (registered later via newFixture) shuts down
	// and drains in-flight requests before the store directory is removed.
	store := history.NewJSONLStore(t.TempDir())

	backend := httptest.NewServer(backendHandler)
	t.Cleanup(backend.Close)

	backendPort := backend.Listener.Addr().(*net.TCPAddr).Port

	// newFixture registers httpSrv.Shutdown cleanup — runs before store cleanup (LIFO).
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID: 1,
		Port:    backendPort,
	}

	f.srv.SetHistoryStore(store)

	return &historyFixture{
		serverFixture: f,
		store:         store,
		backend:       backend,
	}
}

func jsonBackend(responseJSON string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(responseJSON))
	}
}

func chatBody(userMsg string) string {
	return `{"model":"test-model","messages":[{"role":"user","content":"` + userMsg + `"}]}`
}

func postChat(t *testing.T, f *historyFixture, body string, headers map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.ContentLength = int64(len(body))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// TestHistory_NonPersistent_NoFiles verifies non-persistent requests leave no files.
func TestHistory_NonPersistent_NoFiles(t *testing.T) {
	f := newHistoryFixture(t, jsonBackend(`{"choices":[{"message":{"content":"hi"}}]}`))

	body := chatBody("hello")
	resp := postChat(t, f, body, nil) // no X-Foundry-Persist header
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	// Give the server a moment to process then confirm no turns.
	time.Sleep(20 * time.Millisecond)
	turns, err := f.store.Read("anysession")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(turns) != 0 {
		t.Errorf("expected no turns for non-persistent request, got %d", len(turns))
	}
}

// TestHistory_GeneratesSessionID verifies a session ID is generated when none provided.
func TestHistory_GeneratesSessionID(t *testing.T) {
	f := newHistoryFixture(t, jsonBackend(`{"choices":[{"message":{"content":"hi"}}]}`))

	body := chatBody("hello")
	resp := postChat(t, f, body, map[string]string{"X-Foundry-Persist": "true"})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	sid := resp.Header.Get("X-Foundry-Session-Id")
	if sid == "" {
		t.Fatal("expected X-Foundry-Session-Id in response header")
	}
	// Generated ID must pass the store's allowlist (hex chars only); wait for turns to settle.
	readTurns(t, f.store, sid, 1)
	if _, err := f.store.Read(sid); err != nil {
		t.Errorf("generated session ID %q is not valid: %v", sid, err)
	}
}

// TestHistory_InvalidSessionID_Returns400 checks the allowlist enforcement.
func TestHistory_InvalidSessionID_Returns400(t *testing.T) {
	f := newHistoryFixture(t, jsonBackend(`{"choices":[]}`))

	body := chatBody("hello")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "../evil",
	})
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

// TestHistory_PrependsPriorTurns verifies prior turns are prepended to the request.
func TestHistory_PrependsPriorTurns(t *testing.T) {
	var receivedBody []byte
	f := newHistoryFixture(t, func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"sure"}}]}`))
	})

	ts := time.Now()
	_ = f.store.Append("sess1", "user", "hi", ts)
	_ = f.store.Append("sess1", "assistant", "hello!", ts.Add(time.Second))

	body := chatBody("what's up?")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "sess1",
	})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	var sentBody map[string]json.RawMessage
	if err := json.Unmarshal(receivedBody, &sentBody); err != nil {
		t.Fatalf("backend received invalid JSON: %v", err)
	}
	var msgs []chatMessage
	if err := json.Unmarshal(sentBody["messages"], &msgs); err != nil {
		t.Fatalf("messages field invalid: %v", err)
	}

	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (2 prior + 1 new), got %d", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[0].Content != "hi" {
		t.Errorf("prior turn[0] = %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "hello!" {
		t.Errorf("prior turn[1] = %+v", msgs[1])
	}
	if msgs[2].Role != "user" || msgs[2].Content != "what's up?" {
		t.Errorf("new user message = %+v", msgs[2])
	}
}

// TestHistory_RecordsTurns verifies turns are appended after a successful response.
func TestHistory_RecordsTurns(t *testing.T) {
	f := newHistoryFixture(t, jsonBackend(`{"choices":[{"message":{"content":"I'm fine"}}]}`))

	body := chatBody("how are you?")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "record1",
	})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	turns := readTurns(t, f.store, "record1", 2)
	if len(turns) != 2 {
		t.Fatalf("expected 2 turns recorded, got %d", len(turns))
	}
	if turns[0].Role != "user" || turns[0].Content != "how are you?" {
		t.Errorf("user turn = %+v", turns[0])
	}
	if turns[1].Role != "assistant" || turns[1].Content != "I'm fine" {
		t.Errorf("assistant turn = %+v", turns[1])
	}
}

// TestHistory_NoTurnsOnUpstreamError verifies no turns are written when upstream returns 500.
func TestHistory_NoTurnsOnUpstreamError(t *testing.T) {
	f := newHistoryFixture(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	})

	body := chatBody("hello?")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "errortest",
	})
	// Proxy returns upstream status.
	_ = resp.Body.Close()

	time.Sleep(20 * time.Millisecond)
	turns, err := f.store.Read("errortest")
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(turns) != 0 {
		t.Errorf("expected no turns after upstream error, got %d", len(turns))
	}
}

// TestHistory_SSEResponse verifies streaming assistant content is assembled and recorded.
func TestHistory_SSEResponse(t *testing.T) {
	sseBody := "data: {\"choices\":[{\"delta\":{\"content\":\"Nice\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" day!\"}}]}\n\n" +
		"data: [DONE]\n\n"

	f := newHistoryFixture(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseBody))
	})

	body := chatBody("how's the weather?")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "ssetest",
	})
	assertStatus(t, resp, http.StatusOK)
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	turns := readTurns(t, f.store, "ssetest", 2)
	if len(turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(turns))
	}
	if turns[1].Content != "Nice day!" {
		t.Errorf("assembled SSE content = %q, want %q", turns[1].Content, "Nice day!")
	}
}

// TestHistory_SSEPartialStream_NoTurnRecorded verifies a cut-off stream produces no turn.
func TestHistory_SSEPartialStream_NoTurnRecorded(t *testing.T) {
	// Stream without [DONE].
	sseBody := "data: {\"choices\":[{\"delta\":{\"content\":\"incomplete\"}}]}\n\n"

	f := newHistoryFixture(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseBody))
	})

	body := chatBody("hello")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "partialsse",
	})
	assertStatus(t, resp, http.StatusOK)
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	time.Sleep(20 * time.Millisecond)
	turns, err := f.store.Read("partialsse")
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(turns) != 0 {
		t.Errorf("expected no turns for partial SSE stream, got %d", len(turns))
	}
}

// TestHistory_NewSession_NoPriorTurns verifies a new persistent session (no prior
// history) forwards the original request body unchanged.
func TestHistory_NewSession_NoPriorTurns(t *testing.T) {
	var receivedBody []byte
	f := newHistoryFixture(t, func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"}}]}`))
	})

	body := chatBody("hello")
	resp := postChat(t, f, body, map[string]string{
		"X-Foundry-Persist":    "true",
		"X-Foundry-Session-Id": "newbrand",
	})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	var sentBody map[string]json.RawMessage
	if err := json.Unmarshal(receivedBody, &sentBody); err != nil {
		t.Fatalf("backend received invalid JSON: %v", err)
	}
	var msgs []chatMessage
	if err := json.Unmarshal(sentBody["messages"], &msgs); err != nil {
		t.Fatalf("messages field invalid: %v", err)
	}

	if len(msgs) != 1 {
		t.Fatalf("expected 1 message (no prior turns prepended), got %d", len(msgs))
	}
}

// TestHistory_PersistFalseHeaderIgnored verifies non-"true" persist values are ignored.
func TestHistory_PersistFalseHeaderIgnored(t *testing.T) {
	f := newHistoryFixture(t, jsonBackend(`{"choices":[{"message":{"content":"hi"}}]}`))

	body := chatBody("hello")
	for _, val := range []string{"false", "1", "yes", "True"} {
		resp := postChat(t, f, body, map[string]string{
			"X-Foundry-Persist":    val,
			"X-Foundry-Session-Id": "shouldnotexist",
		})
		assertStatus(t, resp, http.StatusOK)
		_ = resp.Body.Close()

		if sid := resp.Header.Get("X-Foundry-Session-Id"); sid != "" {
			t.Errorf("persist=%q: expected no session ID in response, got %q", val, sid)
		}
	}

	// No turns should have been written for any of those requests.
	time.Sleep(20 * time.Millisecond)
	turns, _ := f.store.Read("shouldnotexist")
	if len(turns) != 0 {
		t.Errorf("expected no turns for non-persistent requests, got %d", len(turns))
	}
}

// TestHistory_NoStoreSet_RequestPassesThrough verifies no-op when no store is configured.
func TestHistory_NoStoreSet_RequestPassesThrough(t *testing.T) {
	backend := jsonBackend(`{"choices":[{"message":{"content":"hi"}}]}`)
	backendServer := httptest.NewServer(backend)
	t.Cleanup(backendServer.Close)
	backendPort := backendServer.Listener.Addr().(*net.TCPAddr).Port

	f := newFixture(t) // no SetHistoryStore
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{ModelID: 1, Port: backendPort}

	body := chatBody("hello")
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", strings.NewReader(body))
	req.ContentLength = int64(len(body))
	req.Header.Set("X-Foundry-Persist", "true")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	// Should succeed without a history store set.
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
}

// TestHistory_Accumulates_AcrossRequests verifies the full multi-turn scenario.
func TestHistory_Accumulates_AcrossRequests(t *testing.T) {
	var receivedBodies [][]byte
	f := newHistoryFixture(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		receivedBodies = append(receivedBodies, b)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"reply"}}]}`))
	})

	const sid = "multi1"
	headers := map[string]string{"X-Foundry-Persist": "true", "X-Foundry-Session-Id": sid}

	// First request: no prior history.
	resp1 := postChat(t, f, chatBody("turn1"), headers)
	assertStatus(t, resp1, http.StatusOK)
	_ = resp1.Body.Close()

	// Wait for the first request's turns to be recorded before sending the second.
	readTurns(t, f.store, sid, 2)

	// Second request: should see turn1+reply prepended.
	resp2 := postChat(t, f, chatBody("turn2"), headers)
	assertStatus(t, resp2, http.StatusOK)
	_ = resp2.Body.Close()

	// The second backend request should have 3 messages: turn1, reply, turn2.
	if len(receivedBodies) < 2 {
		t.Fatalf("expected 2 backend calls, got %d", len(receivedBodies))
	}
	var sentBody map[string]json.RawMessage
	if err := json.Unmarshal(receivedBodies[1], &sentBody); err != nil {
		t.Fatalf("backend received invalid JSON: %v", err)
	}
	var msgs []chatMessage
	if err := json.Unmarshal(sentBody["messages"], &msgs); err != nil {
		t.Fatalf("messages field invalid: %v", err)
	}

	if len(msgs) != 3 {
		t.Fatalf("2nd request: expected 3 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Content != "turn1" || msgs[1].Content != "reply" || msgs[2].Content != "turn2" {
		t.Errorf("message sequence wrong: %+v", msgs)
	}
}
