package history

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func newStore(t *testing.T) (*JSONLStore, string) {
	t.Helper()
	dir := t.TempDir()
	return NewJSONLStore(dir), dir
}

func TestAppendAndRead(t *testing.T) {
	s, _ := newStore(t)
	ts := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)

	if err := s.Append("sess1", "user", "hello", ts); err != nil {
		t.Fatalf("Append user: %v", err)
	}
	if err := s.Append("sess1", "assistant", "hi", ts.Add(time.Second)); err != nil {
		t.Fatalf("Append assistant: %v", err)
	}

	turns, err := s.Read("sess1")
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(turns))
	}
	if turns[0].Role != "user" || turns[0].Content != "hello" {
		t.Errorf("turn[0] unexpected: %+v", turns[0])
	}
	if turns[1].Role != "assistant" || turns[1].Content != "hi" {
		t.Errorf("turn[1] unexpected: %+v", turns[1])
	}
}

func TestReadMissingSession(t *testing.T) {
	s, _ := newStore(t)
	turns, err := s.Read("nonexistent")
	if err != nil {
		t.Fatalf("expected no error for missing session, got %v", err)
	}
	if len(turns) != 0 {
		t.Errorf("expected 0 turns, got %d", len(turns))
	}
}

func TestInvalidSessionID(t *testing.T) {
	s, _ := newStore(t)
	cases := []string{"../evil", "foo/bar", "a b", "a\x00b", ""}
	for _, id := range cases {
		if err := s.Append(id, "user", "x", time.Now()); !errors.Is(err, ErrInvalidSessionID) {
			t.Errorf("Append(%q): expected ErrInvalidSessionID, got %v", id, err)
		}
		if _, err := s.Read(id); !errors.Is(err, ErrInvalidSessionID) {
			t.Errorf("Read(%q): expected ErrInvalidSessionID, got %v", id, err)
		}
	}
}

func TestEmptyContentNotWritten(t *testing.T) {
	s, dir := newStore(t)
	if err := s.Append("sess2", "assistant", "", time.Now()); err != nil {
		t.Fatalf("Append empty: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sess2.jsonl")); !errors.Is(err, os.ErrNotExist) {
		t.Error("expected no file for empty content turn")
	}
}

func TestJSONLFileFormat(t *testing.T) {
	s, dir := newStore(t)
	ts := time.Date(2024, 6, 1, 10, 0, 0, 0, time.UTC)
	_ = s.Append("sess3", "user", "msg1", ts)
	_ = s.Append("sess3", "assistant", "msg2", ts.Add(time.Minute))

	f, err := os.Open(filepath.Join(dir, "sess3.jsonl"))
	if err != nil {
		t.Fatalf("open file: %v", err)
	}
	defer func() { _ = f.Close() }()

	scanner := bufio.NewScanner(f)
	var lines []map[string]string
	for scanner.Scan() {
		var m map[string]string
		if err := json.Unmarshal(scanner.Bytes(), &m); err != nil {
			t.Fatalf("line not valid JSON: %v", err)
		}
		for _, field := range []string{"role", "content", "timestamp"} {
			if _, ok := m[field]; !ok {
				t.Errorf("line missing field %q: %v", field, m)
			}
		}
		lines = append(lines, m)
	}
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
	}
}

func TestConcurrentAppend(t *testing.T) {
	s, _ := newStore(t)
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_ = s.Append("concurrent", "user", "msg", time.Now())
		}()
	}
	wg.Wait()

	turns, err := s.Read("concurrent")
	if err != nil {
		t.Fatalf("Read after concurrent appends: %v", err)
	}
	if len(turns) != n {
		t.Errorf("expected %d turns, got %d", n, len(turns))
	}
}

func TestValidSessionIDs(t *testing.T) {
	s, _ := newStore(t)
	valid := []string{
		"abc123",
		"ABC-123_def",
		"550e8400-e29b-41d4-a716-446655440000", // UUID v4
	}
	for _, id := range valid {
		if err := s.Append(id, "user", "hello", time.Now()); err != nil {
			t.Errorf("Append(%q): unexpected error %v", id, err)
		}
		if _, err := s.Read(id); err != nil {
			t.Errorf("Read(%q): unexpected error %v", id, err)
		}
	}
}
