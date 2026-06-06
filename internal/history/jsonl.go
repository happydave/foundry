package history

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"time"
)

// sessionIDPattern is the allowlist for session IDs: URL-safe, filesystem-safe,
// compatible with UUID v4 and common ID formats. Enforced before any file operation
// to prevent path traversal.
var sessionIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// jsonlTurn is the on-disk representation of a turn (RFC 3339 timestamp).
type jsonlTurn struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
}

// JSONLStore is a history.Store backed by JSONL files, one per session.
// Concurrent writes to the same session are serialised via a per-session mutex.
type JSONLStore struct {
	dir string

	mu        sync.Mutex // guards sessionMu map
	sessionMu map[string]*sync.Mutex
}

// ErrInvalidSessionID is returned when a session ID contains disallowed characters.
var ErrInvalidSessionID = errors.New("session ID contains disallowed characters (allowed: [a-zA-Z0-9_-])")

// NewJSONLStore returns a JSONLStore rooted at dir. dir must already exist.
func NewJSONLStore(dir string) *JSONLStore {
	return &JSONLStore{
		dir:       dir,
		sessionMu: make(map[string]*sync.Mutex),
	}
}

func (s *JSONLStore) lockFor(sessionID string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.sessionMu[sessionID]
	if !ok {
		m = &sync.Mutex{}
		s.sessionMu[sessionID] = m
	}
	return m
}

// Append adds a turn to the session file, creating the file if necessary.
// Returns ErrInvalidSessionID if the session ID fails the character allowlist check.
func (s *JSONLStore) Append(sessionID, role, content string, ts time.Time) error {
	if !sessionIDPattern.MatchString(sessionID) {
		return ErrInvalidSessionID
	}
	if content == "" {
		// Guard: do not write empty content turns.
		return nil
	}

	rec := jsonlTurn{
		Role:      role,
		Content:   content,
		Timestamp: ts.UTC().Format(time.RFC3339),
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("history: marshal turn: %w", err)
	}

	mu := s.lockFor(sessionID)
	mu.Lock()
	defer mu.Unlock()

	path := filepath.Join(s.dir, sessionID+".jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("history: open session file: %w", err)
	}
	defer func() { _ = f.Close() }()

	line = append(line, '\n')
	if _, err := f.Write(line); err != nil {
		return fmt.Errorf("history: write turn: %w", err)
	}
	return nil
}

// Read returns all turns for sessionID in chronological order. Returns an empty
// slice (no error) if the session file does not exist.
// Returns ErrInvalidSessionID if the session ID fails the character allowlist check.
func (s *JSONLStore) Read(sessionID string) ([]Turn, error) {
	if !sessionIDPattern.MatchString(sessionID) {
		return nil, ErrInvalidSessionID
	}

	path := filepath.Join(s.dir, sessionID+".jsonl")
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return []Turn{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("history: open session file: %w", err)
	}
	defer func() { _ = f.Close() }()

	var turns []Turn
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec jsonlTurn
		if err := json.Unmarshal(line, &rec); err != nil {
			return nil, fmt.Errorf("history: parse session file: %w", err)
		}
		ts, err := time.Parse(time.RFC3339, rec.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("history: parse timestamp: %w", err)
		}
		turns = append(turns, Turn{
			Role:      rec.Role,
			Content:   rec.Content,
			Timestamp: ts,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("history: scan session file: %w", err)
	}
	if turns == nil {
		turns = []Turn{}
	}
	return turns, nil
}
