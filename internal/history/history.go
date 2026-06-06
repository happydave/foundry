package history

import "time"

// Turn is a single message in a conversation session.
type Turn struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// Store is the interface for reading and writing session history.
// Implementations must be safe for concurrent use.
type Store interface {
	// Append adds a turn to the session identified by sessionID.
	Append(sessionID, role, content string, ts time.Time) error
	// Read returns all turns for sessionID in chronological order.
	// If no session exists, it returns an empty slice and no error.
	Read(sessionID string) ([]Turn, error)
}
