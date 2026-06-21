package server

import (
	"embed"
	"io/fs"
	"net/http"
)

// uiAssets is the embedded operator console (vanilla HTML/CSS/JS). Embedding is
// compile-time and unconditional; serving is gated at request time by the
// uiEnabled flag so the asset tree is only reachable when the operator opts in.
//
//go:embed ui
var uiAssets embed.FS

// uiFileServer returns a file server rooted at the embedded ui directory.
func uiFileServer() http.Handler {
	sub, err := fs.Sub(uiAssets, "ui")
	if err != nil {
		// The ui directory is embedded at compile time; a failure here is a
		// build/packaging error, not a runtime condition.
		panic(err)
	}
	return http.FileServer(http.FS(sub))
}

// registerUI wires the operator console under /ui/. When the UI is disabled,
// every path under /ui/ (and /ui itself) returns 404 — the console is not served.
func (s *Server) registerUI(mux *http.ServeMux) {
	fileServer := http.StripPrefix("/ui", uiFileServer())

	mux.HandleFunc("GET /ui", func(w http.ResponseWriter, r *http.Request) {
		if !s.uiEnabled {
			http.NotFound(w, r)
			return
		}
		http.Redirect(w, r, "/ui/", http.StatusMovedPermanently)
	})

	mux.Handle("GET /ui/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.uiEnabled {
			http.NotFound(w, r)
			return
		}
		fileServer.ServeHTTP(w, r)
	}))
}

// SetUIEnabled toggles serving of the operator console. Must be called before
// ListenAndServe.
func (s *Server) SetUIEnabled(enabled bool) {
	s.uiEnabled = enabled
}
