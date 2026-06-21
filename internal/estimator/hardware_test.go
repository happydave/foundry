package estimator

import (
	"os"
	"path/filepath"
	"testing"
)

func writeAttr(t *testing.T, dir, name, value string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(value), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestResolveCardIdentity_Precedence(t *testing.T) {
	t.Run("product_name wins", func(t *testing.T) {
		dir := t.TempDir()
		writeAttr(t, dir, "product_name", "Radeon RX 7900 XTX\n")
		writeAttr(t, dir, "vendor", "0x1002\n")
		writeAttr(t, dir, "device", "0x744c\n")
		if got := resolveCardIdentity(dir, "card2"); got != "Radeon RX 7900 XTX" {
			t.Fatalf("got %q, want product name", got)
		}
	})

	t.Run("vendor:device when product_name absent", func(t *testing.T) {
		dir := t.TempDir()
		writeAttr(t, dir, "vendor", "0x1002\n")
		writeAttr(t, dir, "device", "0x164E\n")
		if got := resolveCardIdentity(dir, "card2"); got != "1002:164e" {
			t.Fatalf("got %q, want 1002:164e", got)
		}
	})

	t.Run("empty product_name ignored", func(t *testing.T) {
		dir := t.TempDir()
		writeAttr(t, dir, "product_name", "\n")
		writeAttr(t, dir, "vendor", "0x1002\n")
		writeAttr(t, dir, "device", "0x164e\n")
		if got := resolveCardIdentity(dir, "card2"); got != "1002:164e" {
			t.Fatalf("got %q, want 1002:164e", got)
		}
	})

	t.Run("falls back to card name", func(t *testing.T) {
		dir := t.TempDir()
		if got := resolveCardIdentity(dir, "card2"); got != "card2" {
			t.Fatalf("got %q, want card2", got)
		}
	})
}

func TestCardIndex(t *testing.T) {
	cases := map[string]int{"card0": 0, "card2": 2, "card11": 11, "bogus": -1, "": -1}
	for in, want := range cases {
		if got := cardIndex(in); got != want {
			t.Fatalf("cardIndex(%q) = %d, want %d", in, got, want)
		}
	}
}
