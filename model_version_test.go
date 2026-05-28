package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

// TestParseModelVersion covers the recognized Anthropic model ID shapes plus
// unrecognized inputs that should report ok=false.
func TestParseModelVersion(t *testing.T) {
	cases := []struct {
		model     string
		wantMajor int
		wantMinor int
		wantOK    bool
	}{
		// Current naming scheme: claude-<family>-<major>-<minor>
		{"claude-opus-4-8", 4, 8, true},
		{"claude-opus-4-7", 4, 7, true},
		{"claude-opus-4-6", 4, 6, true},
		{"claude-opus-4-5", 4, 5, true},
		{"claude-opus-4-1", 4, 1, true},
		{"claude-opus-4-0", 4, 0, true},
		{"claude-sonnet-4-6", 4, 6, true},
		{"claude-sonnet-4-5", 4, 5, true},
		{"claude-haiku-4-5", 4, 5, true},

		// With trailing date suffix
		{"claude-haiku-4-5-20251001", 4, 5, true},
		{"claude-3-5-sonnet-20241022", 3, 5, true},
		{"claude-3-7-sonnet-20250219", 3, 7, true},
		{"claude-3-5-haiku-20241022", 3, 5, true},

		// Legacy naming with no minor component → minor reports as 0
		{"claude-3-opus-20240229", 3, 0, true},
		{"claude-3-sonnet-20240229", 3, 0, true},
		{"claude-3-haiku-20240307", 3, 0, true},

		// Dotted versions
		{"claude-2.1", 2, 1, true},
		{"claude-2.0", 2, 0, true},
		{"claude-instant-1.2", 1, 2, true},

		// Unrecognized — these should return ok=false so callers can apply a
		// conservative default.
		{"", 0, 0, false},
		{"gpt-4", 0, 0, false},
		{"claude", 0, 0, false}, // no leading "claude-"
		{"claude-future-model-xyz", 0, 0, false},
		{"claude-", 0, 0, false}, // empty after prefix
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			gotMajor, gotMinor, gotOK := parseModelVersion(tc.model)
			if gotOK != tc.wantOK {
				t.Fatalf("parseModelVersion(%q) ok=%v, want %v", tc.model, gotOK, tc.wantOK)
			}
			if !gotOK {
				return
			}
			if gotMajor != tc.wantMajor || gotMinor != tc.wantMinor {
				t.Errorf("parseModelVersion(%q) = (%d, %d), want (%d, %d)",
					tc.model, gotMajor, gotMinor, tc.wantMajor, tc.wantMinor)
			}
		})
	}
}

// TestSupportsSampling locks in the deprecation cutoff: models at version 4.7
// or newer must report false (per the Opus 4.7/4.8 docs), and unrecognized
// model IDs must also report false (conservative default — a 400 from a real
// model is a worse failure than dropping a sampling param for an unknown one).
func TestSupportsSampling(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		// Deprecated (>= 4.7)
		{"claude-opus-4-8", false},
		{"claude-opus-4-7", false},

		// Hypothetical future bumps follow the same rule family-agnostically.
		{"claude-sonnet-4-7", false},
		{"claude-haiku-4-7", false},
		{"claude-opus-5-0", false},

		// Still supported (<= 4.6)
		{"claude-opus-4-6", true},
		{"claude-opus-4-5", true},
		{"claude-opus-4-1", true},
		{"claude-opus-4-0", true},
		{"claude-sonnet-4-6", true},
		{"claude-sonnet-4-5", true},
		{"claude-haiku-4-5", true},
		{"claude-haiku-4-5-20251001", true},
		{"claude-3-7-sonnet-20250219", true},
		{"claude-3-5-sonnet-20241022", true},
		{"claude-3-opus-20240229", true},
		{"claude-2.1", true},
		{"claude-instant-1.2", true},

		// Unrecognized → false (conservative)
		{"", false},
		{"gpt-4", false},
		{"claude-future-model-xyz", false},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := supportsSampling(tc.model); got != tc.want {
				t.Errorf("supportsSampling(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// TestResolveSampling_UnsupportedModelOmitsAll verifies that on an unsupported
// model all three sampling params are zeroed (Temperature returns nil so the
// pointer omits the field; TopP and TopK return 0 so omitempty drops them).
func TestResolveSampling_UnsupportedModelOmitsAll(t *testing.T) {
	settings := &SampleSettings{
		Model:       "claude-opus-4-8",
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
	}
	override := llmapi.Sampling{Temperature: 0.5, TopP: 0.8, TopK: 20}

	temperature, topP, topK := resolveSampling(settings, override)

	if temperature != nil {
		t.Errorf("Temperature should be nil for opus-4-8, got %v", *temperature)
	}
	if topP != 0 {
		t.Errorf("TopP should be 0 for opus-4-8, got %v", topP)
	}
	if topK != 0 {
		t.Errorf("TopK should be 0 for opus-4-8, got %v", topK)
	}
}

// TestResolveSampling_SupportedModelKeepsZeroTemperature verifies that on a
// supported model an explicit Temperature=0 still produces a non-nil pointer.
// This is the core reason Temperature must be *float64: float64+omitempty
// would silently drop intentional deterministic sampling.
func TestResolveSampling_SupportedModelKeepsZeroTemperature(t *testing.T) {
	settings := &SampleSettings{
		Model:       "claude-sonnet-4-6",
		Temperature: 0.0,
	}
	temperature, _, _ := resolveSampling(settings, llmapi.Sampling{})

	if temperature == nil {
		t.Fatal("Temperature must be non-nil on supported model even when value is 0")
	}
	if *temperature != 0 {
		t.Errorf("Temperature = %v, want 0", *temperature)
	}
}

// TestResolveSampling_OverrideAppliedOnSupportedModel verifies that per-call
// overrides take precedence over conversation defaults on supported models.
func TestResolveSampling_OverrideAppliedOnSupportedModel(t *testing.T) {
	settings := &SampleSettings{
		Model:       "claude-sonnet-4-6",
		Temperature: 0.2,
		TopP:        0.5,
		TopK:        10,
	}
	override := llmapi.Sampling{Temperature: 0.7, TopP: 0.9, TopK: 50}

	temperature, topP, topK := resolveSampling(settings, override)
	if temperature == nil || *temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", temperature)
	}
	if topP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", topP)
	}
	if topK != 50 {
		t.Errorf("TopK = %v, want 50", topK)
	}
}

// TestResolveSampling_DefaultUsedWhenOverrideZero matches the existing
// "non-zero override wins" convention: a zero in llmapi.Sampling means
// "use the conversation's configured value."
func TestResolveSampling_DefaultUsedWhenOverrideZero(t *testing.T) {
	settings := &SampleSettings{
		Model:       "claude-sonnet-4-6",
		Temperature: 0.4,
		TopP:        0.6,
		TopK:        30,
	}
	temperature, topP, topK := resolveSampling(settings, llmapi.Sampling{})

	if temperature == nil || *temperature != 0.4 {
		t.Errorf("Temperature = %v, want 0.4 from settings", temperature)
	}
	if topP != 0.6 {
		t.Errorf("TopP = %v, want 0.6 from settings", topP)
	}
	if topK != 30 {
		t.Errorf("TopK = %v, want 30 from settings", topK)
	}
}

// TestResolveSampling_UnrecognizedModelOmitsAll documents that the
// conservative default (unrecognized model → no sampling params) is what
// callers actually see. A 400 from a real future model is a worse failure
// than dropping a sampling param for an unknown ID.
func TestResolveSampling_UnrecognizedModelOmitsAll(t *testing.T) {
	settings := &SampleSettings{
		Model:       "some-other-vendor-model",
		Temperature: 0.5,
		TopP:        0.9,
		TopK:        40,
	}
	temperature, topP, topK := resolveSampling(settings, llmapi.Sampling{})

	if temperature != nil {
		t.Errorf("Temperature should be nil for unrecognized model, got %v", *temperature)
	}
	if topP != 0 {
		t.Errorf("TopP should be 0 for unrecognized model, got %v", topP)
	}
	if topK != 0 {
		t.Errorf("TopK should be 0 for unrecognized model, got %v", topK)
	}
}
