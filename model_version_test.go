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

// TestSupportsAdaptiveThinking locks in the adaptive-thinking cutoff: models
// at version 4.6 or newer report true, models older than that report false,
// and unrecognized model IDs report false (conservative default — this
// library's pre-existing budget-based thinking behavior for a model it
// doesn't recognize, rather than guessing at a request shape that may not be
// supported at all).
func TestSupportsAdaptiveThinking(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		// Supports adaptive thinking (>= 4.6)
		{"claude-opus-4-8", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-6", true},
		{"claude-sonnet-5", true},
		{"claude-fable-5", true},
		{"claude-mythos-5", true},

		// Hypothetical future bumps follow the same rule family-agnostically.
		{"claude-haiku-5-0", true},
		{"claude-opus-5-0", true},

		// Legacy budget-based thinking only (< 4.6)
		{"claude-opus-4-5", false},
		{"claude-opus-4-1", false},
		{"claude-opus-4-0", false},
		{"claude-sonnet-4-5", false},
		{"claude-haiku-4-5", false},
		{"claude-haiku-4-5-20251001", false},
		{"claude-3-7-sonnet-20250219", false},
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-opus-20240229", false},
		{"claude-2.1", false},
		{"claude-instant-1.2", false},

		// Unrecognized → false (conservative)
		{"", false},
		{"gpt-4", false},
		{"claude-future-model-xyz", false},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := supportsAdaptiveThinking(tc.model); got != tc.want {
				t.Errorf("supportsAdaptiveThinking(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// TestMustAlwaysThink locks in the always-on-thinking family: only Fable 5
// and Mythos 5 name a model that cannot be made to stop thinking. This is a
// family match, not a version cutoff — a hypothetical claude-opus-5-0 must
// NOT be swept in just because it shares Fable/Mythos's major version.
func TestMustAlwaysThink(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"claude-fable-5", true},
		{"claude-mythos-5", true},

		// Same major version, different family — must NOT match.
		{"claude-sonnet-5", false},
		{"claude-opus-5-0", false},

		// Everything else this library recognizes can genuinely stop thinking.
		{"claude-opus-4-8", false},
		{"claude-opus-4-7", false},
		{"claude-opus-4-6", false},
		{"claude-sonnet-4-6", false},
		{"claude-sonnet-4-5", false},
		{"claude-haiku-4-5", false},
		{"", false},
		{"gpt-4", false},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := mustAlwaysThink(tc.model); got != tc.want {
				t.Errorf("mustAlwaysThink(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// TestDefaultsToAdaptiveThinking locks in which disableable models run
// adaptive thinking when the request OMITS the thinking field. Claude Sonnet 5
// does (omission = adaptive on); Opus 4.6-4.8 and Sonnet 4.6 do not (omission
// = thinking off). The always-thinking family (Fable/Mythos) is out of this
// predicate's scope — those cannot be disabled at all and report false here;
// mustAlwaysThink handles them first.
func TestDefaultsToAdaptiveThinking(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		// Sonnet 5+: omitting the thinking field runs adaptive thinking.
		{"claude-sonnet-5", true},
		{"claude-sonnet-5-1", true},

		// Sonnet 4.x: omission = off.
		{"claude-sonnet-4-6", false},
		{"claude-sonnet-4-5", false},

		// Opus 4.6-4.8: omission = off (explicitly documented for 4.7/4.8).
		{"claude-opus-4-8", false},
		{"claude-opus-4-7", false},
		{"claude-opus-4-6", false},

		// Always-thinking family: not this predicate's concern (cannot disable).
		{"claude-fable-5", false},
		{"claude-mythos-5", false},

		// Other families and unrecognized IDs → false (conservative).
		{"claude-haiku-4-5", false},
		{"claude-opus-5-0", false},
		{"", false},
		{"gpt-4", false},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := defaultsToAdaptiveThinking(tc.model); got != tc.want {
				t.Errorf("defaultsToAdaptiveThinking(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// TestModelOutputCeiling locks in each model generation's real per-request
// output ceiling, per Anthropic's models overview (synchronous Messages API):
// 128000 for every 4.6+ model (Fable/Mythos 5, Opus 4.6-4.8, Sonnet 4.6,
// Sonnet 5), 64000 for the 4.5 tier (Sonnet 4.5, Opus 4.5, Haiku 4.5), 32000
// for Opus 4.1. Generations older than that are retired at the API, and
// unrecognized IDs are unknown — both report 0, meaning "no ceiling known, do
// not clamp", which preserves this library's pre-existing behavior for
// models it cannot vouch for.
func TestModelOutputCeiling(t *testing.T) {
	cases := []struct {
		model string
		want  int
	}{
		// 4.6+ (including all major-5 families): 128000
		{"claude-fable-5", 128000},
		{"claude-mythos-5", 128000},
		{"claude-sonnet-5", 128000},
		{"claude-opus-4-8", 128000},
		{"claude-opus-4-7", 128000},
		{"claude-opus-4-6", 128000},
		{"claude-sonnet-4-6", 128000},

		// 4.5 tier: 64000
		{"claude-sonnet-4-5", 64000},
		{"claude-opus-4-5", 64000},
		{"claude-haiku-4-5", 64000},
		{"claude-haiku-4-5-20251001", 64000},

		// 4.1: 32000
		{"claude-opus-4-1", 32000},

		// Retired generations and unrecognized IDs: unknown → no clamp.
		{"claude-opus-4-0", 0},
		{"claude-3-7-sonnet-20250219", 0},
		{"claude-3-5-sonnet-20241022", 0},
		{"claude-3-opus-20240229", 0},
		{"claude-2.1", 0},
		{"", 0},
		{"gpt-4", 0},
		{"claude-future-model-xyz", 0},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := modelOutputCeiling(tc.model); got != tc.want {
				t.Errorf("modelOutputCeiling(%q) = %d, want %d", tc.model, got, tc.want)
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

	temperature, topP, topK := resolveSampling(settings, override, false)

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
	temperature, topP, topK := resolveSampling(settings, llmapi.Sampling{}, false)

	if temperature == nil {
		t.Fatal("Temperature must be non-nil on supported model even when value is 0")
	}
	if *temperature != 0 {
		t.Errorf("Temperature = %v, want 0", *temperature)
	}
	if topP != 0 {
		t.Errorf("TopP = %v, want 0 (unset in settings)", topP)
	}
	if topK != 0 {
		t.Errorf("TopK = %v, want 0 (unset in settings)", topK)
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

	temperature, topP, topK := resolveSampling(settings, override, false)
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
	temperature, topP, topK := resolveSampling(settings, llmapi.Sampling{}, false)

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
	temperature, topP, topK := resolveSampling(settings, llmapi.Sampling{}, false)

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

// TestResolveSampling_ThinkingActiveOmitsAll pins the constraint the real API
// enforces (discovered via TestSendRich_AdaptiveThinking_RealReasoning in
// api_test.go, which got a live 400 before this gate existed): "temperature
// may only be set to 1 when thinking is enabled or in adaptive mode". A model
// that fully supports sampling (claude-sonnet-4-6, well under the 4.7+
// supportsSampling cutoff) must still omit all three sampling params — even
// an explicit override — the moment thinking is active, exactly as it would
// for an unsupported model.
func TestResolveSampling_ThinkingActiveOmitsAll(t *testing.T) {
	settings := &SampleSettings{
		Model:       "claude-sonnet-4-6",
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
	}
	override := llmapi.Sampling{Temperature: 0.5, TopP: 0.8, TopK: 20, ReasoningEffort: llmapi.ReasoningHigh}

	temperature, topP, topK := resolveSampling(settings, override, true)

	if temperature != nil {
		t.Errorf("Temperature should be nil when thinking is active, got %v", *temperature)
	}
	if topP != 0 {
		t.Errorf("TopP should be 0 when thinking is active, got %v", topP)
	}
	if topK != 0 {
		t.Errorf("TopK should be 0 when thinking is active, got %v", topK)
	}
}
