package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

// TestResolveThinkingBudget_LegacyModel pins the mapping for models that only
// accept the legacy thinking: {type: "enabled", budget_tokens: N} shape (see
// supportsAdaptiveThinking): off => no thinking and the wire max_tokens is
// exactly the desired output; every other level gets its fixed approximate
// budget from legacyThinkingBudgets, and the wire max_tokens is desired +
// budget — the budget IS the reasoning headroom on these models, since
// thinking and output share the request pool — clamped to the model's output
// ceiling.
func TestResolveThinkingBudget_LegacyModel(t *testing.T) {
	const model = "claude-sonnet-4-5" // pre-4.6: legacy budget-based thinking; ceiling 64000

	cfg, outCfg, wire := resolveThinkingBudget(model, llmapi.ReasoningOff, 8192)
	if cfg != nil || outCfg != nil {
		t.Errorf("off: got thinking=%v output_config=%v, want nil,nil", cfg, outCfg)
	}
	if wire != 8192 {
		t.Errorf("off: wire = %d, want 8192 (desired, no headroom)", wire)
	}

	scaled := []struct {
		effort     llmapi.ReasoningEffort
		wantBudget int
	}{
		{llmapi.ReasoningLow, 1024},
		{llmapi.ReasoningMedium, 4096},
		{llmapi.ReasoningHigh, 8192},
		{llmapi.ReasoningXHigh, 16384}, // legacy models predate the tier; = max
		{llmapi.ReasoningMax, 16384},
	}
	for _, tc := range scaled {
		cfg, outCfg, wire := resolveThinkingBudget(model, tc.effort, 8192)
		if cfg == nil || cfg.Type != "enabled" || cfg.BudgetTokens != tc.wantBudget {
			t.Errorf("%v: got thinking=%+v, want {Type:enabled BudgetTokens:%d}", tc.effort, cfg, tc.wantBudget)
		}
		if outCfg != nil {
			t.Errorf("%v: got output_config=%+v, want nil on a legacy model", tc.effort, outCfg)
		}
		if want := 8192 + tc.wantBudget; wire != want {
			t.Errorf("%v: wire = %d, want %d (desired + budget)", tc.effort, wire, want)
		}
	}

	// The ceiling clamps the wire total; the budget stays intact beneath it
	// (every legacy budget is far below every known ceiling, so the API's
	// budget_tokens < max_tokens invariant holds by construction).
	if _, _, wire := resolveThinkingBudget(model, llmapi.ReasoningMax, 60000); wire != 64000 {
		t.Errorf("max @ desired=60000: wire = %d, want 64000 (ceiling clamp)", wire)
	}
}

// TestResolveThinkingBudget_AdaptiveModel pins the adaptive path: thinking
// {type: adaptive, display: summarized} + output_config.effort, with the wire
// max_tokens = desired + the effort tier's reasoning headroom (adaptive
// thinking shares the request pool with the output), clamped to the model's
// output ceiling. Off omits thinking entirely and reserves nothing.
func TestResolveThinkingBudget_AdaptiveModel(t *testing.T) {
	const model = "claude-opus-4-8" // adaptive; ceiling 128000

	cfg, outCfg, wire := resolveThinkingBudget(model, llmapi.ReasoningOff, 8192)
	if cfg != nil || outCfg != nil {
		t.Errorf("off: got thinking=%v output_config=%v, want nil,nil", cfg, outCfg)
	}
	if wire != 8192 {
		t.Errorf("off: wire = %d, want 8192 (desired, no headroom)", wire)
	}

	headrooms := []struct {
		effort       llmapi.ReasoningEffort
		wantEffort   string
		wantHeadroom int
	}{
		{llmapi.ReasoningLow, "low", 4096},
		{llmapi.ReasoningMedium, "medium", 8192},
		{llmapi.ReasoningHigh, "high", 16384},
		{llmapi.ReasoningXHigh, "xhigh", 65536},
		{llmapi.ReasoningMax, "max", 65536},
	}
	for _, tc := range headrooms {
		cfg, outCfg, wire := resolveThinkingBudget(model, tc.effort, 8192)
		if cfg == nil || cfg.Type != "adaptive" || cfg.BudgetTokens != 0 || cfg.Display != "summarized" {
			t.Errorf("%v: got thinking=%+v, want {Type:adaptive BudgetTokens:0 Display:summarized}", tc.effort, cfg)
		}
		if outCfg == nil || outCfg.Effort != tc.wantEffort {
			t.Errorf("%v: got output_config=%+v, want effort=%q", tc.effort, outCfg, tc.wantEffort)
		}
		if want := 8192 + tc.wantHeadroom; wire != want {
			t.Errorf("%v: wire = %d, want %d (desired + headroom)", tc.effort, wire, want)
		}
	}

	// Ceiling clamp: a large desired at max effort cannot push the wire total
	// past the model's real output ceiling.
	if _, _, wire := resolveThinkingBudget(model, llmapi.ReasoningMax, 100000); wire != 128000 {
		t.Errorf("max @ desired=100000: wire = %d, want 128000 (ceiling clamp)", wire)
	}
}

// TestResolveThinkingBudget_AdaptiveDefaultModel pins the Sonnet 5 shape:
// off sends an explicit thinking: {type: "disabled"} (omission would run
// adaptive thinking) and reserves no headroom; non-off levels behave like any
// other adaptive model.
func TestResolveThinkingBudget_AdaptiveDefaultModel(t *testing.T) {
	const model = "claude-sonnet-5"

	cfg, outCfg, wire := resolveThinkingBudget(model, llmapi.ReasoningOff, 8192)
	if cfg == nil || cfg.Type != "disabled" || cfg.BudgetTokens != 0 || cfg.Display != "" {
		t.Errorf("off: got thinking=%+v, want {Type:disabled}", cfg)
	}
	if outCfg != nil {
		t.Errorf("off: got output_config=%+v, want nil", outCfg)
	}
	if wire != 8192 {
		t.Errorf("off: wire = %d, want 8192 (disabled thinking reserves nothing)", wire)
	}

	cfg, outCfg, wire = resolveThinkingBudget(model, llmapi.ReasoningHigh, 8192)
	if cfg == nil || cfg.Type != "adaptive" || outCfg == nil || outCfg.Effort != "high" || wire != 24576 {
		t.Errorf("high: got thinking=%+v output_config=%+v wire=%d, want adaptive/high/24576", cfg, outCfg, wire)
	}
}

// TestResolveThinkingBudget_AlwaysThinkingModel pins the Fable/Mythos shape:
// those models cannot stop thinking, so caller-off resolves to adaptive at
// effort low with summarized display — and the wire budget reserves
// headroom(low) for it, because that thinking is real and shares the output
// pool. Keying headroom on the caller's "off" instead of the effective mode
// would let the unavoidable reasoning eat the content budget.
func TestResolveThinkingBudget_AlwaysThinkingModel(t *testing.T) {
	for _, model := range []string{"claude-fable-5", "claude-mythos-5"} {
		t.Run(model, func(t *testing.T) {
			cfg, outCfg, wire := resolveThinkingBudget(model, llmapi.ReasoningOff, 8192)
			if cfg == nil || cfg.Type != "adaptive" || cfg.Display != "summarized" {
				t.Errorf("off: got thinking=%+v, want {Type:adaptive Display:summarized}", cfg)
			}
			if outCfg == nil || outCfg.Effort != "low" {
				t.Errorf("off: got output_config=%+v, want effort=\"low\"", outCfg)
			}
			if wire != 8192+4096 {
				t.Errorf("off: wire = %d, want 12288 (desired + headroom(low) for the unavoidable thinking)", wire)
			}

			_, outCfg, wire = resolveThinkingBudget(model, llmapi.ReasoningMax, 8192)
			if outCfg == nil || outCfg.Effort != "max" || wire != 8192+65536 {
				t.Errorf("max: got output_config=%+v wire=%d, want max/73728", outCfg, wire)
			}
		})
	}
}

// TestResolveThinkingBudget_UnknownModelNoClamp pins the conservative default
// for a model this library cannot vouch for: legacy thinking shape (per
// supportsAdaptiveThinking's unrecognized-model behavior) and no known
// ceiling — the wire total is desired + budget, unclamped, preserving the
// pre-existing behavior for unknown IDs.
func TestResolveThinkingBudget_UnknownModelNoClamp(t *testing.T) {
	cfg, _, wire := resolveThinkingBudget("claude-future-model-xyz", llmapi.ReasoningHigh, 200000)
	if cfg == nil || cfg.Type != "enabled" || cfg.BudgetTokens != 8192 {
		t.Errorf("got thinking=%+v, want {Type:enabled BudgetTokens:8192}", cfg)
	}
	if wire != 208192 {
		t.Errorf("wire = %d, want 208192 (desired + budget, no clamp without a known ceiling)", wire)
	}
}

// TestResolveThinkingBudget_SmallCeilingModel pins the smallest known ceiling
// (Opus 4.1, 32000): it caps the wire total while the legacy budget beneath
// it stays valid — budget_tokens < max_tokens holds by construction, since
// the largest legacy budget (16384) is below the smallest known ceiling.
func TestResolveThinkingBudget_SmallCeilingModel(t *testing.T) {
	cfg, _, wire := resolveThinkingBudget("claude-opus-4-1", llmapi.ReasoningMax, 32768)
	if wire != 32000 {
		t.Errorf("wire = %d, want 32000 (ceiling clamp)", wire)
	}
	if cfg == nil || cfg.BudgetTokens != 16384 {
		t.Errorf("thinking = %+v, want BudgetTokens=16384 intact beneath the clamped wire", cfg)
	}
}
