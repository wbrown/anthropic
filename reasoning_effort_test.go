package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

// TestThinkingForEffort_LegacyModel pins the reasoning-effort mapping for
// models that don't support adaptive thinking (see supportsAdaptiveThinking):
// off => no thinking, no output_config; every other level gets its fixed
// approximate budget from legacyThinkingBudgets, clamped to stay under
// maxTokens and floored at the API's stated 1024 minimum.
func TestThinkingForEffort_LegacyModel(t *testing.T) {
	const model = "claude-sonnet-4-5" // pre-4.6: legacy budget-based thinking only

	if cfg, outCfg := thinkingForEffort(model, llmapi.ReasoningOff, 8192); cfg != nil || outCfg != nil {
		t.Errorf("off: got thinking=%v output_config=%v, want nil,nil", cfg, outCfg)
	}

	// No clamping needed: legacyThinkingBudgets values all fit comfortably
	// under maxTokens=32768.
	scaled := []struct {
		effort     llmapi.ReasoningEffort
		wantBudget int
	}{
		{llmapi.ReasoningLow, 1024},
		{llmapi.ReasoningMedium, 4096},
		{llmapi.ReasoningHigh, 8192},
		{llmapi.ReasoningMax, 16384},
	}
	for _, tc := range scaled {
		cfg, outCfg := thinkingForEffort(model, tc.effort, 32768)
		if cfg == nil || cfg.Type != "enabled" || cfg.BudgetTokens != tc.wantBudget {
			t.Errorf("%v: got thinking=%+v, want {Type:enabled BudgetTokens:%d}", tc.effort, cfg, tc.wantBudget)
		}
		if outCfg != nil {
			t.Errorf("%v: got output_config=%+v, want nil on a legacy model", tc.effort, outCfg)
		}
	}

	// A maxTokens smaller than the level's approximate budget clamps down
	// to fit under it, rather than erroring.
	if cfg, _ := thinkingForEffort(model, llmapi.ReasoningMax, 2048); cfg == nil || cfg.BudgetTokens != 2047 {
		t.Errorf("max @ maxTokens=2048: got %+v, want BudgetTokens=2047 (maxTokens-1)", cfg)
	}

	// The 1024 floor can still exceed maxTokens when maxTokens itself is
	// below the API's minimum — that's a genuinely invalid request (Anthropic
	// requires budget_tokens < max_tokens), and this library no longer
	// pre-validates that case (see thinkingForEffort's doc comment); it
	// surfaces as a real 400 from the API rather than a Go-level error.
	if cfg, _ := thinkingForEffort(model, llmapi.ReasoningLow, 1024); cfg == nil || cfg.BudgetTokens != 1024 {
		t.Errorf("low @ maxTokens=1024: got %+v, want BudgetTokens=1024 (floor wins over the clamp)", cfg)
	}
}

// thinkingForEffortNonOffCases is shared by both adaptive-model tests below:
// regardless of whether ReasoningOff also happens to think (mustAlwaysThink),
// every explicit non-off level maps to thinking: {type: "adaptive",
// display: "summarized"} with output_config.effort set to the level's own
// wire value (llmapi.ReasoningEffort.String() already returns
// "low"/"medium"/"high"/"max" — Anthropic's own effort vocabulary — so no
// translation is needed).
func thinkingForEffortNonOffCases(t *testing.T, model string) {
	t.Helper()
	cases := []struct {
		effort     llmapi.ReasoningEffort
		wantEffort string
	}{
		{llmapi.ReasoningLow, "low"},
		{llmapi.ReasoningMedium, "medium"},
		{llmapi.ReasoningHigh, "high"},
		{llmapi.ReasoningMax, "max"},
	}
	for _, tc := range cases {
		cfg, outCfg := thinkingForEffort(model, tc.effort, 8192)
		if cfg == nil || cfg.Type != "adaptive" || cfg.BudgetTokens != 0 || cfg.Display != "summarized" {
			t.Errorf("%v: got thinking=%+v, want {Type:adaptive BudgetTokens:0 Display:summarized}", tc.effort, cfg)
		}
		if outCfg == nil || outCfg.Effort != tc.wantEffort {
			t.Errorf("%v: got output_config=%+v, want effort=%q", tc.effort, outCfg, tc.wantEffort)
		}
	}
}

// TestThinkingForEffort_AdaptiveModel pins the reasoning-effort mapping for
// adaptive-thinking models where omitting the thinking field genuinely turns
// thinking off (Opus 4.6-4.8 and Sonnet 4.6 run without thinking when the
// request carries no thinking field): off => no thinking, no output_config.
func TestThinkingForEffort_AdaptiveModel(t *testing.T) {
	models := []string{
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-sonnet-4-6",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			if cfg, outCfg := thinkingForEffort(model, llmapi.ReasoningOff, 8192); cfg != nil || outCfg != nil {
				t.Errorf("off: got thinking=%v output_config=%v, want nil,nil", cfg, outCfg)
			}
			thinkingForEffortNonOffCases(t, model)
		})
	}
}

// TestThinkingForEffort_AdaptiveDefaultModel pins the mapping for models where
// OMITTING the thinking field runs adaptive thinking by default (Claude Sonnet
// 5 — unlike Opus 4.7/4.8, whose omission-default is off). ReasoningOff must
// therefore send an explicit thinking: {type: "disabled"} — omitting the field
// would silently run adaptive thinking: billed, slow, and invisible (no
// display parameter is sent, so the reasoning streams as empty-text deltas).
// Non-off levels behave identically to any other adaptive-thinking model.
func TestThinkingForEffort_AdaptiveDefaultModel(t *testing.T) {
	const model = "claude-sonnet-5"

	cfg, outCfg := thinkingForEffort(model, llmapi.ReasoningOff, 8192)
	if cfg == nil || cfg.Type != "disabled" || cfg.BudgetTokens != 0 || cfg.Display != "" {
		t.Errorf("off: got thinking=%+v, want {Type:disabled} (omission runs adaptive by default on this model)", cfg)
	}
	if outCfg != nil {
		t.Errorf("off: got output_config=%+v, want nil (no effort when thinking is disabled)", outCfg)
	}
	thinkingForEffortNonOffCases(t, model)
}

// TestThinkingForEffort_AlwaysThinkingModel pins the mustAlwaysThink branch:
// Claude Fable 5 and Claude Mythos 5 think unconditionally, so even
// ReasoningOff must still request display: "summarized" — otherwise the
// unavoidable, billed reasoning comes back as empty-text thinking blocks
// with zero visibility. Off requests the lowest effort, since the caller
// didn't ask for reasoning at all; non-off levels behave identically to any
// other adaptive-thinking model.
func TestThinkingForEffort_AlwaysThinkingModel(t *testing.T) {
	models := []string{"claude-fable-5", "claude-mythos-5"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			cfg, outCfg := thinkingForEffort(model, llmapi.ReasoningOff, 8192)
			if cfg == nil || cfg.Type != "adaptive" || cfg.BudgetTokens != 0 || cfg.Display != "summarized" {
				t.Errorf("off: got thinking=%+v, want {Type:adaptive BudgetTokens:0 Display:summarized}", cfg)
			}
			if outCfg == nil || outCfg.Effort != "low" {
				t.Errorf("off: got output_config=%+v, want effort=\"low\"", outCfg)
			}
			thinkingForEffortNonOffCases(t, model)
		})
	}
}
