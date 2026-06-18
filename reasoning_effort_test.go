package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

// TestThinkingForEffort pins the reasoning-effort -> extended-thinking-budget
// mapping: off => no thinking; low => the 1024 floor; medium/high/max => 1/4, 1/2,
// 3/4 of MaxTokens; and a MaxTokens too small to fit a >=1024 budget under
// max_tokens is an error (you cannot ask for reasoning with <1024 budget).
func TestThinkingForEffort(t *testing.T) {
	if cfg, err := thinkingForEffort(llmapi.ReasoningOff, 8192); err != nil || cfg != nil {
		t.Errorf("off: got cfg=%v err=%v, want nil,nil", cfg, err)
	}
	if cfg, err := thinkingForEffort(llmapi.ReasoningLow, 8192); err != nil || cfg == nil || cfg.Type != "enabled" || cfg.BudgetTokens != 1024 {
		t.Errorf("low: got %+v err=%v, want enabled/1024", cfg, err)
	}

	scaled := []struct {
		effort     llmapi.ReasoningEffort
		maxTokens  int
		wantBudget int
	}{
		{llmapi.ReasoningMedium, 8192, 2048},
		{llmapi.ReasoningHigh, 8192, 4096},
		{llmapi.ReasoningMax, 8192, 6144},
	}
	for _, tc := range scaled {
		cfg, err := thinkingForEffort(tc.effort, tc.maxTokens)
		if err != nil || cfg == nil || cfg.BudgetTokens != tc.wantBudget {
			t.Errorf("%v @ max=%d: got %+v err=%v, want budget %d", tc.effort, tc.maxTokens, cfg, err, tc.wantBudget)
		}
	}

	// MaxTokens too small to fit a >=1024 budget under max_tokens must error.
	if _, err := thinkingForEffort(llmapi.ReasoningHigh, 1024); err == nil {
		t.Error("max_tokens=1024: want error (no room for a >=1024 budget under max_tokens)")
	}
	if _, err := thinkingForEffort(llmapi.ReasoningLow, 512); err == nil {
		t.Error("max_tokens=512: want error")
	}
}
