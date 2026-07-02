package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/wbrown/llmapi"
)

// These tests verify the wire-level outcome of the adaptive-thinking gate:
// for each of the three send paths (sendInternal via Send, SendStreaming, and
// SendRichStreaming), a requested reasoning effort must produce thinking:
// {type:"adaptive"} + output_config.effort on models that support adaptive
// thinking, and the legacy thinking:{type:"enabled",budget_tokens:N} shape
// (with no output_config at all) on older models. ReasoningOff must omit
// both keys on every model. Reuses the stub servers and Settings.Model
// pattern from sampling_gate_test.go.

// decodedThinkingFields is the subset of the request body this file inspects.
type decodedThinkingFields struct {
	Thinking *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
		Display      string `json:"display"`
	} `json:"thinking"`
	OutputConfig *struct {
		Effort string `json:"effort"`
	} `json:"output_config"`
}

func decodeThinkingFields(t *testing.T, body []byte, context string) decodedThinkingFields {
	t.Helper()
	var decoded decodedThinkingFields
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("%s: unmarshal request body: %v\nbody=%s", context, err, string(body))
	}
	return decoded
}

// assertThinkingAbsent fails the test if the body contains a thinking or
// output_config key at all.
func assertThinkingAbsent(t *testing.T, body []byte, context string) {
	t.Helper()
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("%s: unmarshal request body: %v\nbody=%s", context, err, string(body))
	}
	for _, key := range []string{"thinking", "output_config"} {
		if _, present := decoded[key]; present {
			t.Errorf("%s: request body must not contain %q for ReasoningOff; body=%s", context, key, string(body))
		}
	}
}

// assertAdaptiveThinking asserts the body carries thinking:{type:"adaptive",
// display:"summarized"} (no budget_tokens) and output_config.effort ==
// wantEffort. display:"summarized" is required, not cosmetic: Anthropic
// defaults thinking.display to "omitted" on Fable 5/Mythos 5/Opus 4.7/4.8/
// Sonnet 5, which streams real-but-empty-text thinking blocks — a request
// missing this field "succeeds" but produces no observable reasoning.
func assertAdaptiveThinking(t *testing.T, body []byte, wantEffort, context string) {
	t.Helper()
	decoded := decodeThinkingFields(t, body, context)
	if decoded.Thinking == nil {
		t.Fatalf("%s: thinking must be present; body=%s", context, string(body))
	}
	if decoded.Thinking.Type != "adaptive" {
		t.Errorf("%s: thinking.type = %q, want \"adaptive\"", context, decoded.Thinking.Type)
	}
	if decoded.Thinking.BudgetTokens != 0 {
		t.Errorf("%s: thinking.budget_tokens = %d, want absent/0 on an adaptive-thinking model", context, decoded.Thinking.BudgetTokens)
	}
	if decoded.Thinking.Display != "summarized" {
		t.Errorf("%s: thinking.display = %q, want \"summarized\" (omitted default streams empty thinking text)", context, decoded.Thinking.Display)
	}
	if decoded.OutputConfig == nil {
		t.Fatalf("%s: output_config must be present; body=%s", context, string(body))
	}
	if decoded.OutputConfig.Effort != wantEffort {
		t.Errorf("%s: output_config.effort = %q, want %q", context, decoded.OutputConfig.Effort, wantEffort)
	}
}

// assertThinkingDisabled asserts the body carries an explicit
// thinking:{type:"disabled"} (no budget_tokens, no display) and no
// output_config key. Required on models whose omission-default is adaptive-on
// (Claude Sonnet 5): omitting the field there silently runs adaptive thinking
// — billed, slow, and invisible, since no display parameter is sent.
func assertThinkingDisabled(t *testing.T, body []byte, context string) {
	t.Helper()
	decoded := decodeThinkingFields(t, body, context)
	if decoded.Thinking == nil {
		t.Fatalf("%s: thinking must be present (explicit disabled); body=%s", context, string(body))
	}
	if decoded.Thinking.Type != "disabled" {
		t.Errorf("%s: thinking.type = %q, want \"disabled\"", context, decoded.Thinking.Type)
	}
	if decoded.Thinking.BudgetTokens != 0 {
		t.Errorf("%s: thinking.budget_tokens = %d, want absent/0 when disabled", context, decoded.Thinking.BudgetTokens)
	}
	if decoded.Thinking.Display != "" {
		t.Errorf("%s: thinking.display = %q, want absent when disabled", context, decoded.Thinking.Display)
	}
	if decoded.OutputConfig != nil {
		t.Errorf("%s: output_config must be absent when thinking is disabled; got %+v", context, decoded.OutputConfig)
	}
}

// assertLegacyThinking asserts the body carries the legacy
// thinking:{type:"enabled",budget_tokens:N} shape and no output_config key.
func assertLegacyThinking(t *testing.T, body []byte, wantBudget int, context string) {
	t.Helper()
	decoded := decodeThinkingFields(t, body, context)
	if decoded.Thinking == nil {
		t.Fatalf("%s: thinking must be present; body=%s", context, string(body))
	}
	if decoded.Thinking.Type != "enabled" {
		t.Errorf("%s: thinking.type = %q, want \"enabled\"", context, decoded.Thinking.Type)
	}
	if decoded.Thinking.BudgetTokens != wantBudget {
		t.Errorf("%s: thinking.budget_tokens = %d, want %d", context, decoded.Thinking.BudgetTokens, wantBudget)
	}
	if decoded.OutputConfig != nil {
		t.Errorf("%s: output_config must be absent on a legacy-thinking model; got %+v", context, decoded.OutputConfig)
	}
}

// TestSendInternal_GatesThinkingByModel exercises the sendInternal path
// (reached via Conversation.Send).
func TestSendInternal_GatesThinkingByModel(t *testing.T) {
	t.Run("opus-4-8 uses adaptive thinking", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.MaxTokens = 8192

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertAdaptiveThinking(t, captured, "high", "opus-4-8 Send")
	})

	t.Run("sonnet-4-5 uses legacy budget-based thinking", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-5"
		conv.Settings.MaxTokens = 32768

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertLegacyThinking(t, captured, legacyThinkingBudgets[llmapi.ReasoningHigh], "sonnet-4-5 Send")
	})

	t.Run("ReasoningOff omits thinking on either model tier", func(t *testing.T) {
		for _, model := range []string{"claude-opus-4-8", "claude-sonnet-4-5"} {
			var captured []byte
			server := stubMessagesServer(t, &captured)

			conv := NewConversation("sys")
			conv.ApiToken = "test-token"
			conv.SetEndpoint(server.URL)
			conv.Settings.Model = model
			conv.Settings.MaxTokens = 8192

			if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
				server.Close()
				t.Fatalf("Send (%s): %v", model, err)
			}
			assertThinkingAbsent(t, captured, model+" Send with ReasoningOff")
			server.Close()
		}
	})

	t.Run("ReasoningOff sends explicit disabled on an adaptive-default model", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-5"
		conv.Settings.MaxTokens = 8192

		// Sonnet 5 runs adaptive thinking when the field is OMITTED, so
		// ReasoningOff must say "disabled" out loud rather than staying silent.
		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertThinkingDisabled(t, captured, "sonnet-5 Send with ReasoningOff")
	})

	t.Run("ReasoningOff still requests summarized thinking on an always-thinking model", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-fable-5"
		conv.Settings.MaxTokens = 8192

		// No ReasoningEffort at all — this is what "scenario doesn't pass any
		// reasoning flags" looks like at the wire level.
		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		// Fable 5 thinks regardless of what's sent; omitting thinking would
		// only forfeit visibility into reasoning that happens (and is billed)
		// either way. Effort is "low" since the caller asked for nothing.
		assertAdaptiveThinking(t, captured, "low", "fable-5 Send with ReasoningOff")
	})
}

// TestThinkingConfigActive pins the sampling-gate predicate: an explicit
// disabled config means thinking is NOT active (sampling params may still be
// sent on models that accept them), while enabled/adaptive configs mean it is.
func TestThinkingConfigActive(t *testing.T) {
	cases := []struct {
		name string
		cfg  *ThinkingConfig
		want bool
	}{
		{"nil (field omitted)", nil, false},
		{"explicit disabled", &ThinkingConfig{Type: "disabled"}, false},
		{"adaptive", &ThinkingConfig{Type: "adaptive"}, true},
		{"legacy enabled", &ThinkingConfig{Type: "enabled", BudgetTokens: 2048}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.active(); got != tc.want {
				t.Errorf("(%+v).active() = %v, want %v", tc.cfg, got, tc.want)
			}
		})
	}
}

// TestSendStreaming_GatesThinkingByModel exercises the SendStreaming path.
func TestSendStreaming_GatesThinkingByModel(t *testing.T) {
	t.Run("opus-4-8 uses adaptive thinking", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.MaxTokens = 8192

		if _, _, _, _, _, _, err := conv.SendStreaming("hello", llmapi.Sampling{ReasoningEffort: llmapi.ReasoningMax}, nil); err != nil {
			t.Fatalf("SendStreaming: %v", err)
		}
		assertAdaptiveThinking(t, captured, "max", "opus-4-8 SendStreaming")
	})

	t.Run("sonnet-4-5 uses legacy budget-based thinking", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-5"
		conv.Settings.MaxTokens = 32768

		if _, _, _, _, _, _, err := conv.SendStreaming("hello", llmapi.Sampling{ReasoningEffort: llmapi.ReasoningMax}, nil); err != nil {
			t.Fatalf("SendStreaming: %v", err)
		}
		assertLegacyThinking(t, captured, legacyThinkingBudgets[llmapi.ReasoningMax], "sonnet-4-5 SendStreaming")
	})
}

// TestSendRichStreaming_GatesThinkingByModel exercises the SendRichStreaming
// path, which has its own local copy of the thinking-resolution block.
func TestSendRichStreaming_GatesThinkingByModel(t *testing.T) {
	t.Run("opus-4-8 uses adaptive thinking", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.MaxTokens = 8192

		content := []llmapi.ContentBlock{llmapi.NewTextBlock("hello")}
		if _, err := conv.SendRichStreaming(content, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningMedium}, nil); err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		assertAdaptiveThinking(t, captured, "medium", "opus-4-8 SendRichStreaming")
	})

	t.Run("sonnet-4-5 uses legacy budget-based thinking", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-5"
		conv.Settings.MaxTokens = 32768

		content := []llmapi.ContentBlock{llmapi.NewTextBlock("hello")}
		if _, err := conv.SendRichStreaming(content, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningMedium}, nil); err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		assertLegacyThinking(t, captured, legacyThinkingBudgets[llmapi.ReasoningMedium], "sonnet-4-5 SendRichStreaming")
	})
}
