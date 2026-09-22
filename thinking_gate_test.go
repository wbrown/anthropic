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
	MaxTokens int `json:"max_tokens"`
	Thinking  *struct {
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

// TestSend_WireMaxTokens pins the wire max_tokens computation across all
// three send paths: the request's max_tokens is the desired output (per-call
// Sampling.DesiredOutputTokens, else Settings.MaxTokens) plus the effective
// thinking mode's reasoning headroom, clamped to the model's output ceiling.
// Settings.MaxTokens is the default DESIRED OUTPUT, not the wire value — the
// wire value is computed from it.
func TestSend_WireMaxTokens(t *testing.T) {
	send := func(t *testing.T, model string, settingsMaxTokens int, sampling llmapi.Sampling) []byte {
		t.Helper()
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = model
		conv.Settings.MaxTokens = settingsMaxTokens

		if _, _, _, _, _, _, err := conv.Send("hello", sampling); err != nil {
			t.Fatalf("Send: %v", err)
		}
		return captured
	}

	t.Run("adaptive effort adds headroom to the settings default", func(t *testing.T) {
		body := send(t, "claude-opus-4-8", 8192, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
		if got := decodeThinkingFields(t, body, "opus-4-8 high").MaxTokens; got != 24576 {
			t.Errorf("max_tokens = %d, want 24576 (8192 desired + 16384 high headroom)", got)
		}
	})

	t.Run("per-call DesiredOutputTokens overrides the settings default", func(t *testing.T) {
		body := send(t, "claude-opus-4-8", 8192, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh, DesiredOutputTokens: 4096})
		if got := decodeThinkingFields(t, body, "opus-4-8 high desired-override").MaxTokens; got != 20480 {
			t.Errorf("max_tokens = %d, want 20480 (4096 desired + 16384 high headroom)", got)
		}
	})

	t.Run("disabled thinking reserves nothing on sonnet-5", func(t *testing.T) {
		body := send(t, "claude-sonnet-5", 8192, llmapi.Sampling{})
		if got := decodeThinkingFields(t, body, "sonnet-5 off").MaxTokens; got != 8192 {
			t.Errorf("max_tokens = %d, want 8192 (desired only; thinking disabled)", got)
		}
	})

	t.Run("always-thinking model reserves low headroom at caller-off", func(t *testing.T) {
		body := send(t, "claude-fable-5", 8192, llmapi.Sampling{})
		if got := decodeThinkingFields(t, body, "fable-5 off").MaxTokens; got != 12288 {
			t.Errorf("max_tokens = %d, want 12288 (8192 desired + 4096 low headroom for unavoidable thinking)", got)
		}
	})

	t.Run("ceiling clamps the wire total", func(t *testing.T) {
		body := send(t, "claude-opus-4-8", 100000, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningMax})
		if got := decodeThinkingFields(t, body, "opus-4-8 max clamp").MaxTokens; got != 128000 {
			t.Errorf("max_tokens = %d, want 128000 (ceiling)", got)
		}
	})

	t.Run("SendStreaming computes the same wire budget", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.MaxTokens = 8192

		if _, _, _, _, _, _, err := conv.SendStreaming("hello", llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh, DesiredOutputTokens: 4096}, nil); err != nil {
			t.Fatalf("SendStreaming: %v", err)
		}
		if got := decodeThinkingFields(t, captured, "streaming wire budget").MaxTokens; got != 20480 {
			t.Errorf("max_tokens = %d, want 20480 (4096 desired + 16384 high headroom)", got)
		}
	})

	t.Run("SendRichStreaming computes the same wire budget", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.MaxTokens = 8192

		content := []llmapi.ContentBlock{llmapi.NewTextBlock("hello")}
		if _, err := conv.SendRichStreaming(content, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh, DesiredOutputTokens: 4096}, nil); err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		if got := decodeThinkingFields(t, captured, "rich streaming wire budget").MaxTokens; got != 20480 {
			t.Errorf("max_tokens = %d, want 20480 (4096 desired + 16384 high headroom)", got)
		}
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
// path.
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
