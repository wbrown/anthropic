package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

// These tests pin the account a RichResponse carries beside its content on
// both rich send paths: the max_tokens the request carried on the wire and
// the server's own stop reason, which this API reports in the vocabulary
// StopReason already uses. Anthropic's usage block does not attribute output
// tokens by channel, so the split stays unknown.

// TestSendRich_AccountsForTheRequest drives the non-streaming rich path
// against the stub server and checks the account against the captured body.
func TestSendRich_AccountsForTheRequest(t *testing.T) {
	var captured []byte
	server := stubMessagesServer(t, &captured)
	defer server.Close()

	conv := NewConversation("sys")
	conv.ApiToken = "test-token"
	conv.SetEndpoint(server.URL)
	conv.Settings.Model = "claude-opus-4-8"
	conv.Settings.MaxTokens = 8192

	rr, err := conv.SendRich([]llmapi.ContentBlock{llmapi.NewTextBlock("hello")}, llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh})
	if err != nil {
		t.Fatalf("SendRich: %v", err)
	}
	wire := decodeThinkingFields(t, captured, "opus-4-8 SendRich").MaxTokens
	if rr.CompletionBudget != 24576 || wire != rr.CompletionBudget {
		t.Errorf("CompletionBudget = %d, wire max_tokens = %d; want both 24576 (8192 desired + 16384 high headroom)", rr.CompletionBudget, wire)
	}
	if rr.FinishReason != "end_turn" || rr.StopReason != "end_turn" {
		t.Errorf("FinishReason = %q, StopReason = %q; want both %q (the server's own word is already the normalized one)", rr.FinishReason, rr.StopReason, "end_turn")
	}
	if rr.OutputSplit.Known {
		t.Errorf("OutputSplit = %+v, want unknown: the usage block attributes no channel", rr.OutputSplit)
	}
}

// TestSendRichStreaming_AccountsForTheRequest drives the streaming rich path
// against the stub stream and checks the same account.
func TestSendRichStreaming_AccountsForTheRequest(t *testing.T) {
	var captured []byte
	server := stubStreamingServer(t, &captured)
	defer server.Close()

	conv := NewConversation("sys")
	conv.ApiToken = "test-token"
	conv.SetEndpoint(server.URL)
	conv.Settings.Model = "claude-opus-4-8"
	conv.Settings.MaxTokens = 8192

	rr, err := conv.SendRichStreaming([]llmapi.ContentBlock{llmapi.NewTextBlock("hello")},
		llmapi.Sampling{ReasoningEffort: llmapi.ReasoningHigh, DesiredOutputTokens: 4096}, nil)
	if err != nil {
		t.Fatalf("SendRichStreaming: %v", err)
	}
	wire := decodeThinkingFields(t, captured, "opus-4-8 SendRichStreaming").MaxTokens
	if rr.CompletionBudget != 20480 || wire != rr.CompletionBudget {
		t.Errorf("CompletionBudget = %d, wire max_tokens = %d; want both 20480 (4096 desired + 16384 high headroom)", rr.CompletionBudget, wire)
	}
	if rr.FinishReason != "end_turn" || rr.StopReason != "end_turn" {
		t.Errorf("FinishReason = %q, StopReason = %q; want both %q", rr.FinishReason, rr.StopReason, "end_turn")
	}
	if rr.OutputSplit.Known {
		t.Errorf("OutputSplit = %+v, want unknown: the usage block attributes no channel", rr.OutputSplit)
	}
}
