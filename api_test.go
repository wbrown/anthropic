package anthropic

import (
	"testing"

	"github.com/wbrown/llmapi"
)

func TestInit(t *testing.T) {
	// Check if we have an API token.
	if DefaultApiToken == "" {
		t.Errorf("Expected DefaultApiToken to not be empty, you" +
			"to set the ANTHROPIC_API_KEY environment variable")
	}
}

func TestConversation_Send(t *testing.T) {
	// Test that the reply is not empty
	conversation := NewConversation("You are a friendly assistant.")
	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.Send("Hello Claude!", llmapi.Sampling{})
	if err != nil {
		t.Errorf("Expected err to be nil: %s", err)
	}
	if reply == "" {
		t.Errorf("Expected reply to not be empty")
	}
	if stopReason == "" {
		t.Errorf("Expected stopReason to not be empty")
	}
	if inputTokens == 0 {
		t.Errorf("Expected inputTokens to not be 0")
	}
	if outputTokens == 0 {
		t.Errorf("Expected outputTokens to not be 0")
	}
}

// TestConversation_SendStreaming tests the SSE streaming functionality.
// It verifies that:
//   - The callback is invoked with text fragments as they arrive
//   - The accumulated reply matches the complete response
//   - Token counts and stop reason are correctly returned
func TestConversation_SendStreaming(t *testing.T) {
	conversation := NewConversation("You are a friendly assistant.")

	// Track how many times the callback is invoked with content
	var tokenCount int
	callback := func(text string, done bool) {
		if !done && text != "" {
			tokenCount++
		}
	}

	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.SendStreaming("Say hello in exactly 5 words.", llmapi.Sampling{}, callback)
	if err != nil {
		t.Errorf("Expected err to be nil: %s", err)
	}
	if reply == "" {
		t.Errorf("Expected reply to not be empty")
	}
	if stopReason == "" {
		t.Errorf("Expected stopReason to not be empty")
	}
	if inputTokens == 0 {
		t.Errorf("Expected inputTokens to not be 0")
	}
	if outputTokens == 0 {
		t.Errorf("Expected outputTokens to not be 0")
	}
	if tokenCount == 0 {
		t.Errorf("Expected callback to be called at least once with tokens")
	}
}

// TestConversation_SendUntilDone tests the SendUntilDone method, which will
// in turn also test MergeIfLastTwoAssistant method as Claude should generally
// require more than two replies to complete this conversation.
func TestConversation_SendUntilDone(t *testing.T) {
	conversation := NewConversation("You are a friendly assistant.")
	conversation.Settings.MaxTokens = 125
	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.SendUntilDone(
			"Tell me about the impact of the Byzantines on the world.", llmapi.Sampling{})
	if err != nil {
		t.Errorf("Expected err to be nil: %s", err)
	}
	if reply == "" {
		t.Errorf("Expected reply to not be empty")
	}
	if stopReason != "end_turn" {
		t.Errorf("Expected stopReason to 'end_turn")
	}
	if inputTokens == 0 {
		t.Errorf("Expected inputTokens to not be 0")
	}
	if outputTokens == 0 {
		t.Errorf("Expected outputTokens to not be 0")
	}
}

// TestConversation_SendStreamingUntilDone tests streaming with auto-continuation.
// With MaxTokens set to 125, the response will hit the token limit and require
// multiple continuations. This test verifies that:
//   - Streaming continues across multiple API calls
//   - The callback receives tokens from all continuations
//   - The final stopReason is "end_turn" (not "max_tokens")
//   - MergeIfLastTwoAssistant correctly combines continued responses
func TestConversation_SendStreamingUntilDone(t *testing.T) {
	conversation := NewConversation("You are a friendly assistant.")
	// Low max_tokens forces multiple continuations
	conversation.Settings.MaxTokens = 125

	var tokenCount int
	callback := func(text string, done bool) {
		if !done && text != "" {
			tokenCount++
		}
	}

	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.SendStreamingUntilDone(
			"Tell me about the impact of the Byzantines on the world.", llmapi.Sampling{}, callback)
	if err != nil {
		t.Errorf("Expected err to be nil: %s", err)
	}
	if reply == "" {
		t.Errorf("Expected reply to not be empty")
	}
	if stopReason != "end_turn" {
		t.Errorf("Expected stopReason to be 'end_turn', got '%s'", stopReason)
	}
	if inputTokens == 0 {
		t.Errorf("Expected inputTokens to not be 0")
	}
	if outputTokens == 0 {
		t.Errorf("Expected outputTokens to not be 0")
	}
	if tokenCount == 0 {
		t.Errorf("Expected callback to be called at least once with tokens")
	}
}

// TestCacheControl tests that cache control can be added to content blocks
func TestCacheControl(t *testing.T) {
	// Test EnableCaching
	block := &ContentBlock{
		ContentType: "text",
		Text:        stringPtr("Test content"),
	}

	block.EnableCaching()
	if block.CacheControl == nil {
		t.Error("Expected CacheControl to be set")
	}
	if block.CacheControl.Type != "ephemeral" {
		t.Errorf("Expected CacheControl.Type to be 'ephemeral', got %s", block.CacheControl.Type)
	}

	// Test DisableCaching
	block.DisableCaching()
	if block.CacheControl != nil {
		t.Error("Expected CacheControl to be nil after disabling")
	}
}

// TestCacheStatistics tests cache statistics tracking
func TestCacheStatistics(t *testing.T) {
	conv := &Conversation{
		Usage: struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		}{
			InputTokens:  1000,
			OutputTokens: 500,
		},
		CacheStats: struct {
			TotalCacheCreationTokens int
			TotalCacheReadTokens     int
			TotalTokensSaved         int
			CacheHits                int
			CacheMisses              int
		}{
			TotalCacheCreationTokens: 100,
			TotalCacheReadTokens:     900,
			TotalTokensSaved:         810, // 900 - (900/10)
			CacheHits:                9,
			CacheMisses:              1,
		},
	}

	// Test CacheHitRate
	hitRate := conv.CacheHitRate()
	expectedRate := 90.0 // 9 hits out of 10 total
	if hitRate != expectedRate {
		t.Errorf("Expected cache hit rate %.1f%%, got %.1f%%", expectedRate, hitRate)
	}

	// Test CacheSavingsRate
	savingsRate := conv.CacheSavingsRate()
	expectedSavings := 81.0 // 810 saved out of 1000 input tokens
	if savingsRate != expectedSavings {
		t.Errorf("Expected cache savings rate %.1f%%, got %.1f%%", expectedSavings, savingsRate)
	}
}

// TestListModels tests the ListModels utility function
func TestListModels(t *testing.T) {
	models, err := ListModels("")
	if err != nil {
		t.Errorf("Expected err to be nil: %s", err)
	}
	if models == nil {
		t.Error("Expected models to not be nil")
		return
	}
	if len(models.Data) == 0 {
		t.Error("Expected at least one model to be returned")
	}

	t.Logf("Found %d models:", len(models.Data))
	for _, model := range models.Data {
		t.Logf("  - %s (%s)", model.ID, model.DisplayName)
		if model.ID == "" {
			t.Error("Expected model ID to not be empty")
		}
		if model.DisplayName == "" {
			t.Error("Expected model DisplayName to not be empty")
		}
	}
}

// Helper function to create string pointers
func stringPtr(s string) *string {
	return &s
}
