package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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
	models, err := ListModels(nil, "")
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

// TestSendRich tests the SendRich method with text content blocks.
// Verifies that:
//   - Rich content can be sent and receives a response
//   - Response contains proper content blocks
//   - Token counts and stop reason are returned
func TestSendRich(t *testing.T) {
	conversation := NewConversation("You are a friendly assistant.")

	content := []llmapi.ContentBlock{
		llmapi.NewTextBlock("Hello Claude! Say hi back in 5 words or less."),
	}

	response, err := conversation.SendRich(content, llmapi.Sampling{})
	if err != nil {
		t.Fatalf("Expected err to be nil: %s", err)
	}
	if response == nil {
		t.Fatal("Expected response to not be nil")
	}
	if len(response.Content) == 0 {
		t.Error("Expected response to have content blocks")
	}
	if response.Text() == "" {
		t.Error("Expected response text to not be empty")
	}
	if response.StopReason == "" {
		t.Error("Expected stopReason to not be empty")
	}
	if response.InputTokens == 0 {
		t.Error("Expected inputTokens to not be 0")
	}
	if response.OutputTokens == 0 {
		t.Error("Expected outputTokens to not be 0")
	}

	// Verify the message was added to history
	messages := conversation.GetRichMessages()
	if len(messages) < 2 {
		t.Errorf("Expected at least 2 messages in history, got %d", len(messages))
	}
}

// TestAddRichMessage_GetRichMessages tests adding and retrieving rich messages.
func TestAddRichMessage_GetRichMessages(t *testing.T) {
	conversation := NewConversation("You are helpful.")

	// Add a user message with multiple content blocks
	userContent := []llmapi.ContentBlock{
		llmapi.NewTextBlock("First part."),
		llmapi.NewTextBlock("Second part."),
	}
	conversation.AddRichMessage(llmapi.RoleUser, userContent)

	// Add an assistant message
	assistantContent := []llmapi.ContentBlock{
		llmapi.NewTextBlock("Here is my response."),
	}
	conversation.AddRichMessage(llmapi.RoleAssistant, assistantContent)

	// Retrieve and verify
	messages := conversation.GetRichMessages()
	if len(messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(messages))
	}

	// Verify user message
	if messages[0].Role != "user" {
		t.Errorf("Expected first message role to be 'user', got '%s'", messages[0].Role)
	}
	if len(messages[0].Content) != 2 {
		t.Errorf("Expected user message to have 2 content blocks, got %d", len(messages[0].Content))
	}

	// Verify assistant message
	if messages[1].Role != "assistant" {
		t.Errorf("Expected second message role to be 'assistant', got '%s'", messages[1].Role)
	}
	if len(messages[1].Content) != 1 {
		t.Errorf("Expected assistant message to have 1 content block, got %d", len(messages[1].Content))
	}
}

// TestSetTools_GetTools tests tool configuration.
func TestSetTools_GetTools(t *testing.T) {
	conversation := NewConversation("You are helpful.")

	// Initially no tools
	tools := conversation.GetTools()
	if len(tools) != 0 {
		t.Errorf("Expected no tools initially, got %d", len(tools))
	}

	// Set some tools
	newTools := []llmapi.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			InputSchema: []byte(`{"type": "object", "properties": {"location": {"type": "string"}}}`),
		},
		{
			Name:        "get_time",
			Description: "Get the current time",
			InputSchema: []byte(`{"type": "object", "properties": {}}`),
		},
	}
	conversation.SetTools(newTools)

	// Verify tools were set
	tools = conversation.GetTools()
	if len(tools) != 2 {
		t.Fatalf("Expected 2 tools, got %d", len(tools))
	}
	if tools[0].Name != "get_weather" {
		t.Errorf("Expected first tool name to be 'get_weather', got '%s'", tools[0].Name)
	}
	if tools[1].Name != "get_time" {
		t.Errorf("Expected second tool name to be 'get_time', got '%s'", tools[1].Name)
	}

	// Clear tools
	conversation.SetTools(nil)
	tools = conversation.GetTools()
	if len(tools) != 0 {
		t.Errorf("Expected no tools after clearing, got %d", len(tools))
	}
}

// TestGetCapabilities tests the capability reporting.
func TestGetCapabilities(t *testing.T) {
	conversation := NewConversation("You are helpful.")

	caps := conversation.GetCapabilities()

	// Anthropic should support all features
	if !caps.SupportsImages {
		t.Error("Expected SupportsImages to be true")
	}
	if !caps.SupportsDocuments {
		t.Error("Expected SupportsDocuments to be true")
	}
	if !caps.SupportsToolUse {
		t.Error("Expected SupportsToolUse to be true")
	}
	if !caps.SupportsThinking {
		t.Error("Expected SupportsThinking to be true")
	}
	if !caps.SupportsStreaming {
		t.Error("Expected SupportsStreaming to be true")
	}
}

// TestContentBlockConversion tests the conversion between llmapi and anthropic content blocks.
func TestContentBlockConversion(t *testing.T) {
	// Test text block conversion
	t.Run("TextBlock", func(t *testing.T) {
		llmapiBlock := llmapi.NewTextBlock("Hello world")
		anthropicBlocks := toAnthropicContentBlocks([]llmapi.ContentBlock{llmapiBlock})

		if len(anthropicBlocks) != 1 {
			t.Fatalf("Expected 1 block, got %d", len(anthropicBlocks))
		}
		if anthropicBlocks[0].ContentType != "text" {
			t.Errorf("Expected type 'text', got '%s'", anthropicBlocks[0].ContentType)
		}
		if anthropicBlocks[0].Text == nil || *anthropicBlocks[0].Text != "Hello world" {
			t.Error("Expected text to be 'Hello world'")
		}

		// Convert back
		llmapiBlocks := fromAnthropicContentBlocks(anthropicBlocks)
		if len(llmapiBlocks) != 1 {
			t.Fatalf("Expected 1 block back, got %d", len(llmapiBlocks))
		}
		if llmapiBlocks[0].Type != llmapi.ContentTypeText {
			t.Errorf("Expected type text, got %s", llmapiBlocks[0].Type)
		}
		if llmapiBlocks[0].Text != "Hello world" {
			t.Errorf("Expected text 'Hello world', got '%s'", llmapiBlocks[0].Text)
		}
	})

	// Test tool result block conversion
	t.Run("ToolResultBlock", func(t *testing.T) {
		llmapiBlock := llmapi.NewToolResultBlock("tool_123", "Result data", false)
		anthropicBlocks := toAnthropicContentBlocks([]llmapi.ContentBlock{llmapiBlock})

		if len(anthropicBlocks) != 1 {
			t.Fatalf("Expected 1 block, got %d", len(anthropicBlocks))
		}
		if anthropicBlocks[0].ContentType != "tool_result" {
			t.Errorf("Expected type 'tool_result', got '%s'", anthropicBlocks[0].ContentType)
		}
		if anthropicBlocks[0].ToolUseID == nil || *anthropicBlocks[0].ToolUseID != "tool_123" {
			t.Error("Expected tool_use_id to be 'tool_123'")
		}
	})

	// Test thinking block conversion (from anthropic to llmapi)
	t.Run("ThinkingBlock", func(t *testing.T) {
		thinking := "Let me think about this..."
		anthropicBlocks := []ContentBlock{
			{
				ContentType: "thinking",
				Thinking:    &thinking,
			},
		}

		llmapiBlocks := fromAnthropicContentBlocks(anthropicBlocks)
		if len(llmapiBlocks) != 1 {
			t.Fatalf("Expected 1 block, got %d", len(llmapiBlocks))
		}
		if llmapiBlocks[0].Type != llmapi.ContentTypeThinking {
			t.Errorf("Expected type thinking, got %s", llmapiBlocks[0].Type)
		}
		if llmapiBlocks[0].Thinking == nil {
			t.Fatal("Expected Thinking to not be nil")
		}
		if llmapiBlocks[0].Thinking.Thinking != thinking {
			t.Errorf("Expected thinking text '%s', got '%s'", thinking, llmapiBlocks[0].Thinking.Thinking)
		}
	})
}

// TestRichResponseHelpers tests the helper methods on RichResponse.
func TestRichResponseHelpers(t *testing.T) {
	response := llmapi.RichResponse{
		Content: []llmapi.ContentBlock{
			{Type: llmapi.ContentTypeThinking, Thinking: &llmapi.ThinkingContent{Thinking: "My reasoning"}},
			{Type: llmapi.ContentTypeText, Text: "Hello "},
			{Type: llmapi.ContentTypeText, Text: "world!"},
			{Type: llmapi.ContentTypeToolUse, ToolUse: &llmapi.ToolUseContent{ID: "tool_1", Name: "test_tool"}},
		},
		StopReason:   "end_turn",
		InputTokens:  100,
		OutputTokens: 50,
	}

	// Test Text()
	text := response.Text()
	if text != "Hello world!" {
		t.Errorf("Expected text 'Hello world!', got '%s'", text)
	}

	// Test ThinkingText()
	thinking := response.ThinkingText()
	if thinking != "My reasoning" {
		t.Errorf("Expected thinking 'My reasoning', got '%s'", thinking)
	}

	// Test ToolUses()
	toolUses := response.ToolUses()
	if len(toolUses) != 1 {
		t.Fatalf("Expected 1 tool use, got %d", len(toolUses))
	}
	if toolUses[0].Name != "test_tool" {
		t.Errorf("Expected tool name 'test_tool', got '%s'", toolUses[0].Name)
	}

	// Test HasToolUse()
	if !response.HasToolUse() {
		t.Error("Expected HasToolUse() to return true")
	}

	// Test with no tool use
	noToolResponse := llmapi.RichResponse{
		Content: []llmapi.ContentBlock{
			{Type: llmapi.ContentTypeText, Text: "Just text"},
		},
	}
	if noToolResponse.HasToolUse() {
		t.Error("Expected HasToolUse() to return false for text-only response")
	}
}

// TestParseSSEStreamRich_TextOnly tests parsing SSE stream with text content only.
func TestParseSSEStreamRich_TextOnly(t *testing.T) {
	// Mock SSE stream with text deltas
	sseData := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":25}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}
`

	conv := NewConversation("Test")
	var callbackText string
	var callbackDone bool
	callback := func(text string, done bool) {
		callbackText += text
		callbackDone = done
	}

	reader := strings.NewReader(sseData)
	fullText, stopReason, inputTokens, outputTokens, err := conv.parseSSEStreamRich(reader, callback)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if fullText != "Hello world!" {
		t.Errorf("Expected fullText 'Hello world!', got '%s'", fullText)
	}
	if stopReason != "end_turn" {
		t.Errorf("Expected stopReason 'end_turn', got '%s'", stopReason)
	}
	if inputTokens != 25 {
		t.Errorf("Expected inputTokens 25, got %d", inputTokens)
	}
	if outputTokens != 10 {
		t.Errorf("Expected outputTokens 10, got %d", outputTokens)
	}
	if callbackText != "Hello world!" {
		t.Errorf("Expected callback to receive 'Hello world!', got '%s'", callbackText)
	}
	if !callbackDone {
		t.Error("Expected callback to be called with done=true")
	}
}

// TestParseSSEStreamRich_WithThinking tests parsing SSE stream with thinking blocks.
func TestParseSSEStreamRich_WithThinking(t *testing.T) {
	// Mock SSE stream with thinking and text
	sseData := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":50}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" Done thinking."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30}}

event: message_stop
data: {"type":"message_stop"}
`

	conv := NewConversation("Test")
	reader := strings.NewReader(sseData)
	fullText, stopReason, inputTokens, outputTokens, err := conv.parseSSEStreamRich(reader, nil)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Log the actual output for verification
	t.Logf("Full text output:\n%s", fullText)

	// Should contain thinking tags
	expectedThinking := "<thinking>\nLet me think... Done thinking.\n</thinking>\n"
	if !strings.Contains(fullText, expectedThinking) {
		t.Errorf("Expected fullText to contain thinking block, got '%s'", fullText)
	}

	// Should contain text
	if !strings.Contains(fullText, "The answer is 42.") {
		t.Errorf("Expected fullText to contain 'The answer is 42.', got '%s'", fullText)
	}

	if stopReason != "end_turn" {
		t.Errorf("Expected stopReason 'end_turn', got '%s'", stopReason)
	}
	if inputTokens != 50 {
		t.Errorf("Expected inputTokens 50, got %d", inputTokens)
	}
	if outputTokens != 30 {
		t.Errorf("Expected outputTokens 30, got %d", outputTokens)
	}

	// Should set HasThinkingContent flag
	if !conv.HasThinkingContent {
		t.Error("Expected HasThinkingContent to be true")
	}
}

// TestParseSSEStreamRich_MaxTokens tests parsing SSE stream that stops due to max_tokens.
func TestParseSSEStreamRich_MaxTokens(t *testing.T) {
	sseData := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":100}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial response..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":50}}

event: message_stop
data: {"type":"message_stop"}
`

	conv := NewConversation("Test")
	reader := strings.NewReader(sseData)
	fullText, stopReason, _, _, err := conv.parseSSEStreamRich(reader, nil)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if fullText != "Partial response..." {
		t.Errorf("Expected 'Partial response...', got '%s'", fullText)
	}
	if stopReason != "max_tokens" {
		t.Errorf("Expected stopReason 'max_tokens', got '%s'", stopReason)
	}
}

// TestContextCancellation tests that requests respect context cancellation.
// It creates a mock server that delays response, then cancels the context
// before the response arrives.
func TestContextCancellation(t *testing.T) {
	// Create a server that delays its response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Delay longer than our context timeout
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"type":"message","content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn"}`))
	}))
	defer server.Close()

	// Save original URI and restore after test
	originalURI := messagesURI
	messagesURI = server.URL
	defer func() { messagesURI = originalURI }()

	// Create a context that will be cancelled quickly
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	// Should get a context deadline exceeded error
	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	// Check that the error is related to context cancellation
	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
}

// TestContextCancellationImmediate tests immediate context cancellation.
func TestContextCancellationImmediate(t *testing.T) {
	// Create a server (won't actually be reached)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Save original URI and restore after test
	originalURI := messagesURI
	messagesURI = server.URL
	defer func() { messagesURI = originalURI }()

	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	// Should get a context cancelled error
	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}
}

// TestContextCancellationStreaming tests that streaming requests respect context cancellation.
func TestContextCancellationStreaming(t *testing.T) {
	// Create a server that delays its response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Save original URI and restore after test
	originalURI := messagesURI
	messagesURI = server.URL
	defer func() { messagesURI = originalURI }()

	// Create a context that will be cancelled quickly
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"

	_, _, _, _, err := conv.SendStreaming("Hello", llmapi.Sampling{}, nil)

	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
}

// =============================================================================
// Real-world context cancellation tests (no mock server)
// =============================================================================

// TestContextCancellationPreCancelled tests that an already-cancelled context
// results in a context cancellation error.
func TestContextCancellationPreCancelled(t *testing.T) {
	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	conv := NewConversation("Test")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token" // Doesn't need to be valid

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to pre-cancelled context")
	}

	// The error should indicate context cancellation
	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}
}

// TestContextCancellationTinyTimeout tests cancellation with a very short timeout
// against the real API endpoint. This tests the actual HTTP client behavior.
func TestContextCancellationTinyTimeout(t *testing.T) {
	// 1ms timeout - will fail during TCP/TLS handshake
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token" // Doesn't need to be valid

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to timeout")
	}

	// The error should indicate context deadline exceeded
	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context error, got: %v", err)
	}
}

// TestContextCancellationMidStream tests cancelling a real streaming request
// mid-generation. This is an integration test that requires valid API credentials.
func TestContextCancellationMidStream(t *testing.T) {
	if DefaultApiToken == "" {
		t.Skip("Skipping integration test: ANTHROPIC_API_KEY not set")
	}

	ctx, cancel := context.WithCancel(context.Background())

	conv := NewConversation("You are a helpful assistant.")
	conv.SetContext(ctx)

	var tokensReceived int
	var cancelled bool

	// Cancel after receiving some tokens
	callback := func(text string, done bool) {
		if done {
			return
		}
		tokensReceived++
		// Cancel after receiving a few tokens
		if tokensReceived >= 5 && !cancelled {
			cancelled = true
			cancel()
		}
	}

	// Ask for a long response
	_, _, _, _, err := conv.SendStreaming(
		"Write a detailed 500 word essay about the history of computing.",
		llmapi.Sampling{},
		callback,
	)

	// Should get an error due to cancellation
	if err == nil {
		t.Log("Request completed without error (generation finished before cancellation)")
		return
	}

	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}

	t.Logf("Successfully cancelled after receiving %d tokens", tokensReceived)
}
