package anthropic

import (
	"testing"
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
		conversation.Send("Hello Claude!")
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

// TestConversation_SendUntilDone tests the SendUntilDone method, which will
// in turn also test MergeIfLastTwoAssistant method as Claude should generally
// require more than two replies to complete this conversation.
func TestConversation_SendUntilDone(t *testing.T) {
	conversation := NewConversation("You are a friendly assistant.")
	conversation.Settings.MaxTokens = 125
	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.SendUntilDone(
			"Tell me about the impact of the Byzantines on the world.")
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
