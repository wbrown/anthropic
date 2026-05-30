package anthropic

import (
	"context"
	"strings"
	"testing"

	"github.com/wbrown/llmapi"
)

// TestSendStreaming_EmptyText_SendsPendingUserTurn pins the fix: an empty-text SendStreaming
// whose conversation ends in a plain user turn (no tool_result) must SEND it as-is — the
// pending request — not reject it with "cannot continue conversation". This is the retry
// path: a failed send leaves the user turn queued, and the retry re-sends it via empty text.
//
// The guard runs before any network I/O, so a pre-cancelled context keeps this offline and
// deterministic: past the guard the request fails fast at the transport, and the only thing
// asserted is that the failure is NOT the guard's "cannot continue conversation".
//
// Before the fix this returns "cannot continue conversation" (api.go ~846) and the test fails.
func TestSendStreaming_EmptyText_SendsPendingUserTurn(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "where does this take place?") // pending plain-user turn

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	_, _, _, _, _, _, err := conv.SendStreaming("", llmapi.Sampling{}, nil)
	if err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	}
	if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("empty-text send of a pending user turn was rejected by the guard: %v", err)
	}
}

// TestSendStreaming_EmptyText_RejectsEmptyTrailingTurn pins that the guard is PRESERVED for a
// trailing user turn with no content — there is nothing the model can answer, so
// "cannot continue conversation" is still returned. The fix must not flatten this away.
func TestSendStreaming_EmptyText_RejectsEmptyTrailingTurn(t *testing.T) {
	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddRichMessage(llmapi.RoleUser, nil) // user turn with no content blocks

	_, _, _, _, _, _, err := conv.SendStreaming("", llmapi.Sampling{}, nil)
	if err == nil || !strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("expected 'cannot continue conversation' for an empty trailing user turn, got: %v", err)
	}
}

// TestSendStreaming_EmptyText_AllowsToolResultTurn pins that the tool_result continuation path
// is PRESERVED: a trailing user turn carrying a tool_result is sent as-is (it can only be
// delivered via AddRichMessage + empty-text send), not rejected. narrative-generators doesn't
// exercise tools, so this guards the library contract directly.
func TestSendStreaming_EmptyText_AllowsToolResultTurn(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddRichMessage(llmapi.RoleUser, []llmapi.ContentBlock{
		llmapi.NewToolResultBlock("tool-123", "result data", false),
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	_, _, _, _, _, _, err := conv.SendStreaming("", llmapi.Sampling{}, nil)
	if err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	}
	if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("tool_result continuation was rejected by the guard: %v", err)
	}
}

// The non-streaming Send path (sendInternal) has the same empty-text guard, with a > 2 message
// threshold, so these mirror the SendStreaming cases on a multi-turn history where the guard
// actually applies.

func TestSendInternal_EmptyText_SendsPendingUserTurn(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "q1")
	conv.AddMessage(llmapi.RoleAssistant, "a1")
	conv.AddMessage(llmapi.RoleUser, "q2") // trailing plain-user turn, len 3 > 2

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	if _, err := conv.sendInternal("", llmapi.Sampling{}); err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	} else if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("empty-text send of a pending user turn was rejected by the guard: %v", err)
	}
}

func TestSendInternal_EmptyText_RejectsEmptyTrailingTurn(t *testing.T) {
	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "q1")
	conv.AddMessage(llmapi.RoleAssistant, "a1")
	conv.AddRichMessage(llmapi.RoleUser, nil) // trailing user turn with no content, len 3 > 2

	_, err := conv.sendInternal("", llmapi.Sampling{})
	if err == nil || !strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("expected 'cannot continue conversation' for an empty trailing user turn, got: %v", err)
	}
}

func TestSendInternal_EmptyText_AllowsToolResultTurn(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "q1")
	conv.AddMessage(llmapi.RoleAssistant, "a1")
	conv.AddRichMessage(llmapi.RoleUser, []llmapi.ContentBlock{
		llmapi.NewToolResultBlock("tool-123", "result data", false),
	}) // trailing tool_result turn, len 3 > 2

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	if _, err := conv.sendInternal("", llmapi.Sampling{}); err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	} else if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("tool_result continuation was rejected by the guard: %v", err)
	}
}

// appendAssistantToolUse adds an assistant turn that contains a tool_use block — an open tool
// call awaiting the client's tool_result. Built directly (white-box) because the guard keys on
// the block's ContentType, which is what a real tool_use turn carries.
func appendAssistantToolUse(conv *Conversation) {
	*conv.Messages = append(*conv.Messages, &Message{
		Role:    "assistant",
		Content: &[]ContentBlock{{ContentType: "tool_use"}},
	})
}

// TestSendStreaming_EmptyText_RejectsOpenToolUseRequest pins the request-side fix: an empty-text
// send when the trailing turn is an assistant tool_use REQUEST must be rejected — the model is
// awaiting the client's tool_result and cannot produce output. (You add the tool_result, a user
// turn, and *then* send.)
func TestSendStreaming_EmptyText_RejectsOpenToolUseRequest(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "weather?")
	appendAssistantToolUse(conv) // open tool_use request, no tool_result yet

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	_, _, _, _, _, _, err := conv.SendStreaming("", llmapi.Sampling{}, nil)
	if err == nil || !strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("expected 'cannot continue conversation' for empty text after an open tool_use request, got: %v", err)
	}
}

// TestSendStreaming_EmptyText_ContinuesAfterAssistantResponse pins that the request-side check
// does NOT break the legitimate continuation: empty text after a finished/cut-off assistant turn
// (no tool_use) must still proceed (prefill/continue), not be rejected.
func TestSendStreaming_EmptyText_ContinuesAfterAssistantResponse(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "hi")
	conv.AddMessage(llmapi.RoleAssistant, "hello") // a normal assistant turn

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	if _, _, _, _, _, _, err := conv.SendStreaming("", llmapi.Sampling{}, nil); err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	} else if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("empty text after a normal assistant turn must continue, not be rejected: %v", err)
	}
}

// TestSendInternal_EmptyText_RejectsOpenToolUseRequest mirrors the request-side fix for Send, and
// pins that it fires at len 2 — below the > 2 user-side gate — because an open tool_use request
// is invalid regardless of conversation length.
func TestSendInternal_EmptyText_RejectsOpenToolUseRequest(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "weather?")
	appendAssistantToolUse(conv) // len 2: open tool_use request

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	if _, err := conv.sendInternal("", llmapi.Sampling{}); err == nil || !strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("expected 'cannot continue conversation' for empty text after an open tool_use request, got: %v", err)
	}
}

// TestSendInternal_EmptyText_ContinuesAfterAssistantResponse pins that a normal trailing assistant
// turn still continues under Send (no tool_use → proceed).
func TestSendInternal_EmptyText_ContinuesAfterAssistantResponse(t *testing.T) {
	orig := retryDelay
	retryDelay = 0
	defer func() { retryDelay = orig }()

	conv := NewConversation("system")
	conv.ApiToken = "dummy"
	conv.AddMessage(llmapi.RoleUser, "hi")
	conv.AddMessage(llmapi.RoleAssistant, "hello") // len 2, trailing assistant turn

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	conv.SetContext(ctx)

	if _, err := conv.sendInternal("", llmapi.Sampling{}); err == nil {
		t.Fatal("expected a transport error from the cancelled request")
	} else if strings.Contains(err.Error(), "cannot continue conversation") {
		t.Fatalf("empty text after a normal assistant turn must continue, not be rejected: %v", err)
	}
}
