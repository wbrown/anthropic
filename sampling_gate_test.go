package anthropic

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/wbrown/llmapi"
)

// These tests verify the wire-level outcome of the sampling deprecation gate:
// for each of the three send paths (sendInternal via Send, SendStreaming, and
// SendRichStreaming), the JSON body sent to Anthropic must omit temperature,
// top_p, and top_k for Opus 4.7+ models, and include temperature for older
// models. The presence-based 400 from the real API can't be exercised in
// tests, so we capture the body that would have been sent and inspect it.

// stubMessagesServer returns an httptest.Server that captures the request body
// into *captured and responds with a minimal valid non-streaming message.
func stubMessagesServer(t *testing.T, captured *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("server: read body: %v", err)
		}
		*captured = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "test",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 1, "output_tokens": 1}
		}`))
	}))
}

// stubStreamingServer returns an httptest.Server that captures the request
// body and responds with a minimal valid SSE stream.
func stubStreamingServer(t *testing.T, captured *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("server: read body: %v", err)
		}
		*captured = body
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("server: response writer does not support flushing")
		}
		sse := `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}
`
		_, _ = w.Write([]byte(sse))
		flusher.Flush()
	}))
}

// assertSamplingFieldsAbsent inspects the captured request body and fails the
// test if any of temperature/top_p/top_k appear as JSON keys at the top level.
func assertSamplingFieldsAbsent(t *testing.T, body []byte, context string) {
	t.Helper()
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("%s: unmarshal request body: %v\nbody=%s", context, err, string(body))
	}
	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if _, present := decoded[key]; present {
			t.Errorf("%s: request body must not contain %q (Opus 4.7+ rejects it); body=%s",
				context, key, string(body))
		}
	}
}

// assertTemperaturePresent asserts the request body included a temperature
// field with the expected value.
func assertTemperaturePresent(t *testing.T, body []byte, want float64, context string) {
	t.Helper()
	var decoded struct {
		Temperature *float64 `json:"temperature"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("%s: unmarshal request body: %v\nbody=%s", context, err, string(body))
	}
	if decoded.Temperature == nil {
		t.Fatalf("%s: temperature must be present on supported model; body=%s", context, string(body))
	}
	if *decoded.Temperature != want {
		t.Errorf("%s: temperature = %v, want %v", context, *decoded.Temperature, want)
	}
}

// TestSendInternal_GatesSamplingByModel exercises the sendInternal path
// (reached via Conversation.Send) for both deprecated and supported models.
func TestSendInternal_GatesSamplingByModel(t *testing.T) {
	t.Run("opus-4-8 omits all three", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.Temperature = 0.7
		conv.Settings.TopP = 0.9
		conv.Settings.TopK = 40

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertSamplingFieldsAbsent(t, captured, "opus-4-8 Send")
	})

	t.Run("opus-4-7 omits all three", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-7"
		conv.Settings.Temperature = 0.7

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertSamplingFieldsAbsent(t, captured, "opus-4-7 Send")
	})

	t.Run("sonnet-4-6 includes temperature", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-6"
		conv.Settings.Temperature = 0.4

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertTemperaturePresent(t, captured, 0.4, "sonnet-4-6 Send")
	})

	// The whole reason Temperature is *float64: an explicit 0 on a supported
	// model must still be sent (greedy decoding), not silently dropped by an
	// over-eager omitempty.
	t.Run("sonnet-4-6 sends explicit zero temperature", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-6"
		conv.Settings.Temperature = 0.0

		if _, _, _, _, _, _, err := conv.Send("hello", llmapi.Sampling{}); err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertTemperaturePresent(t, captured, 0.0, "sonnet-4-6 Send T=0")
	})

	// Override on a 4.7+ model must also be dropped (silently, per design).
	t.Run("opus-4-8 ignores per-call Sampling override", func(t *testing.T) {
		var captured []byte
		server := stubMessagesServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"

		_, _, _, _, _, _, err := conv.Send("hello",
			llmapi.Sampling{Temperature: 0.9, TopP: 0.95, TopK: 100})
		if err != nil {
			t.Fatalf("Send: %v", err)
		}
		assertSamplingFieldsAbsent(t, captured, "opus-4-8 Send with override")
	})
}

// TestSendStreaming_GatesSamplingByModel exercises the SendStreaming path.
func TestSendStreaming_GatesSamplingByModel(t *testing.T) {
	t.Run("opus-4-8 omits all three", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.Temperature = 0.7
		conv.Settings.TopP = 0.9
		conv.Settings.TopK = 40

		if _, _, _, _, _, _, err := conv.SendStreaming("hello", llmapi.Sampling{}, nil); err != nil {
			t.Fatalf("SendStreaming: %v", err)
		}
		assertSamplingFieldsAbsent(t, captured, "opus-4-8 SendStreaming")
		// Stream flag must still be set on streaming requests.
		if !strings.Contains(string(captured), `"stream":true`) {
			t.Errorf("expected stream=true in body; body=%s", string(captured))
		}
	})

	t.Run("sonnet-4-6 includes temperature", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-6"
		conv.Settings.Temperature = 0.4

		if _, _, _, _, _, _, err := conv.SendStreaming("hello", llmapi.Sampling{}, nil); err != nil {
			t.Fatalf("SendStreaming: %v", err)
		}
		assertTemperaturePresent(t, captured, 0.4, "sonnet-4-6 SendStreaming")
	})
}

// TestSendRichStreaming_GatesSamplingByModel exercises the SendRichStreaming
// path, which has its own local copy of the sampling-resolution block.
func TestSendRichStreaming_GatesSamplingByModel(t *testing.T) {
	t.Run("opus-4-8 omits all three", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-opus-4-8"
		conv.Settings.Temperature = 0.7
		conv.Settings.TopP = 0.9
		conv.Settings.TopK = 40

		content := []llmapi.ContentBlock{llmapi.NewTextBlock("hello")}
		if _, err := conv.SendRichStreaming(content, llmapi.Sampling{}, nil); err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		assertSamplingFieldsAbsent(t, captured, "opus-4-8 SendRichStreaming")
	})

	t.Run("sonnet-4-6 includes temperature", func(t *testing.T) {
		var captured []byte
		server := stubStreamingServer(t, &captured)
		defer server.Close()

		conv := NewConversation("sys")
		conv.ApiToken = "test-token"
		conv.SetEndpoint(server.URL)
		conv.Settings.Model = "claude-sonnet-4-6"
		conv.Settings.Temperature = 0.4

		content := []llmapi.ContentBlock{llmapi.NewTextBlock("hello")}
		if _, err := conv.SendRichStreaming(content, llmapi.Sampling{}, nil); err != nil {
			t.Fatalf("SendRichStreaming: %v", err)
		}
		assertTemperaturePresent(t, captured, 0.4, "sonnet-4-6 SendRichStreaming")
	})
}
