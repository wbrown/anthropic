package anthropic

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// Compile-time interface check
var _ llmapi.Conversation = (*Conversation)(nil)

// API URIs
var messagesURI = "https://api.anthropic.com/v1/messages"
var modelsURI = "https://api.anthropic.com/v1/models"

var retries = 3
var retryDelay = 3 * time.Second

// ToolDefinition represents a tool that can be used by Claude
type ToolDefinition struct {
	Name         string          `json:"name"`
	Description  string          `json:"description"`
	InputSchema  json.RawMessage `json:"input_schema"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

// DefaultApiToken is set to the environment variable ANTHROPIC_API_KEY, if
// it exists. It can be overridden by setting it directly. It is used as the
// default API token for all conversations.
var DefaultApiToken = ""

// DefaultSettings is the default settings for the Conversation. It is used
// as the default settings for all Conversations, and can be overridden by
// setting it directly.
var DefaultSettings = SampleSettings{
	Model:       "claude-sonnet-4-20250514",
	Version:     "2023-06-01",
	Beta:        "", // Beta features can be enabled per conversation
	MaxTokens:   20000,
	Temperature: 0.0,
}

// Anthropic Claude API Messages
// Each input message content may be either a single string or an array of
// content blocks, where each block has a specific type. Using a string
// for content is shorthand for an array of one content block of type "text".
//
// The following input messages are equivalent:
//   {"role": "user", "content": "Hello, Claude"}
//   {"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}
//
// We strictly use the content block format for text, as it simplifies the
// serialization and deserialization of messages.

// Message is a single message in a conversation.
type Message struct {
	// Role of the message sender.
	// Possible values are:
	//   "user": The message is from the user.
	//   "assistant": The message is from the assistant.
	Role string `json:"role"`
	// Content is the content of the message, and may be a single string or
	// image, or an array of content blocks.
	Content *[]ContentBlock `json:"content"`
}

// CacheControl specifies caching behavior for content blocks
type CacheControl struct {
	Type string `json:"type"`          // "ephemeral" for caching
	TTL  string `json:"ttl,omitempty"` // "5m" or "1h" (requires beta)
}

// ContentBlock is a single block of content in a message.
type ContentBlock struct {
	ContentType  string           `json:"type"`
	Text         *string          `json:"text,omitempty"`
	Thinking     *string          `json:"thinking,omitempty"`  // For thinking content blocks
	Signature    *string          `json:"signature,omitempty"` // For thinking block verification
	Source       *ContentSource   `json:"source,omitempty"`
	ID           *string          `json:"id,omitempty"`
	Name         *string          `json:"name,omitempty"`
	Input        *json.RawMessage `json:"input,omitempty"`
	ToolUseID    *string          `json:"tool_use_id,omitempty"`
	Content      *string          `json:"content,omitempty"`
	CacheControl *CacheControl    `json:"cache_control,omitempty"`
	tokens       int
}

// SystemPrompt represents a system prompt with optional cache control
type SystemPrompt struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// Messages is a sequence of messages in a conversation. It usually starts with
// a user message, and alternates between user and assistant messages.
type Messages struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
	TopP        float64          `json:"top_p,omitempty"`
	TopK        int              `json:"top_k,omitempty"`
	System      interface{}      `json:"system,omitempty"` // Can be string or []SystemPrompt
	Messages    *[]*Message      `json:"messages"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	Thinking    *ThinkingConfig  `json:"thinking,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

// ContentSource is the encoded data for the content block. It is presently
// used for images only.
type ContentSource struct {
	// Encoding type
	// Possible values are:
	//    base64: The content is base64 encoded.
	Encoding string `json:"type"`
	// MediaType type
	// Possible values are:
	//    image/png: The content is a PNG image.
	//    image/jpeg: The content is a JPEG image.
	//    image/gif: The content is a GIF image.
	//    image/webp: The content is a WebP image.
	MediaType string `json:"media_type"`
	// Data is the base64 encoded image data.
	Data string `json:"data"`
}

// Response is the response from the Anthropic API for a message.
type Response struct {
	// id: The ID of the response.
	ID string `json:"id"`
	// MessageType is the type of the message.
	MessageType string `json:"type"`
	// Role of the message sender. This is usually almost always "assistant".
	Role string `json:"role"`
	// Model is the model used for the response.
	Model string `json:"model"`
	// Content is the content of the response. This usually only has one
	// content block.
	Content *[]ContentBlock `json:"content"`
	// StopReason is the reason the response was stopped. They can be:
	//   "end_turn": The response reached the end of the turn.
	//   "max_tokens": The response reached the maximum token limit.
	//   "stop_sequence": The response reached a stop sequence.
	//   "tools_use": The model invoked one or more tools.
	StopReason string `json:"stop_reason"`
	// StopSequence is the stop sequence that caused the response to stop,
	// in the case of a StopReason of "stop_sequence"
	StopSequence *string `json:"stop_sequence"`
	// Usage is the usage statistics for the response.
	Usage struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	} `json:"usage"`
}

// Streaming response types for SSE parsing
//
// Anthropic's streaming API sends Server-Sent Events (SSE) with these event types:
//   - message_start: Contains initial message metadata and input token count
//   - content_block_start: Signals the start of a content block
//   - content_block_delta: Contains incremental text updates
//   - content_block_stop: Signals the end of a content block
//   - message_delta: Contains final stop_reason and output token count
//   - message_stop: Signals stream completion

// StreamEvent represents an SSE event from the Anthropic streaming API.
// The Type field corresponds to the SSE event name, and other fields are
// populated based on the event type.
type StreamEvent struct {
	Type    string        `json:"type"`
	Message *Response     `json:"message,omitempty"`       // Populated in message_start
	Index   int           `json:"index,omitempty"`         // Content block index
	Delta   *StreamDelta  `json:"delta,omitempty"`         // Populated in content_block_delta and message_delta
	Usage   *StreamUsage  `json:"usage,omitempty"`         // Populated in message_delta
	Content *ContentBlock `json:"content_block,omitempty"` // Populated in content_block_start
}

// StreamDelta contains incremental content updates from streaming responses.
// For content_block_delta events, Text contains the new text fragment.
// For message_delta events, StopReason contains the final stop reason.
type StreamDelta struct {
	Type        string  `json:"type,omitempty"`
	Text        string  `json:"text,omitempty"`
	Thinking    string  `json:"thinking,omitempty"`
	PartialJSON string  `json:"partial_json,omitempty"`
	StopReason  *string `json:"stop_reason,omitempty"`
}

// StreamUsage contains token usage information from streaming responses.
// This is sent in the message_delta event at the end of the stream.
type StreamUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

// SampleSettings is used to set the settings for the request. Usually it
// pertains to samplers.
type SampleSettings struct {
	// Model to use for the sample.
	Model string `json:"model"`
	// Version of the model to use for the sample.
	Version string `json:"version"`
	// Beta is the beta version of the model to use for the sample.
	Beta string `json:"beta"`
	// MaxTokens is the max number of tokens to generate.
	MaxTokens int `json:"max_tokens"`
	// Temperature to use for sampling (0.0-1.0).
	Temperature float64 `json:"temperature"`
	// TopP for nucleus sampling (0.0-1.0). Use either Temperature or TopP, not both.
	TopP float64 `json:"top_p,omitempty"`
	// TopK limits sampling to the top K tokens.
	TopK int `json:"top_k,omitempty"`
	// Thinking configuration for extended reasoning
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig configures extended thinking for Claude
type ThinkingConfig struct {
	Type         string `json:"type"`          // "enabled"
	BudgetTokens int    `json:"budget_tokens"` // minimum 1024
}

// A Conversation is a sequence of messages between a user and an assistant.
type Conversation struct {
	// System is the system prompt to use for the conversation.
	System *string
	// SystemCacheable indicates if the system prompt should be cached
	SystemCacheable bool
	// Messages is the sequence of messages in the conversation. This always
	// starts with a user message, and alternates between user and assistant.
	Messages *[]*Message
	// Usage is the usage statistics for the conversation in total.
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	}
	// CacheStats tracks cache usage statistics
	CacheStats struct {
		TotalCacheCreationTokens int
		TotalCacheReadTokens     int
		TotalTokensSaved         int
		CacheHits                int
		CacheMisses              int
	}
	// HttpClient is the HTTP client used for API requests
	HttpClient *http.Client
	// apiToken is the API token used for API requests
	ApiToken string `json:"-"`
	// Settings is the settings for the conversation.
	Settings *SampleSettings
	// Tools are optional tool definitions for the conversation
	Tools []ToolDefinition
	// ToolsCacheable indicates if tools should be cached
	ToolsCacheable bool
	// HasThinkingContent tracks if any responses included thinking blocks
	HasThinkingContent bool
}

// NewConversation creates a new conversation with the given system prompt. It
// initializes the messages and usage statistics, as well as reasonable
// defaults.
func NewConversation(system string) *Conversation {
	messages := make([]*Message, 0)
	conversation := Conversation{
		System:   &system,
		Messages: &messages,
	}
	conversation.Usage.OutputTokens = 0
	conversation.Usage.InputTokens = 0
	// Copy our default settings
	settings := DefaultSettings
	conversation.Settings = &settings
	conversation.ApiToken = DefaultApiToken

	// Initialize the HTTP client
	conversation.HttpClient = &http.Client{}
	return &conversation
}

func (c *Conversation) sendInternal(text string, sampling llmapi.Sampling) (*Response, error) {
	if c.Settings == nil {
		return nil, fmt.Errorf("conversation settings not set")
	}
	if c.ApiToken == "" {
		return nil, fmt.Errorf("API token not set")
	}
	if text != "" {
		c.AddMessage(llmapi.RoleUser, text)
	} else if len(*c.Messages) > 2 &&
		(*c.Messages)[len(*c.Messages)-1].Role !=
			"assistant" {
		// Check if the last user message contains tool results
		lastMsg := (*c.Messages)[len(*c.Messages)-1]
		if lastMsg.Role == "user" && lastMsg.Content != nil && len(*lastMsg.Content) > 0 {
			// Check if this is a tool result message
			hasToolResult := false
			for _, block := range *lastMsg.Content {
				if block.ContentType == "tool_result" {
					hasToolResult = true
					break
				}
			}
			if !hasToolResult {
				// If the text is empty, and the last message is not from the
				// assistant, we can't continue the conversation, so return
				return nil, fmt.Errorf("cannot continue conversation")
			}
			// If it's a tool result, allow continuation
		} else {
			return nil, fmt.Errorf("cannot continue conversation")
		}
	}

	// Build system prompt with cache control if needed
	var system interface{}
	if c.System != nil && *c.System != "" {
		if c.SystemCacheable {
			// Use array format with cache control
			system = []SystemPrompt{{
				Type:         "text",
				Text:         *c.System,
				CacheControl: &CacheControl{Type: "ephemeral", TTL: "1h"},
			}}
		} else {
			// Use simple string format
			system = c.System
		}
	}

	// Build tools with cache control if needed
	tools := c.Tools
	// Note: The API doesn't support caching individual tools, only the entire tools array
	// Tool caching is handled at the API level, not per-tool

	// Use sampling overrides if provided (non-zero), otherwise use conversation defaults
	temperature := c.Settings.Temperature
	if sampling.Temperature != 0 {
		temperature = sampling.Temperature
	}
	topP := c.Settings.TopP
	if sampling.TopP != 0 {
		topP = sampling.TopP
	}
	topK := c.Settings.TopK
	if sampling.TopK != 0 {
		topK = sampling.TopK
	}

	messages := Messages{
		Model:       c.Settings.Model,
		MaxTokens:   c.Settings.MaxTokens,
		Temperature: temperature,
		TopP:        topP,
		TopK:        topK,
		System:      system,
		Messages:    c.Messages,
		Tools:       tools,
		Thinking:    c.Settings.Thinking,
	}

	// Marshal messages to JSON
	jsonData, marshalErr := json.Marshal(messages)
	if marshalErr != nil {
		return nil, fmt.Errorf("error marshalling to JSON: %s", marshalErr)
	}

	// Debug: Log request if ANTHROPIC_DEBUG is set
	if os.Getenv("ANTHROPIC_DEBUG") == "true" {
		fmt.Printf("DEBUG: Request thinking config: %+v\n", messages.Thinking)
	}

	req, err := http.NewRequest("POST", messagesURI,
		bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %s", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.ApiToken)
	req.Header.Set("anthropic-version", c.Settings.Version)
	if c.Settings.Beta != "" {
		req.Header.Set("anthropic-beta", c.Settings.Beta)
	}

	// Perform API request via HTTP POST
	httpComplete := false
	var resp *http.Response
	errCt := 0
	for !httpComplete {
		var httpErr error
		resp, httpErr = c.HttpClient.Do(req)
		errCt++
		if httpErr != nil && errCt > retries {
			return nil, fmt.Errorf("http error: %s", httpErr)
		} else if httpErr == nil {
			httpComplete = true
		} else {
			time.Sleep(retryDelay)
		}
	}
	if resp == nil {
		return nil, fmt.Errorf("HTTP response is nil")
	}

	defer func(Body io.ReadCloser) {
		if closeErr := Body.Close(); closeErr != nil {
			log.Printf("error closing response body: %v", closeErr)
		}
	}(resp.Body)

	bodyBytes, bodyErr := io.ReadAll(resp.Body)
	if bodyErr != nil {
		return nil, fmt.Errorf("error reading response body: %s", bodyErr)
	}

	// Check response status first
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, bodyBytes)
	}

	// Deserialize response
	var response Response
	if jsonErr := json.Unmarshal(bodyBytes, &response); jsonErr != nil {
		// Check if response looks like HTML
		bodyStr := string(bodyBytes)
		if strings.HasPrefix(strings.TrimSpace(bodyStr), "<") {
			// Likely an HTML error page
			return nil, fmt.Errorf("received HTML error page from API (status %d)", resp.StatusCode)
		}
		return nil, fmt.Errorf("error unmarshaling JSON response: %s", jsonErr)
	}
	if response.MessageType == "error" {
		return nil, fmt.Errorf("API error: %s", bodyBytes)
	}

	// Set tokens on all content blocks
	for i := range *response.Content {
		(*response.Content)[i].tokens = response.Usage.OutputTokens
	}

	return &response, nil
}

// SendRich sends a message with rich content blocks.
func (c *Conversation) SendRich(content []llmapi.ContentBlock, sampling llmapi.Sampling) (*llmapi.RichResponse, error) {
	// Add the content as a user message if provided
	if len(content) > 0 {
		c.AddRichMessage(llmapi.RoleUser, content)
	}

	// Call internal send directly to get full response
	response, err := c.sendInternal("", sampling)
	if err != nil {
		return nil, err
	}

	// Add assistant message with full content blocks to history
	c.addResponseAsMessage(response)

	// Update usage statistics
	c.Usage.InputTokens += response.Usage.InputTokens
	c.Usage.OutputTokens += response.Usage.OutputTokens

	// Update cache statistics
	if response.Usage.CacheCreationInputTokens > 0 {
		c.CacheStats.TotalCacheCreationTokens += response.Usage.CacheCreationInputTokens
		c.CacheStats.CacheMisses++
	}
	if response.Usage.CacheReadInputTokens > 0 {
		c.CacheStats.TotalCacheReadTokens += response.Usage.CacheReadInputTokens
		c.CacheStats.CacheHits++
		regularCost := response.Usage.CacheReadInputTokens
		cacheCost := regularCost / 10
		c.CacheStats.TotalTokensSaved += (regularCost - cacheCost)
	}

	// Convert response content blocks to llmapi format
	llmapiContent := fromAnthropicContentBlocks(*response.Content)

	return &llmapi.RichResponse{
		Content:      llmapiContent,
		StopReason:   response.StopReason,
		InputTokens:  response.Usage.InputTokens,
		OutputTokens: response.Usage.OutputTokens,
	}, nil
}

// SendRichStreaming sends rich content with streaming.
func (c *Conversation) SendRichStreaming(content []llmapi.ContentBlock, sampling llmapi.Sampling, callback llmapi.StreamCallback) (*llmapi.RichResponse, error) {
	if c.Settings == nil {
		return nil, fmt.Errorf("conversation settings not set")
	}
	if c.ApiToken == "" {
		return nil, fmt.Errorf("API token not set")
	}

	if len(content) > 0 {
		c.AddRichMessage(llmapi.RoleUser, content)
	}

	// Build system prompt with cache control if needed
	var system interface{}
	if c.System != nil && *c.System != "" {
		if c.SystemCacheable {
			system = []SystemPrompt{{
				Type:         "text",
				Text:         *c.System,
				CacheControl: &CacheControl{Type: "ephemeral", TTL: "1h"},
			}}
		} else {
			system = c.System
		}
	}

	// Use sampling overrides if provided
	temperature := c.Settings.Temperature
	if sampling.Temperature != 0 {
		temperature = sampling.Temperature
	}
	topP := c.Settings.TopP
	if sampling.TopP != 0 {
		topP = sampling.TopP
	}
	topK := c.Settings.TopK
	if sampling.TopK != 0 {
		topK = sampling.TopK
	}

	messages := Messages{
		Model:       c.Settings.Model,
		MaxTokens:   c.Settings.MaxTokens,
		Temperature: temperature,
		TopP:        topP,
		TopK:        topK,
		System:      system,
		Messages:    c.Messages,
		Tools:       c.Tools,
		Thinking:    c.Settings.Thinking,
		Stream:      true,
	}

	jsonData, marshalErr := json.Marshal(messages)
	if marshalErr != nil {
		return nil, fmt.Errorf("error marshalling to JSON: %s", marshalErr)
	}

	req, err := http.NewRequest("POST", messagesURI, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %s", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.ApiToken)
	req.Header.Set("anthropic-version", c.Settings.Version)
	req.Header.Set("Accept", "text/event-stream")
	if c.Settings.Beta != "" {
		req.Header.Set("anthropic-beta", c.Settings.Beta)
	}

	// Use a client without timeout for streaming
	client := &http.Client{Timeout: 0}
	if c.HttpClient != nil && c.HttpClient.Transport != nil {
		client.Transport = c.HttpClient.Transport
	}

	// Perform request with retries
	var resp *http.Response
	for attempt := 0; attempt <= retries; attempt++ {
		resp, err = client.Do(req)
		if err == nil {
			break
		}
		if attempt < retries {
			time.Sleep(retryDelay)
			req, err = http.NewRequest("POST", messagesURI, bytes.NewBuffer(jsonData))
			if err != nil {
				// Request creation failed, continue to next retry attempt
				continue
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-api-key", c.ApiToken)
			req.Header.Set("anthropic-version", c.Settings.Version)
			req.Header.Set("Accept", "text/event-stream")
			if c.Settings.Beta != "" {
				req.Header.Set("anthropic-beta", c.Settings.Beta)
			}
		}
	}
	if err != nil {
		return nil, fmt.Errorf("HTTP error after %d retries: %s", retries, err)
	}
	if resp == nil {
		return nil, fmt.Errorf("HTTP response is nil")
	}
	defer func(Body io.ReadCloser) {
		if closeErr := Body.Close(); closeErr != nil {
			log.Printf("error closing response body: %v", closeErr)
		}
	}(resp.Body)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	// Parse SSE stream
	reply, stopReason, inputTokens, outputTokens, err := c.parseSSEStreamRich(resp.Body, callback)
	if err != nil {
		return nil, err
	}

	// Add assistant message to history
	c.AddMessage(llmapi.RoleAssistant, reply)

	// Update usage statistics
	c.Usage.InputTokens += inputTokens
	c.Usage.OutputTokens += outputTokens

	return &llmapi.RichResponse{
		Content: []llmapi.ContentBlock{
			llmapi.NewTextBlock(reply),
		},
		StopReason:   stopReason,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}, nil
}

// AddRichMessage adds a message with multiple content blocks.
func (c *Conversation) AddRichMessage(role llmapi.Role, content []llmapi.ContentBlock) {
	blocks := toAnthropicContentBlocks(content)
	msg := Message{
		Role:    string(role),
		Content: &blocks,
	}
	*c.Messages = append(*c.Messages, &msg)
}

// GetRichMessages returns the full conversation with content blocks.
func (c *Conversation) GetRichMessages() []llmapi.RichMessage {
	result := make([]llmapi.RichMessage, len(*c.Messages))
	for i, msg := range *c.Messages {
		result[i] = llmapi.RichMessage{
			Role:    llmapi.Role(msg.Role),
			Content: fromAnthropicContentBlocks(*msg.Content),
		}
	}
	return result
}

// SetTools sets the tools for the conversation.
func (c *Conversation) SetTools(tools []llmapi.ToolDefinition) {
	c.Tools = make([]ToolDefinition, len(tools))
	for i, t := range tools {
		c.Tools[i] = ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
}

// GetTools returns currently configured tools.
func (c *Conversation) GetTools() []llmapi.ToolDefinition {
	result := make([]llmapi.ToolDefinition, len(c.Tools))
	for i, t := range c.Tools {
		result[i] = llmapi.ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	return result
}

// GetCapabilities returns Anthropic's supported features.
func (c *Conversation) GetCapabilities() llmapi.Capabilities {
	// @todo: Probably a way to get this programmatically.
	return llmapi.Capabilities{
		SupportsImages:      true,
		SupportsDocuments:   true,
		SupportsToolUse:     true,
		SupportsThinking:    true,
		SupportsStreaming:   true,
		MaxImageSize:        20 * 1024 * 1024, // 20MB for now? Idk
		SupportedImageTypes: []string{"image/jpeg", "image/png", "image/gif", "image/webp"},
	}
}

// Send sends a message to the assistant and returns the reply. It also
// returns the reason the conversation stopped, the number of input tokens
// used, and the number of output tokens used.
//
// If the text is empty, it sends the message as is, and does not add a user
// message to the conversation. This is useful for continuing an incomplete
// conversation by "assistant", in the case of a stopReason of "max_tokens".
func (conversation *Conversation) Send(text string, sampling llmapi.Sampling) (reply, stopReason string, inputTokens, outputTokens int, err error) {
	// Call internal send (it handles adding user message)
	response, err := conversation.sendInternal(text, sampling)
	if err != nil {
		return "", "", 0, 0, err
	}

	// Extract text for adding to history
	var responseText string
	for i := range *response.Content {
		if (*response.Content)[i].ContentType == "text" {
			responseText = *(*response.Content)[i].Text
		}
	}

	// Add assistant response to history
	conversation.AddMessage(llmapi.RoleAssistant, responseText)

	// Build reply from text and thinking content blocks
	var hasThinking bool
	for _, contentBlock := range *response.Content {
		if os.Getenv("ANTHROPIC_DEBUG") == "true" {
			fmt.Printf("DEBUG: Content block type: %s\n", contentBlock.ContentType)
		}

		if contentBlock.ContentType == "text" && contentBlock.Text != nil {
			reply += *contentBlock.Text
		} else if contentBlock.ContentType == "thinking" && contentBlock.Thinking != nil {
			reply += "<thinking>\n" + *contentBlock.Thinking + "\n</thinking>\n"
			hasThinking = true
		}
	}

	// Track if we've seen thinking content
	if hasThinking {
		conversation.HasThinkingContent = true
		if os.Getenv("ANTHROPIC_DEBUG") == "true" {
			fmt.Println("DEBUG: Found thinking blocks in response")
		}
	}

	// Add usage statistics to conversation
	conversation.Usage.InputTokens += response.Usage.InputTokens
	conversation.Usage.OutputTokens += response.Usage.OutputTokens

	// Update cache statistics
	if response.Usage.CacheCreationInputTokens > 0 {
		conversation.CacheStats.TotalCacheCreationTokens += response.Usage.CacheCreationInputTokens
		conversation.CacheStats.CacheMisses++
	}

	if response.Usage.CacheReadInputTokens > 0 {
		conversation.CacheStats.TotalCacheReadTokens += response.Usage.CacheReadInputTokens
		conversation.CacheStats.CacheHits++

		// Calculate savings (cache reads are 90% cheaper)
		regularCost := response.Usage.CacheReadInputTokens
		cacheCost := regularCost / 10
		conversation.CacheStats.TotalTokensSaved += (regularCost - cacheCost)
	}

	return reply, response.StopReason, response.Usage.InputTokens, response.Usage.OutputTokens, nil
}

// SendStreaming sends a message with real-time token streaming via SSE.
// The callback is invoked for each text fragment received from the API.
//
// Parameters:
//   - text: The user message to send. If empty, continues from the last message.
//   - sampling: Sampling parameters to override conversation defaults (Temperature, TopP, TopK).
//   - callback: Called with each text fragment; called with ("", true) when done.
//
// Returns the same values as Send, but tokens are streamed via callback as they arrive.
func (conversation *Conversation) SendStreaming(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (reply, stopReason string, inputTokens, outputTokens int, err error) {
	if conversation.Settings == nil {
		return "", "", 0, 0, fmt.Errorf("conversation settings not set")
	}
	if conversation.ApiToken == "" {
		return "", "", 0, 0, fmt.Errorf("API token not set")
	}

	// Add user message if provided
	if text != "" {
		conversation.AddMessage(llmapi.RoleUser, text)
	} else if len(*conversation.Messages) > 0 &&
		(*conversation.Messages)[len(*conversation.Messages)-1].Role != "assistant" {
		// Check if the last user message contains tool results
		lastMsg := (*conversation.Messages)[len(*conversation.Messages)-1]
		if lastMsg.Role == "user" && lastMsg.Content != nil && len(*lastMsg.Content) > 0 {
			hasToolResult := false
			for _, block := range *lastMsg.Content {
				if block.ContentType == "tool_result" {
					hasToolResult = true
					break
				}
			}
			if !hasToolResult {
				return "", "", 0, 0, fmt.Errorf("cannot continue conversation")
			}
		} else {
			return "", "", 0, 0, fmt.Errorf("cannot continue conversation")
		}
	}

	// Build system prompt with cache control if needed
	var system interface{}
	if conversation.System != nil && *conversation.System != "" {
		if conversation.SystemCacheable {
			system = []SystemPrompt{{
				Type:         "text",
				Text:         *conversation.System,
				CacheControl: &CacheControl{Type: "ephemeral", TTL: "1h"},
			}}
		} else {
			system = conversation.System
		}
	}

	// Use sampling overrides if provided (non-zero), otherwise use conversation defaults
	temperature := conversation.Settings.Temperature
	if sampling.Temperature != 0 {
		temperature = sampling.Temperature
	}
	topP := conversation.Settings.TopP
	if sampling.TopP != 0 {
		topP = sampling.TopP
	}
	topK := conversation.Settings.TopK
	if sampling.TopK != 0 {
		topK = sampling.TopK
	}

	messages := Messages{
		Model:       conversation.Settings.Model,
		MaxTokens:   conversation.Settings.MaxTokens,
		Temperature: temperature,
		TopP:        topP,
		TopK:        topK,
		System:      system,
		Messages:    conversation.Messages,
		Tools:       conversation.Tools,
		Thinking:    conversation.Settings.Thinking,
		Stream:      true,
	}

	jsonData, marshalErr := json.Marshal(messages)
	if marshalErr != nil {
		return "", "", 0, 0, fmt.Errorf("error marshalling to JSON: %s", marshalErr)
	}

	req, err := http.NewRequest("POST", messagesURI, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error creating HTTP request: %s", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", conversation.ApiToken)
	req.Header.Set("anthropic-version", conversation.Settings.Version)
	req.Header.Set("Accept", "text/event-stream")
	if conversation.Settings.Beta != "" {
		req.Header.Set("anthropic-beta", conversation.Settings.Beta)
	}

	// Use a client without timeout for streaming
	client := &http.Client{Timeout: 0}
	if conversation.HttpClient != nil && conversation.HttpClient.Transport != nil {
		client.Transport = conversation.HttpClient.Transport
	}

	// Perform request with retries
	var resp *http.Response
	for attempt := 0; attempt <= retries; attempt++ {
		resp, err = client.Do(req)
		if err == nil {
			break
		}
		if attempt < retries {
			time.Sleep(retryDelay)
			req, err = http.NewRequest("POST", messagesURI, bytes.NewBuffer(jsonData))
			if err != nil {
				// Request creation failed, continue to next retry attempt
				continue
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-api-key", conversation.ApiToken)
			req.Header.Set("anthropic-version", conversation.Settings.Version)
			req.Header.Set("Accept", "text/event-stream")
			if conversation.Settings.Beta != "" {
				req.Header.Set("anthropic-beta", conversation.Settings.Beta)
			}
		}
	}
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("HTTP error after %d retries: %s", retries, err)
	}
	if resp == nil {
		return "", "", 0, 0, fmt.Errorf("HTTP response is nil")
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {
			log.Printf("error closing response body: %v", err)
		}
	}(resp.Body)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", "", 0, 0, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	// Parse SSE stream
	reply, stopReason, inputTokens, outputTokens, err = conversation.parseSSEStreamRich(resp.Body, callback)
	if err != nil {
		return reply, stopReason, inputTokens, outputTokens, err
	}

	// Add assistant message to history
	conversation.AddMessage(llmapi.RoleAssistant, reply)

	// Update usage statistics
	conversation.Usage.InputTokens += inputTokens
	conversation.Usage.OutputTokens += outputTokens

	return reply, stopReason, inputTokens, outputTokens, nil
}

// parseSSEStreamRich reads Server-Sent Events from the response body and processes them.
// It extracts text deltas, token counts, and stop reason from the stream.
//
// SSE format from Anthropic:
//
//	event: message_start
//	data: {"type":"message_start","message":{...}}
//
//	event: content_block_delta
//	data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}
//
//	event: message_delta
//	data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}
//
//	event: message_stop
//	data: {"type":"message_stop"}
func (conversation *Conversation) parseSSEStreamRich(body io.Reader, callback llmapi.StreamCallback) (
	fullText string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	scanner := bufio.NewScanner(body)
	var currentBlockType string
	var textBuilder strings.Builder
	var thinkingBuilder strings.Builder
	var currentEvent string

	for scanner.Scan() {
		line := scanner.Text()

		// SSE events come as "event: <name>" followed by "data: <json>"
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		// Skip non-data lines (empty lines, comments, etc.)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var event StreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			// Skip malformed events rather than failing the whole stream
			continue
		}

		switch currentEvent {
		case "message_start":
			// Initial event contains input token count
			if event.Message != nil {
				inputTokens = event.Message.Usage.InputTokens
			}

		case "content_block_start":
			// Track what type of block we're starting
			if event.Content != nil {
				currentBlockType = event.Content.ContentType
			}

		case "content_block_delta":
			if event.Delta != nil {
				switch event.Delta.Type {
				case "text_delta":
					textBuilder.WriteString(event.Delta.Text)
					if callback != nil {
						callback(event.Delta.Text, false)
					}
				case "thinking_delta":
					thinkingBuilder.WriteString(event.Delta.Thinking)
					// Optionally stream thinking to callback too
					if callback != nil && event.Delta.Thinking != "" {
						callback(event.Delta.Thinking, false)
					}
				}
			}

		case "content_block_stop":
			// Block finished - could finalize here if needed
			currentBlockType = ""

		case "message_delta":
			// Final event before message_stop - contains stop reason and output tokens
			if event.Delta != nil && event.Delta.StopReason != nil {
				stopReason = *event.Delta.StopReason
			}
			if event.Usage != nil {
				outputTokens = event.Usage.OutputTokens
			}

		case "message_stop":
			// Stream complete - signal done to callback
			if callback != nil {
				callback("", true)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return textBuilder.String(), stopReason, inputTokens, outputTokens, fmt.Errorf("error reading stream: %w", err)
	}

	// Build full text with thinking if present
	var result strings.Builder
	if thinkingBuilder.Len() > 0 {
		result.WriteString("<thinking>\n")
		result.WriteString(thinkingBuilder.String())
		result.WriteString("\n</thinking>\n")
		conversation.HasThinkingContent = true
	}
	result.WriteString(textBuilder.String())

	// Suppress unused variable warning
	_ = currentBlockType

	return result.String(), stopReason, inputTokens, outputTokens, nil
}

func (conversation *Conversation) SendUntilDone(text string, sampling llmapi.Sampling) (reply, stopReason string, inputTokens, outputTokens int, err error) {
	output := ""
	done := false
	input := text
	for !done {
		var inputTks, outputTks int
		var reply string

		reply, stopReason, inputTks, outputTks, err =
			conversation.Send(input, sampling)
		if err != nil {
			return output, stopReason, inputTokens, outputTokens, err
		}

		// Accumulate the output and token counts
		output += reply
		inputTokens += inputTks
		outputTokens += outputTks

		// Merge the last two assistant messages if from the assistant
		conversation.MergeIfLastTwoAssistant()

		// The only reason we would continue is if the stop reason is
		// "max_tokens", in which case we continue the conversation
		if stopReason != "max_tokens" {
			done = true
		} else {
			input = ""
		}
	}
	return output, stopReason, inputTokens, outputTokens, nil
}

// SendStreamingUntilDone combines streaming with automatic continuation.
// It calls SendStreaming repeatedly until stopReason != "max_tokens",
// streaming tokens via callback throughout the entire generation.
//
// This is useful for long responses that exceed max_tokens - the callback
// receives a continuous stream of tokens across all continuation requests,
// while the returned reply contains the complete accumulated response.
//
// Consecutive assistant messages are automatically merged in the conversation
// history to maintain a clean message structure.
func (conversation *Conversation) SendStreamingUntilDone(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (reply, stopReason string, inputTokens, outputTokens int, err error) {
	var totalReply strings.Builder
	input := text

	for {
		var partReply string
		var inToks, outToks int

		partReply, stopReason, inToks, outToks, err = conversation.SendStreaming(input, sampling, callback)
		if err != nil {
			return totalReply.String(), stopReason, inputTokens, outputTokens, err
		}

		totalReply.WriteString(partReply)
		inputTokens += inToks
		outputTokens += outToks

		// Merge consecutive assistant messages to keep history clean
		conversation.MergeIfLastTwoAssistant()

		if stopReason != "max_tokens" {
			break
		}

		// Continue generation with empty input (picks up from last assistant message)
		input = ""
	}

	return totalReply.String(), stopReason, inputTokens, outputTokens, nil
}

// AddMessage adds a message to the conversation with the given role and
// content. It is used internally and can also be used externally to
// manipulate conversations.
func (conversation *Conversation) AddMessage(role llmapi.Role, content string) {

	if content != "" {
		contentBlock := ContentBlock{
			ContentType: "text",
			Text:        &content,
		}
		message := Message{
			Role:    string(role),
			Content: &[]ContentBlock{contentBlock},
		}

		*conversation.Messages = append(*conversation.Messages, &message)
	}
}

// GetMessages returns the conversation history as llmapi.Message slices.
// This converts from the internal ContentBlock-based format to simple
// role/content strings for interface compliance.
//
// Note: Only the first text content block from each message is extracted.
// Tool use, images, and other content types are not included in the output.
func (conversation *Conversation) GetMessages() []llmapi.Message {
	if conversation.Messages == nil {
		return nil
	}
	result := make([]llmapi.Message, 0, len(*conversation.Messages))
	for _, msg := range *conversation.Messages {
		var content string
		if msg.Content != nil && len(*msg.Content) > 0 {
			// Extract text from first text content block
			for _, block := range *msg.Content {
				if block.ContentType == "text" && block.Text != nil {
					content = *block.Text
					break
				}
			}
		}
		result = append(result, llmapi.Message{
			Role:    llmapi.Role(msg.Role),
			Content: content,
		})
	}
	return result
}

// GetUsage returns the cumulative token usage for this conversation.
// This includes all input and output tokens across all Send calls.
func (conversation *Conversation) GetUsage() llmapi.Usage {
	return llmapi.Usage{
		InputTokens:  conversation.Usage.InputTokens,
		OutputTokens: conversation.Usage.OutputTokens,
	}
}

// GetSystem returns the system prompt for this conversation.
func (conversation *Conversation) GetSystem() string {
	if conversation.System == nil {
		return ""
	}
	return *conversation.System
}

// Clear resets the conversation history and usage statistics.
// The system prompt and settings are preserved.
func (conversation *Conversation) Clear() {
	messages := make([]*Message, 0)
	conversation.Messages = &messages
	conversation.Usage.InputTokens = 0
	conversation.Usage.OutputTokens = 0
}

// SetModel changes the model for subsequent API calls.
// If settings haven't been initialized, they are created from DefaultSettings.
func (conversation *Conversation) SetModel(model string) {
	if conversation.Settings == nil {
		settings := DefaultSettings
		conversation.Settings = &settings
	}
	conversation.Settings.Model = model
}

// MergeIfLastTwoAssistant merges the last two assistant messages if they are
// both from the assistant. This is useful for combining messages that are
// split and continued due to token limits.
//
// This is used on Conversation on all API returns, as it is a no-op if
// there are less than two messages, or if the last two messages are not
// both from the assistant.
func (conversation *Conversation) MergeIfLastTwoAssistant() {
	// If we have less than two messages, we can't merge, so return
	if len(*conversation.Messages) < 2 {
		return
	}

	lastIdx := len(*conversation.Messages) - 1
	lastMessage := (*conversation.Messages)[lastIdx]
	if lastMessage.Role != "assistant" {
		return
	}

	secondLastIdx := len(*conversation.Messages) - 2
	secondLastMessage := (*conversation.Messages)[secondLastIdx]
	if secondLastMessage.Role != "assistant" {
		return
	}

	// Merge the last two assistant messages
	secondLastContent := (*secondLastMessage.Content)[0]
	secondLastText := *secondLastContent.Text
	lastContent := (*lastMessage.Content)[0]
	lastText := *lastContent.Text
	lastText = strings.TrimSpace(lastText)

	// Trim whitespace from the right of the prior message. Anthropic
	// does not permit assistant messages to end with whitespace.
	secondLastText = strings.TrimRight(secondLastText, " \t\n\r")
	secondLastText = secondLastText + lastText
	secondLastContent.Text = &secondLastText
	secondLastContent.tokens += lastContent.tokens

	// Insert our merged message in place of the second last message
	(*conversation.Messages)[secondLastIdx].Content =
		&[]ContentBlock{secondLastContent}
	// Remove the last message, as we merged it into the second last
	// message
	*conversation.Messages = (*conversation.Messages)[:lastIdx]
}

// init loads the API token from token files or environment variable.
// Priority: ./.anthropic_key > ~/.anthropic_key > ANTHROPIC_API_KEY env var
func init() {
	// 1. Current directory token file (highest priority)
	if token := readTokenFile(".anthropic_key"); token != "" {
		DefaultApiToken = token
		return
	}

	// 2. Home directory token file
	if home, err := os.UserHomeDir(); err == nil {
		if token := readTokenFile(home + "/.anthropic_key"); token != "" {
			DefaultApiToken = token
			return
		}
	}

	// 3. Environment variable (lowest priority)
	if token := os.Getenv("ANTHROPIC_API_KEY"); token != "" {
		DefaultApiToken = token
	}
}

// readTokenFile reads a token from a file, returning empty string on error.
func readTokenFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// Helper methods for cache control

// EnableCaching adds cache control to a content block (5-minute TTL by default)
func (cb *ContentBlock) EnableCaching() {
	cb.CacheControl = &CacheControl{Type: "ephemeral"}
}

// EnableLongCaching adds cache control with 1-hour TTL (requires beta)
func (cb *ContentBlock) EnableLongCaching() {
	cb.CacheControl = &CacheControl{Type: "ephemeral", TTL: "1h"}
}

// DisableCaching removes cache control from a content block
func (cb *ContentBlock) DisableCaching() {
	cb.CacheControl = nil
}

// CacheHitRate returns the cache hit rate as a percentage
func (c *Conversation) CacheHitRate() float64 {
	total := c.CacheStats.CacheHits + c.CacheStats.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(c.CacheStats.CacheHits) / float64(total) * 100
}

// CacheSavingsRate returns the percentage of tokens saved by caching
func (c *Conversation) CacheSavingsRate() float64 {
	if c.Usage.InputTokens == 0 {
		return 0
	}
	return float64(c.CacheStats.TotalTokensSaved) / float64(c.Usage.InputTokens) * 100
}

// Beta feature management

// EnableBeta adds a beta feature to the conversation settings
func (c *Conversation) EnableBeta(beta string) {
	if c.Settings == nil {
		settings := DefaultSettings
		c.Settings = &settings
	}

	if c.Settings.Beta == "" {
		c.Settings.Beta = beta
	} else if !strings.Contains(c.Settings.Beta, beta) {
		// Add comma-separated beta
		c.Settings.Beta = c.Settings.Beta + "," + beta
	}
}

// DisableBeta removes a beta feature from the conversation settings
func (c *Conversation) DisableBeta(beta string) {
	if c.Settings == nil || c.Settings.Beta == "" {
		return
	}

	// Split existing betas
	betas := strings.Split(c.Settings.Beta, ",")
	var newBetas []string

	for _, b := range betas {
		b = strings.TrimSpace(b)
		if b != beta {
			newBetas = append(newBetas, b)
		}
	}

	c.Settings.Beta = strings.Join(newBetas, ",")
}

// EnableThinking enables extended thinking with the specified token budget
func (c *Conversation) EnableThinking(budgetTokens int) {
	if c.Settings == nil {
		settings := DefaultSettings
		c.Settings = &settings
	}

	if budgetTokens < 1024 {
		budgetTokens = 1024 // Minimum required
	}
	c.Settings.Thinking = &ThinkingConfig{
		Type:         "enabled",
		BudgetTokens: budgetTokens,
	}
}

// Model represents an available Claude model
type Model struct {
	Type        string    `json:"type"`
	ID          string    `json:"id"`
	DisplayName string    `json:"display_name"`
	CreatedAt   time.Time `json:"created_at"`
}

// ModelsResponse represents the response from the models list API
type ModelsResponse struct {
	Data    []Model `json:"data"`
	FirstID string  `json:"first_id"`
	LastID  string  `json:"last_id"`
	HasMore bool    `json:"has_more"`
}

// ListModels retrieves the list of available models
func ListModels(apiKey string) (*ModelsResponse, error) {
	if apiKey == "" {
		apiKey = DefaultApiToken
	}
	if apiKey == "" {
		return nil, fmt.Errorf("API key not provided")
	}

	req, err := http.NewRequest("GET", modelsURI, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", DefaultSettings.Version)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {
			log.Printf("error closing response body: %v", err)
		}
	}(resp.Body)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	var modelsResp ModelsResponse
	if err := json.Unmarshal(body, &modelsResp); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	return &modelsResp, nil
}

// EnableSystemCaching enables caching for the system prompt
func (c *Conversation) EnableSystemCaching() {
	c.SystemCacheable = true
}

// EnableToolsCaching enables caching for tool definitions
func (c *Conversation) EnableToolsCaching() {
	c.ToolsCacheable = true
}

// CacheLastNMessages enables caching on the last N messages
// This is useful for caching accumulated tool results
func (c *Conversation) CacheLastNMessages(n int) {
	if c.Messages == nil || len(*c.Messages) == 0 {
		return
	}

	messages := *c.Messages
	start := len(messages) - n
	if start < 0 {
		start = 0
	}

	for i := start; i < len(messages); i++ {
		if messages[i].Content != nil {
			for j := range *messages[i].Content {
				(*messages[i].Content)[j].EnableCaching()
			}
		}
	}
}

// @todo: Move this into a utilities file.

// toAnthropicContentBlock converts a llmapi ContentBlock to anthropic format.
func toAnthropicContentBlock(block llmapi.ContentBlock) ContentBlock {
	cb := ContentBlock{ContentType: string(block.Type)}

	switch block.Type {
	case llmapi.ContentTypeText:
		cb.Text = &block.Text
	case llmapi.ContentTypeImage:
		if block.Image != nil {
			cb.Source = &ContentSource{
				Encoding:  block.Image.Source.Type,
				MediaType: string(block.Image.Source.MediaType),
				Data:      block.Image.Source.Data,
			}
		}
	case llmapi.ContentTypeToolUse:
		if block.ToolUse != nil {
			cb.ID = &block.ToolUse.ID
			cb.Name = &block.ToolUse.Name
			cb.Input = &block.ToolUse.Input
		}
	case llmapi.ContentTypeToolResult:
		if block.ToolResult != nil {
			cb.ToolUseID = &block.ToolResult.ToolUseID
			cb.Content = &block.ToolResult.Content
		}
	case llmapi.ContentTypeThinking:
		if block.Thinking != nil {
			cb.Thinking = &block.Thinking.Thinking
			if block.Thinking.Signature != "" {
				cb.Signature = &block.Thinking.Signature
			}
		}
	}
	return cb
}

// toAnthropicContentBlocks converts an array of llmapi ContentBlocks to anthropic format.
func toAnthropicContentBlocks(blocks []llmapi.ContentBlock) []ContentBlock {
	result := make([]ContentBlock, len(blocks))
	for i, block := range blocks {
		result[i] = toAnthropicContentBlock(block)
	}
	return result
}

// fromAnthropicContentBlock converts an anthropic ContentBlock to llmapi format.
func fromAnthropicContentBlock(block ContentBlock) llmapi.ContentBlock {
	cb := llmapi.ContentBlock{Type: llmapi.ContentType(block.ContentType)}

	switch block.ContentType {
	case "text":
		if block.Text != nil {
			cb.Text = *block.Text
		}
	case "image":
		if block.Source != nil {
			cb.Image = &llmapi.ImageContent{
				Source: llmapi.ImageSource{
					Type:      block.Source.Encoding,
					MediaType: llmapi.MediaType(block.Source.MediaType),
					Data:      block.Source.Data,
				},
			}
		}

	case "tool_use":
		cb.ToolUse = &llmapi.ToolUseContent{
			ID:    derefString(block.ID),
			Name:  derefString(block.Name),
			Input: derefRawMessage(block.Input),
		}

	case "tool_result":
		cb.ToolResult = &llmapi.ToolResultContent{
			ToolUseID: derefString(block.ToolUseID),
			Content:   derefString(block.Content),
		}
	case "thinking":
		if block.Thinking != nil {
			cb.Thinking = &llmapi.ThinkingContent{
				Thinking:  *block.Thinking,
				Signature: derefString(block.Signature),
			}
		}
	}
	return cb
}

// fromAnthropicContentBlocks converts an array of anthropic content blocks into llmapi ones.
func fromAnthropicContentBlocks(blocks []ContentBlock) []llmapi.ContentBlock {
	result := make([]llmapi.ContentBlock, len(blocks))
	for i, block := range blocks {
		result[i] = fromAnthropicContentBlock(block)
	}
	return result
}

// Helper functions

func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func derefRawMessage(r *json.RawMessage) json.RawMessage {
	if r == nil {
		return nil
	}
	return *r
}

// addResponseAsMessage - Add response content blocks as assistant messages
func (c *Conversation) addResponseAsMessage(response *Response) {
	// Store ALL content blocks, not just text
	msg := Message{
		Role:    "assistant",
		Content: response.Content, // keep tool_use, thinking, etc.
	}
	*c.Messages = append(*c.Messages, &msg)
}

// flattenResponseToString - flattens response to string for backwards compatability.
func (c *Conversation) flattenResponseToString(response *Response) string {
	var reply string
	for _, block := range *response.Content {
		if block.ContentType == "text" && block.Text != nil {
			reply += derefString(block.Text)
		} else if block.ContentType == "thinking" && block.Thinking != nil {
			reply += "<thinking>\n" + derefString(block.Thinking) + "\n</thinking>\n"
		}
	}
	return reply
}
