package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

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

// Messages is a sequence of messages in a conversation. It usually always
// a user message first, and alternates between user and assistant messages.
type Messages struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
	System      interface{}      `json:"system,omitempty"` // Can be string or []SystemPrompt
	Messages    *[]*Message      `json:"messages"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	Thinking    *ThinkingConfig  `json:"thinking,omitempty"`
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

// SampleSettings is used to set the settings for the request. Usually it
// pertains to samplers.
type SampleSettings struct {
	// Model to use for the sample.
	Model string `json:"model"`
	// Version of the model to use for the sample.
	Version string `json:"version"`
	// Beta is the beta version of the model to use for the sample.
	Beta string `json:"beta"`
	// MaxTokens is the mas number of tokens to generate.
	MaxTokens int `json:"max_tokens"`
	// The temperature to use for sampling.
	Temperature float64 `json:"temperature"`
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

// AddMessage adds a message to the conversation with the given role and
// content. It used internally, and also can be used externally to
// manipulate conversations.
func (conversation *Conversation) AddMessage(
	role string,
	content *[]ContentBlock,
) {
	message := Message{
		Role:    role,
		Content: content,
	}
	*conversation.Messages = append(*conversation.Messages, &message)
}

// API URIs
var messagesURI = "https://api.anthropic.com/v1/messages"
var modelsURI = "https://api.anthropic.com/v1/models"

// API default headers
var headers = map[string]string{
	"Content-Type":      "application/json",
	"x-api-key":         "",
	"anthropic_version": "2023-06-01",
	// "anthropic-beta":    "max-tokens-3-5-sonnet-2024-07-15",
}

var retries = 3
var retryDelay = 3 * time.Second

// Send sends a message to the assistant and returns the reply. It also
// returns the reason the conversation stopped, the number of input tokens
// used, and the number of output tokens used.
//
// If the text is empty, it sends the message as is, and does not add a user
// message to the conversation. This is useful for continuing an incomplete
// conversation by "assistant", in the case of a stopReason of "max_tokens".
func (conversation *Conversation) Send(text string) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	if conversation.Settings == nil {
		return "", "", 0, 0,
			fmt.Errorf("conversation settings not set")
	}
	if conversation.ApiToken == "" {
		return "", "", 0, 0,
			fmt.Errorf("API token not set")
	}
	if text != "" {
		// Form a basic message
		contentBlock := ContentBlock{
			ContentType: "text",
			Text:        &text,
		}
		// Add message to conversation
		conversation.AddMessage("user", &[]ContentBlock{contentBlock})
	} else if len(*conversation.Messages) > 2 &&
		(*conversation.Messages)[len(*conversation.Messages)-1].Role !=
			"assistant" {
		// Check if the last user message contains tool results
		lastMsg := (*conversation.Messages)[len(*conversation.Messages)-1]
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
				return "", "", 0, 0,
					fmt.Errorf("cannot continue conversation")
			}
			// If it's a tool result, allow continuation
		} else {
			return "", "", 0, 0,
				fmt.Errorf("cannot continue conversation")
		}
	}

	// Build system prompt with cache control if needed
	var system interface{}
	if conversation.System != nil && *conversation.System != "" {
		if conversation.SystemCacheable {
			// Use array format with cache control
			system = []SystemPrompt{{
				Type:         "text",
				Text:         *conversation.System,
				CacheControl: &CacheControl{Type: "ephemeral", TTL: "1h"},
			}}
		} else {
			// Use simple string format
			system = conversation.System
		}
	}

	// Build tools with cache control if needed
	tools := conversation.Tools
	// Note: The API doesn't support caching individual tools, only the entire tools array
	// Tool caching is handled at the API level, not per-tool

	messages := Messages{
		Model:       conversation.Settings.Model,
		MaxTokens:   conversation.Settings.MaxTokens,
		Temperature: conversation.Settings.Temperature,
		System:      system,
		Messages:    conversation.Messages,
		Tools:       tools,
		Thinking:    conversation.Settings.Thinking,
	}

	// Marshal messages to JSON
	jsonData, marshalErr := json.Marshal(messages)
	if marshalErr != nil {
		return "", "", 0, 0,
			fmt.Errorf("error marshalling to JSON: %s", marshalErr)
	}

	// Debug: Log request if ANTHROPIC_DEBUG is set
	if os.Getenv("ANTHROPIC_DEBUG") == "true" {
		fmt.Printf("DEBUG: Request thinking config: %+v\n", messages.Thinking)
	}

	req, err := http.NewRequest("POST", messagesURI,
		bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", 0, 0,
			fmt.Errorf("error creating HTTP request: %s", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", conversation.ApiToken)
	req.Header.Set("anthropic-version", conversation.Settings.Version)
	if conversation.Settings.Beta != "" {
		req.Header.Set("anthropic-beta", conversation.Settings.Beta)
	}

	// req.Header.Set("anthropic-beta", "max-tokens-3-5-sonnet-2024-07-15")
	// Perform API request via HTTP POST
	httpComplete := false
	var resp *http.Response
	errCt := 0
	for !httpComplete {
		var httpErr error
		resp, httpErr = conversation.HttpClient.Do(req)
		errCt++
		if httpErr != nil && errCt > retries {
			return "", "", 0, 0,
				fmt.Errorf("http error: %s", httpErr)
		} else if httpErr == nil {
			httpComplete = true
		} else {
			time.Sleep(retryDelay)
		}
	}
	if resp == nil {
		return "", "", 0, 0,
			fmt.Errorf("HTTP response is nil")
	}

	defer func(Body io.ReadCloser) {
		closeErr := Body.Close()
		if closeErr != nil {
			panic(closeErr)
		}
	}(resp.Body)

	bodyBytes, bodyErr := io.ReadAll(resp.Body)
	if bodyErr != nil {
		return "", "", 0, 0,
			fmt.Errorf("error reading response body: %s", bodyErr)
	}

	// Check response status first
	if resp.StatusCode != http.StatusOK {
		return "", "", 0, 0,
			fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	// Deserialize response
	var response Response
	if jsonErr := json.Unmarshal(bodyBytes, &response); jsonErr != nil {
		// Check if response looks like HTML
		bodyStr := string(bodyBytes)
		if strings.HasPrefix(strings.TrimSpace(bodyStr), "<") {
			// Likely an HTML error page
			return "", "", 0, 0,
				fmt.Errorf("received HTML error page from API (status %d)", resp.StatusCode)
		}
		return "", "", 0, 0,
			fmt.Errorf("error unmarshaling JSON response: %s", jsonErr)
	}
	if response.MessageType == "error" {
		return string(bodyBytes), "", 0, 0,
			fmt.Errorf("API error: %s", bodyBytes)
	}

	reply = ""
	// Set tokens on all content blocks before adding
	for i := range *response.Content {
		(*response.Content)[i].tokens = response.Usage.OutputTokens
	}

	// Add ALL response content as a single message
	conversation.AddMessage("assistant", response.Content)

	// Build reply from text and thinking content blocks
	var hasThinking bool
	for _, contentBlock := range *response.Content {
		// Debug: Log all content block types
		if os.Getenv("ANTHROPIC_DEBUG") == "true" {
			fmt.Printf("DEBUG: Content block type: %s\n", contentBlock.ContentType)
		}

		if contentBlock.ContentType == "text" && contentBlock.Text != nil {
			reply += *contentBlock.Text
		} else if contentBlock.ContentType == "thinking" && contentBlock.Thinking != nil {
			// Include thinking in the reply with tags
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

	// It is difficult to distinguish the usage of the system prompt vs
	// the user message, so we will just use the input tokens from the
	// response.
	inputTokens = response.Usage.InputTokens
	outputTokens = response.Usage.OutputTokens
	// Scan for last user message, and update the input tokens
	for i := len(*conversation.Messages) - 1; i >= 0; i-- {
		if (*conversation.Messages)[i].Role == "user" {
			(*(*conversation.Messages)[i].Content)[0].tokens = inputTokens
			break
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

	return reply,
		response.StopReason,
		inputTokens,
		outputTokens,
		nil
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

// SendUntilDone sends a message to the assistant, and continues sending
// messages until the conversation is done. It returns the full output
// text, the reason the conversation stopped, the total number of input
// tokens used, the total number of output tokens used, and any error that
// occurred.
func (conversation *Conversation) SendUntilDone(
	text string,
) (
	output string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	output = ""
	done := false
	input := text
	for !done {
		var inputTks, outputTks int
		var reply string

		reply, stopReason, inputTks, outputTks, err =
			conversation.Send(input)
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

// Obtain API key from environment variable
func init() {
	// Check for API key in environment variable
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "ANTHROPIC_API_KEY=") {
			DefaultApiToken = strings.Split(e, "=")[1]
		}
	}
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
	defer resp.Body.Close()

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
