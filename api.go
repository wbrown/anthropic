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

// DefaultApiToken is set to the environment variable ANTHROPIC_API_KEY, if
// it exists. It can be overridden by setting it directly. It is used as the
// default API token for all conversations.
var DefaultApiToken = ""

// DefaultSettings is the default settings for the Conversation. It is used
// as the default settings for all Conversations, and can be overridden by
// setting it directly.
var DefaultSettings = SampleSettings{
	Model:       "claude-3-5-sonnet-20240620",
	Version:     "2023-06-01",
	Beta:        "", // "max-tokens-3-5-sonnet-2024-07-15"
	MaxTokens:   4096,
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

// ContentBlock is a single block of content in a message.
type ContentBlock struct {
	// ContentType is the type of the content block.
	// Possible values are:
	//   "text": The content is a string of text.
	//   "image": The content is an image.
	ContentType string `json:"type"`
	// Text is the text content.
	Text *string `json:"text,omitempty"`
	// Source is the source of the content.
	Source *ContentSource `json:"source,omitempty"`
	// tokens is the number of tokens used for the content block, it is an
	// internally used field.
	tokens int
}

// Messages is a sequence of messages in a conversation. It usually always
// a user message first, and alternates between user and assistant messages.
type Messages struct {
	// Model is the model to use for the messages.
	Model string `json:"model"`
	// MaxTokens is the maximum number of tokens to generate.
	MaxTokens int `json:"max_tokens"`
	// Temperature is the temperature to use for sampling. Ranges from 0 to 1.
	Temperature float64 `json:"temperature"`
	// System is the system prompt to use for the conversation.
	System *string `json:"system"`
	// Messages is the sequence of messages in the conversation. They must
	// alternate between user and assistant messages.
	Messages *[]*Message `json:"messages"`
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
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	}
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
}

// A Conversation is a sequence of messages between a user and an assistant.
type Conversation struct {
	// System is the system prompt to use for the conversation.
	System *string
	// Messages is the sequence of messages in the conversation. This always
	// starts with a user message, and alternates between user and assistant.
	Messages *[]*Message
	// Usage is the usage statistics for the conversation in total.
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	}
	// HttpClient is the HTTP client used for API requests
	HttpClient *http.Client
	// apiToken is the API token used for API requests
	ApiToken string `json:"-"`
	// Settings is the settings for the conversation.
	Settings *SampleSettings
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

// Messages API URI
// POST /v1/messages
var messagesURI = "https://api.anthropic.com/v1/messages"

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
		// If the text is empty, and the last message is not from the
		// assistant, we can't continue the conversation, so return
		return "", "", 0, 0,
			fmt.Errorf("cannot continue conversation")
	}

	messages := Messages{
		Model:       conversation.Settings.Model,
		MaxTokens:   conversation.Settings.MaxTokens,
		Temperature: conversation.Settings.Temperature,
		System:      conversation.System,
		Messages:    conversation.Messages,
	}

	// Marshal messages to JSON
	jsonData, marshalErr := json.Marshal(messages)
	if marshalErr != nil {
		return "", "", 0, 0,
			fmt.Errorf("error marshalling to JSON: %s", marshalErr)
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

	// Deserialize response
	var response Response
	if jsonErr := json.Unmarshal(bodyBytes, &response); jsonErr != nil {
		return string(bodyBytes), "", 0, 0,
			fmt.Errorf(
				"error unmarshaling response, body is in `reply`: %s",
				jsonErr)
	}
	if response.MessageType == "error" {
		return string(bodyBytes), "", 0, 0,
			fmt.Errorf("API error: %s", bodyBytes)
	}

	reply = ""
	// Add responses to conversation
	for _, contentBlock := range *response.Content {
		if contentBlock.Text != nil {
			contentBlock.tokens = response.Usage.OutputTokens
			conversation.AddMessage("assistant",
				&[]ContentBlock{contentBlock})
			// Currently, the API only produces text content
			reply += *contentBlock.Text
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
