# anthropic
golang interface for [Anthropic's](https://anthropic.com) Machine Learning
inference HTTP API interface, specifically the
[Messaging API](https://docs.anthropic.com/en/api/messages).
Colloquially, this is the API for the Claude family of Machine Learning
model such as [Claude Sonnet 3.5](http://claude.ai).

It presents a `Conversation` struct that can be used to interact with the
API.  The API key is read from the environment variable `ANTHROPIC_API_KEY`.

The `Conversation` struct has a `Send` method that takes a string input and
returns a string reply, a string stop reason, input tokens, output tokens,
and an error.  The stop reason is a string that indicates why the conversation
stopped.  The input and output tokens are the tokens that were used for the
input and output strings, respectively.

More interestingly, the `Conversation` struct contains the history of the
conversation, so you can continue a conversation by calling `Send` with a new
input string.

There is an `anthropic.DefaultSettings` struct that can be used to set the
defaults for the `Conversation` struct.

You may also set the default API token for the `Conversation` struct by
setting `anthropic.DefaultApiToken`. This can be useful if you have multiple
`Conversation`s.

If you want to set the API key on a per `Conversation` basis, you can set the
`ApiToken` field of the `Conversation` struct once it is created using
`NewConversation`.

The `anthropic.DefaultSettings` struct has the following fields:

```golang
var DefaultSettings = SampleSettings{
	Model:       "claude-3-5-sonnet-20240620",
	Version:     "2023-06-01",
	Beta:        "", // "max-tokens-3-5-sonnet-2024-07-15"
	MaxTokens:   4096,
	Temperature: 0.0,
}
```

## Installation
```bash
go get github.com/wbrown/anthropic
```

## Usage
```go
package main

import (
    "fmt"
    "log"

    "github.com/wbrown/anthropic"
)

func main() {
    // API key is read from the environment variable ANTHROPIC_API_KEY
    // You can also set the API key by setting conversation.Settings.ApiToken
    conversation := anthropic.NewConversation("You are a friendly chatbot.")
	reply, stopReason, inputTokens, outputTokens, err :=
		conversation.Send("Hello Claude!")
	if err != nil {
		log.Fatal(err)
	}
    fmt.Println("Reply:", reply)
    fmt.Println("Stop Reason:", stopReason)
    fmt.Println("Input Tokens:", inputTokens)
    fmt.Println("Output Tokens:", outputTokens)

    reply, _, _ ,_, err = conversation.Send("How are you?")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Reply:", reply)
}
```