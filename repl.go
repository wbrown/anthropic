package anthropic

import (
	"bufio"
	"fmt"
	"os"

	"github.com/wbrown/llmapi"
)

// REPL is a Read-Eval-Print Loop for a conversation. It reads input from
// stdin, sends it to the assistant, and prints the assistant's response.
// It also prints the total tokens used for the conversation, and the tokens
// used for the last message.
//
// It is intended as a simple way to interact with the assistant via a
// console.
func (conversation *Conversation) REPL() {
	print("System:\n", *conversation.System, "\n----\n")
	print("> ")
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		text := scanner.Text()
		conversation.Send(text, llmapi.Sampling{})
		// Get the assistant's last response
		messages := *conversation.Messages
		assistantResponse := *(messages[len(*conversation.Messages)-1])
		for _, contentBlock := range *assistantResponse.Content {
			if contentBlock.Text != nil {
				fmt.Println("----")
				fmt.Print(*contentBlock.Text)
			}
		}
		// print total tokens used for this conversation, and tokens used
		//for the last message
		var inputTokens, outputTokens, totalTokens int
		totalTokens = conversation.Usage.InputTokens +
			conversation.Usage.OutputTokens
		// find the last user message, get tokens
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				inputTokens = (*messages[i].Content)[0].tokens
				break
			}
		}
		outputTokens = (*assistantResponse.Content)[0].tokens
		fmt.Printf("\n---- Input: %d, Output: %d, [Total Usage: %d]\n",
			inputTokens,
			outputTokens,
			totalTokens)
		print("> ")
	}
}

func main() {
	systemPrompt := `
You are a helpful and friendly assistant who replies to requests accurately.
You are a good listener and can provide information on a wide range of topics.
You are knowledgeable and can provide helpful advice.`
	// Do a simple conversation via REPL
	conversation := NewConversation(systemPrompt)
	conversation.REPL()
}
