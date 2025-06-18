package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ibanyu/go-openai"
	"github.com/ibanyu/go-openai/internal/test/checks"
	"github.com/ibanyu/go-openai/jsonschema"
)

const (
	xCustomHeader      = "X-CUSTOM-HEADER"
	xCustomHeaderValue = "test"
)

var rateLimitHeaders = map[string]any{
	"x-ratelimit-limit-requests":     60,
	"x-ratelimit-limit-tokens":       150000,
	"x-ratelimit-remaining-requests": 59,
	"x-ratelimit-remaining-tokens":   149984,
	"x-ratelimit-reset-requests":     "1s",
	"x-ratelimit-reset-tokens":       "6m0s",
}

func TestChatCompletionsWrongModel(t *testing.T) {
	config := openai.DefaultConfig("whatever")
	config.BaseURL = "http://localhost/v1"
	client := openai.NewClientWithConfig(config)
	ctx := context.Background()

	req := openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     "ada",
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	}
	_, err := client.CreateChatCompletion(ctx, req)
	msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
	checks.ErrorIs(t, err, openai.ErrChatCompletionInvalidModel, msg)
}

func TestO1ModelsChatCompletionsDeprecatedFields(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "o1-preview_MaxTokens_deprecated",
			in: openai.ChatCompletionRequest{
				MaxTokens: 5,
				Model:     openai.O1Preview,
			},
			expectedError: openai.ErrReasoningModelMaxTokensDeprecated,
		},
		{
			name: "o1-mini_MaxTokens_deprecated",
			in: openai.ChatCompletionRequest{
				MaxTokens: 5,
				Model:     openai.O1Mini,
			},
			expectedError: openai.ErrReasoningModelMaxTokensDeprecated,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestO1ModelsChatCompletionsBetaLimitations(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "log_probs_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				LogProbs:            true,
				Model:               openai.O1Preview,
			},
			expectedError: openai.ErrReasoningModelLimitationsLogprobs,
		},
		{
			name: "set_temperature_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(2),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_top_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_n_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(1),
				N:           2,
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_presence_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				PresencePenalty: float32(1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_frequency_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				FrequencyPenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestO3ModelsChatCompletionsBetaLimitations(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "log_probs_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				LogProbs:            true,
				Model:               openai.O3Mini,
			},
			expectedError: openai.ErrReasoningModelLimitationsLogprobs,
		},
		{
			name: "set_temperature_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(2),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_top_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_n_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(1),
				N:           2,
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_presence_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				PresencePenalty: float32(1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_frequency_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				FrequencyPenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestChatRequestOmitEmpty(t *testing.T) {
	data, err := json.Marshal(openai.ChatCompletionRequest{
		// We set model b/c it's required, so omitempty doesn't make sense
		Model: "gpt-4",
	})
	checks.NoError(t, err)

	// messages is also required so isn't omitted
	const expected = `{"model":"gpt-4","messages":null}`
	if string(data) != expected {
		t.Errorf("expected JSON with all empty fields to be %v but was %v", expected, string(data))
	}
}

func TestChatCompletionsWithStream(t *testing.T) {
	config := openai.DefaultConfig("whatever")
	config.BaseURL = "http://localhost/v1"
	client := openai.NewClientWithConfig(config)
	ctx := context.Background()

	req := openai.ChatCompletionRequest{
		Stream: true,
	}
	_, err := client.CreateChatCompletion(ctx, req)
	checks.ErrorIs(t, err, openai.ErrChatCompletionStreamNotSupported, "unexpected error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestO1ModelChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:               openai.O1Preview,
		MaxCompletionTokens: 1000,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

func TestO3ModelChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:               openai.O3Mini,
		MaxCompletionTokens: 1000,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestChatCompletionsWithHeaders(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")

	a := resp.Header().Get(xCustomHeader)
	_ = a
	if resp.Header().Get(xCustomHeader) != xCustomHeaderValue {
		t.Errorf("expected header %s to be %s", xCustomHeader, xCustomHeaderValue)
	}
}

// TestChatCompletionsWithRateLimitHeaders Tests the completions endpoint of the API using the mocked server.
func TestChatCompletionsWithRateLimitHeaders(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")

	headers := resp.GetRateLimitHeaders()
	resetRequests := headers.ResetRequests.String()
	if resetRequests != rateLimitHeaders["x-ratelimit-reset-requests"] {
		t.Errorf("expected resetRequests %s to be %s", resetRequests, rateLimitHeaders["x-ratelimit-reset-requests"])
	}
	resetRequestsTime := headers.ResetRequests.Time()
	if resetRequestsTime.Before(time.Now()) {
		t.Errorf("unexpected reset requests: %v", resetRequestsTime)
	}

	bs1, _ := json.Marshal(headers)
	bs2, _ := json.Marshal(rateLimitHeaders)
	if string(bs1) != string(bs2) {
		t.Errorf("expected rate limit header %s to be %s", bs2, bs1)
	}
}

// TestChatCompletionsFunctions tests including a function call.
func TestChatCompletionsFunctions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	t.Run("bytes", func(t *testing.T) {
		//nolint:lll
		msg := json.RawMessage(`{"properties":{"count":{"type":"integer","description":"total number of words in sentence"},"words":{"items":{"type":"string"},"type":"array","description":"list of words in sentence"}},"type":"object","required":["count","words"]}`)
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("struct", func(t *testing.T) {
		type testMessage struct {
			Count int      `json:"count"`
			Words []string `json:"words"`
		}
		msg := testMessage{
			Count: 2,
			Words: []string{"hello", "world"},
		}
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinition", func(t *testing.T) {
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinitionWithFunctionDefine", func(t *testing.T) {
		// this is a compatibility check
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefine{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("StructuredOutputs", func(t *testing.T) {
		type testMessage struct {
			Count int      `json:"count"`
			Words []string `json:"words"`
		}
		msg := testMessage{
			Count: 2,
			Words: []string{"hello", "world"},
		}
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Strict:     true,
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
}

func TestAzureChatCompletions(t *testing.T) {
	client, server, teardown := setupAzureTestServer()
	defer teardown()
	server.RegisterHandler("/openai/deployments/*", handleChatCompletionEndpoint)

	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateAzureChatCompletion error")
}

func TestMultipartChatCompletions(t *testing.T) {
	client, server, teardown := setupAzureTestServer()
	defer teardown()
	server.RegisterHandler("/openai/deployments/*", handleChatCompletionEndpoint)

	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role: openai.ChatMessageRoleUser,
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: "Hello!",
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL:    "URL",
							Detail: openai.ImageURLDetailLow,
						},
					},
				},
			},
		},
	})
	checks.NoError(t, err, "CreateAzureChatCompletion error")
}

func TestMultipartChatMessageSerialization(t *testing.T) {
	jsonText := `[{"role":"system","content":"system-message"},` +
		`{"role":"user","content":[{"type":"text","text":"nice-text"},` +
		`{"type":"image_url","image_url":{"url":"URL","detail":"high"}}]}]`

	var msgs []openai.ChatCompletionMessage
	err := json.Unmarshal([]byte(jsonText), &msgs)
	if err != nil {
		t.Fatalf("Expected no error: %s", err)
	}
	if len(msgs) != 2 {
		t.Errorf("unexpected number of messages")
	}
	if msgs[0].Role != "system" || msgs[0].Content != "system-message" || msgs[0].MultiContent != nil {
		t.Errorf("invalid user message: %v", msgs[0])
	}
	if msgs[1].Role != "user" || msgs[1].Content != "" || len(msgs[1].MultiContent) != 2 {
		t.Errorf("invalid user message")
	}
	parts := msgs[1].MultiContent
	if parts[0].Type != "text" || parts[0].Text != "nice-text" {
		t.Errorf("invalid text part: %v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL.URL != "URL" || parts[1].ImageURL.Detail != "high" {
		t.Errorf("invalid image_url part")
	}

	s, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("Expected no error: %s", err)
	}
	res := strings.ReplaceAll(string(s), " ", "")
	if res != jsonText {
		t.Fatalf("invalid message: %s", string(s))
	}

	invalidMsg := []openai.ChatCompletionMessage{
		{
			Role:    "user",
			Content: "some-text",
			MultiContent: []openai.ChatMessagePart{
				{
					Type: "text",
					Text: "nice-text",
				},
			},
		},
	}
	_, err = json.Marshal(invalidMsg)
	if !errors.Is(err, openai.ErrContentFieldsMisused) {
		t.Fatalf("Expected error: %s", err)
	}

	err = json.Unmarshal([]byte(`["not-a-message"]`), &msgs)
	if err == nil {
		t.Fatalf("Expected error")
	}

	emptyMultiContentMsg := openai.ChatCompletionMessage{
		Role:         "user",
		MultiContent: []openai.ChatMessagePart{},
	}
	s, err = json.Marshal(emptyMultiContentMsg)
	if err != nil {
		t.Fatalf("Unexpected error")
	}
	res = strings.ReplaceAll(string(s), " ", "")
	if res != `{"role":"user"}` {
		t.Fatalf("invalid message: %s", string(s))
	}
}

// handleChatCompletionEndpoint Handles the ChatGPT completion endpoint by the test server.
func handleChatCompletionEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error
	var resBytes []byte

	// completions only accepts POST requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
	var completionReq openai.ChatCompletionRequest
	if completionReq, err = getChatCompletionBody(r); err != nil {
		http.Error(w, "could not read request", http.StatusInternalServerError)
		return
	}
	res := openai.ChatCompletionResponse{
		ID:      strconv.Itoa(int(time.Now().Unix())),
		Object:  "test-object",
		Created: time.Now().Unix(),
		// would be nice to validate Model during testing, but
		// this may not be possible with how much upkeep
		// would be required / wouldn't make much sense
		Model: completionReq.Model,
	}
	// create completions
	n := completionReq.N
	if n == 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		// if there are functions, include them
		if len(completionReq.Functions) > 0 {
			var fcb []byte
			b := completionReq.Functions[0].Parameters
			fcb, err = json.Marshal(b)
			if err != nil {
				http.Error(w, "could not marshal function parameters", http.StatusInternalServerError)
				return
			}

			res.Choices = append(res.Choices, openai.ChatCompletionChoice{
				Message: openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleFunction,
					// this is valid json so it should be fine
					FunctionCall: &openai.FunctionCall{
						Name:      completionReq.Functions[0].Name,
						Arguments: string(fcb),
					},
				},
				Index: i,
			})
			continue
		}
		// generate a random string of length completionReq.Length
		completionStr := strings.Repeat("a", completionReq.MaxTokens)

		res.Choices = append(res.Choices, openai.ChatCompletionChoice{
			Message: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: completionStr,
			},
			Index: i,
		})
	}
	inputTokens := numTokens(completionReq.Messages[0].Content) * n
	completionTokens := completionReq.MaxTokens * n
	res.Usage = openai.Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      inputTokens + completionTokens,
	}
	resBytes, _ = json.Marshal(res)
	w.Header().Set(xCustomHeader, xCustomHeaderValue)
	for k, v := range rateLimitHeaders {
		switch val := v.(type) {
		case int:
			w.Header().Set(k, strconv.Itoa(val))
		default:
			w.Header().Set(k, fmt.Sprintf("%s", v))
		}
	}
	fmt.Fprintln(w, string(resBytes))
}

// getChatCompletionBody Returns the body of the request to create a completion.
func getChatCompletionBody(r *http.Request) (openai.ChatCompletionRequest, error) {
	completion := openai.ChatCompletionRequest{}
	// read the request body
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		return openai.ChatCompletionRequest{}, err
	}
	err = json.Unmarshal(reqBody, &completion)
	if err != nil {
		return openai.ChatCompletionRequest{}, err
	}
	return completion, nil
}

func TestFinishReason(t *testing.T) {
	c := &openai.ChatCompletionChoice{
		FinishReason: openai.FinishReasonNull,
	}
	resBytes, _ := json.Marshal(c)
	if !strings.Contains(string(resBytes), `"finish_reason":null`) {
		t.Error("null should not be quoted")
	}

	c.FinishReason = ""

	resBytes, _ = json.Marshal(c)
	if !strings.Contains(string(resBytes), `"finish_reason":null`) {
		t.Error("null should not be quoted")
	}

	otherReasons := []openai.FinishReason{
		openai.FinishReasonStop,
		openai.FinishReasonLength,
		openai.FinishReasonFunctionCall,
		openai.FinishReasonContentFilter,
	}
	for _, r := range otherReasons {
		c.FinishReason = r
		resBytes, _ = json.Marshal(c)
		if !strings.Contains(string(resBytes), fmt.Sprintf(`"finish_reason":"%s"`, r)) {
			t.Errorf("%s should be quoted", r)
		}
	}
}

// TestChatMessagePartRawExtensions 测试ChatMessagePart的RawExtensions功能
func TestChatMessagePartRawExtensions(t *testing.T) {
	t.Run("序列化带扩展字段的ChatMessagePart", func(t *testing.T) {
		part := openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: "Hello world",
		}

		// 添加扩展字段
		part.SetExtension("custom_metadata", map[string]interface{}{
			"source":    "user_input",
			"priority":  "high",
			"timestamp": "2024-01-15T10:30:00Z",
		})
		part.SetExtension("processing_hint", "use_careful_analysis")

		// 序列化
		jsonData, err := json.Marshal(part)
		if err != nil {
			t.Fatalf("序列化失败: %v", err)
		}

		// 验证JSON包含扩展字段
		jsonStr := string(jsonData)
		if !strings.Contains(jsonStr, "custom_metadata") {
			t.Error("序列化结果缺少custom_metadata字段")
		}
		if !strings.Contains(jsonStr, "processing_hint") {
			t.Error("序列化结果缺少processing_hint字段")
		}
		if !strings.Contains(jsonStr, "use_careful_analysis") {
			t.Error("序列化结果缺少扩展字段值")
		}
	})

	t.Run("反序列化包含扩展字段的JSON", func(t *testing.T) {
		jsonWithExtensions := `{
			"type": "image_url",
			"image_url": {
				"url": "https://example.com/image.jpg",
				"detail": "high"
			},
			"custom_style": {
				"filter": "vintage",
				"enhancement": "brightness"
			},
			"usage_tracking": {
				"request_id": "req_123456",
				"user_id": "user_789"
			}
		}`

		var part openai.ChatMessagePart
		err := json.Unmarshal([]byte(jsonWithExtensions), &part)
		if err != nil {
			t.Fatalf("反序列化失败: %v", err)
		}

		// 验证基本字段
		if part.Type != openai.ChatMessagePartTypeImageURL {
			t.Errorf("期望Type为%s，实际为%s", openai.ChatMessagePartTypeImageURL, part.Type)
		}
		if part.ImageURL == nil || part.ImageURL.URL != "https://example.com/image.jpg" {
			t.Error("ImageURL字段解析错误")
		}

		// 验证扩展字段
		if customStyle, exists := part.GetExtension("custom_style"); !exists {
			t.Error("扩展字段custom_style丢失")
		} else {
			styleMap, ok := customStyle.(map[string]interface{})
			if !ok {
				t.Error("custom_style字段类型错误")
			} else if styleMap["filter"] != "vintage" {
				t.Error("custom_style.filter值错误")
			}
		}

		if usageTracking, exists := part.GetExtension("usage_tracking"); !exists {
			t.Error("扩展字段usage_tracking丢失")
		} else {
			trackingMap, ok := usageTracking.(map[string]interface{})
			if !ok {
				t.Error("usage_tracking字段类型错误")
			} else if trackingMap["request_id"] != "req_123456" {
				t.Error("usage_tracking.request_id值错误")
			}
		}

		// 验证原始数据保存
		rawData := part.GetRawData()
		if len(rawData) == 0 {
			t.Error("原始JSON数据未保存")
		}

		extensionRawData := part.GetExtensionRawData()
		if len(extensionRawData) == 0 {
			t.Error("扩展字段原始数据未保存")
		}
	})

	t.Run("ChatMessagePart扩展字段管理", func(t *testing.T) {
		part := openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: "Test message",
		}

		// 测试设置和获取扩展字段
		part.SetExtension("test_key", "test_value")
		if value, exists := part.GetExtension("test_key"); !exists || value != "test_value" {
			t.Error("扩展字段设置或获取失败")
		}

		// 测试获取所有扩展字段
		part.SetExtension("another_key", 123)
		extensions := part.GetExtensions()
		if len(*extensions) != 2 {
			t.Errorf("期望2个扩展字段，实际有%d个", len(*extensions))
		}

		// 测试获取特定扩展字段
		if value, exists := part.GetExtension("another_key"); !exists || value.(int) != 123 {
			t.Error("扩展字段another_key获取失败")
		}
	})
}

// TestChatCompletionMessageOptimized 测试优化后的ChatCompletionMessage序列化/反序列化
func TestChatCompletionMessageOptimized(t *testing.T) {
	t.Run("单内容消息序列化优化", func(t *testing.T) {
		msg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "Hello, world!",
		}
		msg.SetExtension("message_id", "msg_12345")
		msg.SetExtension("priority", "high")

		// 序列化
		jsonData, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("序列化失败: %v", err)
		}

		jsonStr := string(jsonData)

		// 验证不包含MultiContent字段
		if strings.Contains(jsonStr, "MultiContent") {
			t.Error("单内容消息不应包含MultiContent字段")
		}

		// 验证包含扩展字段
		if !strings.Contains(jsonStr, "message_id") {
			t.Error("缺少扩展字段message_id")
		}

		// 反序列化验证
		var parsed openai.ChatCompletionMessage
		err = json.Unmarshal(jsonData, &parsed)
		if err != nil {
			t.Fatalf("反序列化失败: %v", err)
		}

		if parsed.Content != "Hello, world!" {
			t.Error("Content字段解析错误")
		}
		if parsed.MultiContent != nil && len(parsed.MultiContent) > 0 {
			t.Error("单内容消息不应有MultiContent")
		}

		// 验证扩展字段
		if msgID, exists := parsed.GetExtension("message_id"); !exists || msgID != "msg_12345" {
			t.Error("扩展字段message_id丢失或值错误")
		}
	})

	t.Run("多内容消息序列化优化", func(t *testing.T) {
		msg := openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleUser,
			MultiContent: []openai.ChatMessagePart{
				{
					Type: openai.ChatMessagePartTypeText,
					Text: "请分析这张图片",
				},
				{
					Type: openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{
						URL:    "https://example.com/image.jpg",
						Detail: openai.ImageURLDetailHigh,
					},
				},
			},
		}
		msg.SetExtension("conversation_id", "conv_67890")

		// 序列化
		jsonData, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("序列化失败: %v", err)
		}

		jsonStr := string(jsonData)

		// 验证content是数组格式
		if !strings.Contains(jsonStr, `"content":[`) {
			t.Error("多内容消息的content应为数组格式")
		}

		// 验证不包含单独的Content字段（应该被排除）
		if strings.Contains(jsonStr, `"content":"`) {
			t.Error("多内容消息不应包含字符串形式的content字段")
		}

		// 反序列化验证
		var parsed openai.ChatCompletionMessage
		err = json.Unmarshal(jsonData, &parsed)
		if err != nil {
			t.Fatalf("反序列化失败: %v", err)
		}

		if parsed.Content != "" {
			t.Error("多内容消息的Content字段应为空")
		}
		if len(parsed.MultiContent) != 2 {
			t.Errorf("期望2个内容部分，实际有%d个", len(parsed.MultiContent))
		}

		// 验证第一个部分
		if parsed.MultiContent[0].Type != openai.ChatMessagePartTypeText {
			t.Error("第一个部分类型错误")
		}
		if parsed.MultiContent[0].Text != "请分析这张图片" {
			t.Error("第一个部分文本错误")
		}

		// 验证第二个部分
		if parsed.MultiContent[1].Type != openai.ChatMessagePartTypeImageURL {
			t.Error("第二个部分类型错误")
		}
		if parsed.MultiContent[1].ImageURL.URL != "https://example.com/image.jpg" {
			t.Error("第二个部分URL错误")
		}
	})

	t.Run("智能反序列化类型检测", func(t *testing.T) {
		// 测试字符串content的检测
		singleContentJSON := `{
			"role": "user",
			"content": "这是一个字符串内容",
			"custom_field": "test_value"
		}`

		var msg1 openai.ChatCompletionMessage
		err := json.Unmarshal([]byte(singleContentJSON), &msg1)
		if err != nil {
			t.Fatalf("单内容反序列化失败: %v", err)
		}

		if msg1.Content != "这是一个字符串内容" {
			t.Error("单内容解析错误")
		}
		if len(msg1.MultiContent) > 0 {
			t.Error("单内容消息不应有MultiContent")
		}

		// 测试数组content的检测
		multiContentJSON := `{
			"role": "user",
			"content": [
				{"type": "text", "text": "请看这张图"},
				{"type": "image_url", "image_url": {"url": "test.jpg"}}
			],
			"custom_field": "test_value"
		}`

		var msg2 openai.ChatCompletionMessage
		err = json.Unmarshal([]byte(multiContentJSON), &msg2)
		if err != nil {
			t.Fatalf("多内容反序列化失败: %v", err)
		}

		if msg2.Content != "" {
			t.Error("多内容消息Content应为空")
		}
		if len(msg2.MultiContent) != 2 {
			t.Errorf("期望2个内容部分，实际有%d个", len(msg2.MultiContent))
		}
	})

	t.Run("多内容消息中的ChatMessagePart扩展", func(t *testing.T) {
		// 创建带扩展字段的ChatMessagePart
		partWithExt := openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: "带扩展的文本",
		}
		partWithExt.SetExtension("part_id", "part_001")
		partWithExt.SetExtension("metadata", map[string]interface{}{
			"source": "enhanced_input",
			"score":  0.95,
		})

		msg := openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleUser,
			MultiContent: []openai.ChatMessagePart{
				partWithExt,
				{
					Type: openai.ChatMessagePartTypeText,
					Text: "普通文本",
				},
			},
		}

		// 序列化
		jsonData, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("序列化失败: %v", err)
		}

		// 反序列化
		var parsed openai.ChatCompletionMessage
		err = json.Unmarshal(jsonData, &parsed)
		if err != nil {
			t.Fatalf("反序列化失败: %v", err)
		}

		// 验证扩展字段保留
		firstPart := parsed.MultiContent[0]
		if partID, exists := firstPart.GetExtension("part_id"); !exists || partID != "part_001" {
			t.Error("ChatMessagePart的扩展字段丢失")
		}

		if metadata, exists := firstPart.GetExtension("metadata"); !exists {
			t.Error("ChatMessagePart的复杂扩展字段丢失")
		} else {
			metaMap, ok := metadata.(map[string]interface{})
			if !ok {
				t.Error("扩展字段类型错误")
			} else if metaMap["score"].(float64) != 0.95 {
				t.Error("扩展字段值错误")
			}
		}
	})

	t.Run("错误情况处理", func(t *testing.T) {
		// 测试Content和MultiContent同时设置的错误
		invalidMsg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "不能同时设置",
			MultiContent: []openai.ChatMessagePart{
				{Type: openai.ChatMessagePartTypeText, Text: "这个"},
			},
		}

		_, err := json.Marshal(invalidMsg)
		if !errors.Is(err, openai.ErrContentFieldsMisused) {
			t.Errorf("期望ErrContentFieldsMisused错误，实际得到: %v", err)
		}
	})
}

// TestChatCompletionMessagePerformance 测试优化后的性能改进
func TestChatCompletionMessagePerformance(t *testing.T) {
	t.Run("序列化性能测试", func(t *testing.T) {
		// 准备测试数据
		singleMsg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "Hello world",
		}
		singleMsg.SetExtension("test", "value")

		multiMsg := openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleUser,
			MultiContent: []openai.ChatMessagePart{
				{Type: openai.ChatMessagePartTypeText, Text: "Hello"},
				{Type: openai.ChatMessagePartTypeText, Text: "World"},
			},
		}

		// 性能测试：单内容消息
		t.Run("单内容消息", func(t *testing.T) {
			for i := 0; i < 1000; i++ {
				_, err := json.Marshal(singleMsg)
				if err != nil {
					t.Fatalf("序列化失败: %v", err)
				}
			}
		})

		// 性能测试：多内容消息
		t.Run("多内容消息", func(t *testing.T) {
			for i := 0; i < 1000; i++ {
				_, err := json.Marshal(multiMsg)
				if err != nil {
					t.Fatalf("序列化失败: %v", err)
				}
			}
		})
	})

	t.Run("反序列化性能测试", func(t *testing.T) {
		singleJSON := `{"role":"user","content":"Hello world","test":"value"}`
		multiJSON := `{"role":"user","content":[{"type":"text","text":"Hello"},{"type":"text","text":"World"}]}`

		// 性能测试：单内容反序列化
		t.Run("单内容反序列化", func(t *testing.T) {
			for i := 0; i < 1000; i++ {
				var msg openai.ChatCompletionMessage
				err := json.Unmarshal([]byte(singleJSON), &msg)
				if err != nil {
					t.Fatalf("反序列化失败: %v", err)
				}
			}
		})

		// 性能测试：多内容反序列化
		t.Run("多内容反序列化", func(t *testing.T) {
			for i := 0; i < 1000; i++ {
				var msg openai.ChatCompletionMessage
				err := json.Unmarshal([]byte(multiJSON), &msg)
				if err != nil {
					t.Fatalf("反序列化失败: %v", err)
				}
			}
		})
	})
}

// TestChatMessageRoundTrip 测试完整的序列化/反序列化往返
func TestChatMessageRoundTrip(t *testing.T) {
	testCases := []struct {
		name string
		msg  openai.ChatCompletionMessage
	}{
		{
			name: "单内容消息",
			msg: openai.ChatCompletionMessage{
				Role:             openai.ChatMessageRoleUser,
				Content:          "Hello world",
				ReasoningContent: "思考过程",
				Name:             "test_user",
			},
		},
		{
			name: "多内容消息",
			msg: openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleUser,
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: "分析图片",
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL:    "https://example.com/test.jpg",
							Detail: openai.ImageURLDetailHigh,
						},
					},
				},
			},
		},
		{
			name: "带工具调用的消息",
			msg: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: "我来调用工具",
				ToolCalls: []openai.ToolCall{
					{
						ID:   "call_123",
						Type: openai.ToolTypeFunction,
						Function: openai.FunctionCall{
							Name:      "test_function",
							Arguments: `{"param": "value"}`,
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 添加扩展字段
			tc.msg.SetExtension("test_id", "round_trip_test")
			tc.msg.SetExtension("metadata", map[string]interface{}{
				"version": 1.0,
				"flags":   []string{"test", "roundtrip"},
			})

			// 序列化
			jsonData, err := json.Marshal(tc.msg)
			if err != nil {
				t.Fatalf("序列化失败: %v", err)
			}

			// 反序列化
			var parsed openai.ChatCompletionMessage
			err = json.Unmarshal(jsonData, &parsed)
			if err != nil {
				t.Fatalf("反序列化失败: %v", err)
			}

			// 验证基本字段
			if parsed.Role != tc.msg.Role {
				t.Errorf("Role不匹配: 期望%s, 实际%s", tc.msg.Role, parsed.Role)
			}

			// 验证内容字段
			if tc.msg.Content != "" {
				if parsed.Content != tc.msg.Content {
					t.Errorf("Content不匹配: 期望%s, 实际%s", tc.msg.Content, parsed.Content)
				}
			}

			if len(tc.msg.MultiContent) > 0 {
				if len(parsed.MultiContent) != len(tc.msg.MultiContent) {
					t.Errorf("MultiContent长度不匹配: 期望%d, 实际%d",
						len(tc.msg.MultiContent), len(parsed.MultiContent))
				}
			}

			// 验证扩展字段
			if testID, exists := parsed.GetExtension("test_id"); !exists || testID != "round_trip_test" {
				t.Error("扩展字段test_id丢失或值错误")
			}

			if metadata, exists := parsed.GetExtension("metadata"); !exists {
				t.Error("扩展字段metadata丢失")
			} else {
				metaMap, ok := metadata.(map[string]interface{})
				if !ok {
					t.Error("metadata字段类型错误")
				} else if metaMap["version"].(float64) != 1.0 {
					t.Error("metadata.version值错误")
				}
			}

			// 验证原始数据保存
			rawData := parsed.GetRawData()
			if len(rawData) == 0 {
				t.Error("原始JSON数据未保存")
			}
		})
	}
}
