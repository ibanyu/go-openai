package openai

import (
	"context"
	"net/http"
)

type ChatCompletionStreamChoiceDelta struct {
	Content          string        `json:"content,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	Role             string        `json:"role,omitempty"`
	FunctionCall     *FunctionCall `json:"function_call,omitempty"`
	ToolCalls        []ToolCall    `json:"tool_calls,omitempty"`
	Refusal          string        `json:"refusal,omitempty"`
	RawExtensions
}

func (r *ChatCompletionStreamChoiceDelta) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionStreamChoiceDelta
	aux := (*alias)(r)

	// 使用优化的反序列化函数
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionStreamChoiceDelta) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionStreamChoiceDelta
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}

	// 使用优化的序列化函数
	return MarshalWithExtensions(temp, r.Extensions)
}

type ChatCompletionStreamChoiceLogprobs struct {
	Content []ChatCompletionTokenLogprob `json:"content,omitempty"`
	Refusal []ChatCompletionTokenLogprob `json:"refusal,omitempty"`
	RawExtensions
}

func (r *ChatCompletionStreamChoiceLogprobs) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionStreamChoiceLogprobs
	aux := (*alias)(r)

	// 使用优化的反序列化函数
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionStreamChoiceLogprobs) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionStreamChoiceLogprobs
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}

	// 使用优化的序列化函数
	return MarshalWithExtensions(temp, r.Extensions)
}

type ChatCompletionTokenLogprob struct {
	Token       string                                 `json:"token"`
	Bytes       []int64                                `json:"bytes,omitempty"`
	Logprob     float64                                `json:"logprob,omitempty"`
	TopLogprobs []ChatCompletionTokenLogprobTopLogprob `json:"top_logprobs"`
	RawExtensions
}

func (r *ChatCompletionTokenLogprob) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionTokenLogprob
	aux := (*alias)(r)
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionTokenLogprob) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionTokenLogprob
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}
	return MarshalWithExtensions(temp, r.Extensions)
}

type ChatCompletionTokenLogprobTopLogprob struct {
	Token   string  `json:"token"`
	Bytes   []int64 `json:"bytes"`
	Logprob float64 `json:"logprob"`
	RawExtensions
}

func (r *ChatCompletionTokenLogprobTopLogprob) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionTokenLogprobTopLogprob
	aux := (*alias)(r)
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionTokenLogprobTopLogprob) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionTokenLogprobTopLogprob
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}
	return MarshalWithExtensions(temp, r.Extensions)
}

type ChatCompletionStreamChoice struct {
	Index                int                                 `json:"index"`
	Delta                ChatCompletionStreamChoiceDelta     `json:"delta"`
	Logprobs             *ChatCompletionStreamChoiceLogprobs `json:"logprobs,omitempty"`
	FinishReason         FinishReason                        `json:"finish_reason"`
	ContentFilterResults ContentFilterResults                `json:"content_filter_results,omitempty"`
	RawExtensions
}

func (r *ChatCompletionStreamChoice) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionStreamChoice
	aux := (*alias)(r)
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionStreamChoice) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionStreamChoice
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}
	return MarshalWithExtensions(temp, r.Extensions)
}

type PromptFilterResult struct {
	Index                int                  `json:"index"`
	ContentFilterResults ContentFilterResults `json:"content_filter_results,omitempty"`
	RawExtensions
}

func (r *PromptFilterResult) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias PromptFilterResult
	aux := (*alias)(r)
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r PromptFilterResult) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias PromptFilterResult
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}
	return MarshalWithExtensions(temp, r.Extensions)
}

type ChatCompletionStreamResponse struct {
	ID                  string                       `json:"id"`
	Object              string                       `json:"object"`
	Created             int64                        `json:"created"`
	Model               string                       `json:"model"`
	Choices             []ChatCompletionStreamChoice `json:"choices"`
	SystemFingerprint   string                       `json:"system_fingerprint"`
	PromptAnnotations   []PromptAnnotation           `json:"prompt_annotations,omitempty"`
	PromptFilterResults []PromptFilterResult         `json:"prompt_filter_results,omitempty"`
	// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request.
	// When present, it contains a null value except for the last chunk which contains the token usage statistics
	// for the entire request.
	Usage *Usage `json:"usage,omitempty"`
	RawExtensions
}

func (r *ChatCompletionStreamResponse) UnmarshalJSON(data []byte) error {
	// 使用类型别名避免递归调用
	type alias ChatCompletionStreamResponse
	aux := (*alias)(r)
	return UnmarshalWithExtensions(data, aux, &r.RawExtensions)
}

func (r ChatCompletionStreamResponse) MarshalJSON() ([]byte, error) {
	// 使用类型别名避免递归调用，同时排除RawExtensions字段
	type alias ChatCompletionStreamResponse
	temp := &struct {
		*alias
		RawExtensions struct{} `json:"-"` // 排除RawExtensions字段
	}{
		alias: (*alias)(&r),
	}
	return MarshalWithExtensions(temp, r.Extensions)
}

// ChatCompletionStream
// Note: Perhaps it is more elegant to abstract Stream using generics.
type ChatCompletionStream struct {
	*streamReader[ChatCompletionStreamResponse]
}

// CreateChatCompletionStream — API call to create a chat completion w/ streaming
// support. It sets whether to stream back partial progress. If set, tokens will be
// sent as data-only server-sent events as they become available, with the
// stream terminated by a data: [DONE] message.
func (c *Client) CreateChatCompletionStream(
	ctx context.Context,
	request ChatCompletionRequest,
) (stream *ChatCompletionStream, err error) {
	urlSuffix := chatCompletionsSuffix
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrChatCompletionInvalidModel
		return
	}

	request.Stream = true
	reasoningValidator := NewReasoningValidator()
	if err = reasoningValidator.Validate(request); err != nil {
		return
	}

	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(urlSuffix, withModel(request.Model)),
		withBody(request),
	)
	if err != nil {
		return nil, err
	}

	resp, err := sendRequestStream[ChatCompletionStreamResponse](c, req)
	if err != nil {
		return
	}
	stream = &ChatCompletionStream{
		streamReader: resp,
	}
	return
}
