package openai

import (
	"encoding/json"
	"fmt"
)

type Extensions map[string]interface{}

type Extender interface {
	GetExtensions() *Extensions
}

// RawExtensions 增强版扩展机制，支持保存原始字节数据
type RawExtensions struct {
	// 存储扩展字段
	Extensions map[string]interface{} `json:"-"`
	// 存储原始JSON字节数据
	RawData []byte `json:"-"`
	// 存储只包含扩展字段的原始字节数据
	ExtensionRawData []byte `json:"-"`
}

// SetExtension 设置扩展字段
func (r *RawExtensions) SetExtension(key string, value interface{}) {
	if r.Extensions == nil {
		r.Extensions = make(map[string]interface{})
	}
	r.Extensions[key] = value
}

// GetExtension 获取扩展字段
func (r *RawExtensions) GetExtension(key string) (interface{}, bool) {
	if r.Extensions == nil {
		return nil, false
	}
	value, exists := r.Extensions[key]
	return value, exists
}

// GetExtensions 兼容原有接口
func (r *RawExtensions) GetExtensions() *Extensions {
	if r.Extensions == nil {
		r.Extensions = make(map[string]interface{})
	}
	ext := Extensions(r.Extensions)
	return &ext
}

// SetRawData 设置原始数据
func (r *RawExtensions) SetRawData(data []byte) {
	r.RawData = make([]byte, len(data))
	copy(r.RawData, data)
}

// GetRawData 获取原始数据
func (r *RawExtensions) GetRawData() []byte {
	return r.RawData
}

// GetExtensionRawData 获取扩展字段的原始数据
func (r *RawExtensions) GetExtensionRawData() []byte {
	return r.ExtensionRawData
}

// UnmarshalWithExtensions 优化版反序列化函数，使用方案1
func UnmarshalWithExtensions(data []byte, target interface{}, extensions *RawExtensions) error {
	// 保存原始数据
	extensions.SetRawData(data)

	// 先解析到target结构体
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("failed to unmarshal target: %w", err)
	}

	// 解析到map以获取所有字段
	var allFields map[string]interface{}
	if err := json.Unmarshal(data, &allFields); err != nil {
		return fmt.Errorf("failed to unmarshal to map: %w", err)
	}

	// 获取target结构体的已知字段
	knownFields := getKnownFields(target)

	// 分离扩展字段
	extensionFields := make(map[string]interface{})
	for key, value := range allFields {
		if !knownFields[key] {
			extensionFields[key] = value
		}
	}

	// 保存扩展字段
	if len(extensionFields) > 0 {
		extensions.Extensions = extensionFields
		// 保存扩展字段的原始字节
		if extensionData, err := json.Marshal(extensionFields); err == nil {
			extensions.ExtensionRawData = extensionData
		}
	}

	return nil
}

// getKnownFields 通过反序列化获取结构体的已知字段
func getKnownFields(target interface{}) map[string]bool {
	knownFields := make(map[string]bool)

	// 将target序列化再反序列化到map，以获取JSON字段名
	targetBytes, err := json.Marshal(target)
	if err != nil {
		return knownFields
	}

	var targetMap map[string]interface{}
	if err := json.Unmarshal(targetBytes, &targetMap); err != nil {
		return knownFields
	}

	for key := range targetMap {
		knownFields[key] = true
	}

	return knownFields
}

// MarshalWithExtensions 优化版序列化函数，使用方案1
func MarshalWithExtensions(target interface{}, extensions map[string]interface{}) ([]byte, error) {
	// 使用类型别名和字段排除的方式序列化基础字段
	baseData, err := json.Marshal(target)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal target: %w", err)
	}

	// 如果没有扩展字段，直接返回
	if len(extensions) == 0 {
		return baseData, nil
	}

	// 合并扩展字段
	var baseMap map[string]interface{}
	if err := json.Unmarshal(baseData, &baseMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal base data: %w", err)
	}

	// 添加扩展字段
	for key, value := range extensions {
		baseMap[key] = value
	}

	return json.Marshal(baseMap)
}

// 保持向后兼容性
func UnmarshalJSON(data []byte, t ...any) error {
	for _, v := range t {
		if err := json.Unmarshal(data, v); err != nil {
			return err
		}
	}
	return nil
}
