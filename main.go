package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/bits"
	"net/http"
	"net/http/cookiejar"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
)

const (
	deepseekPowURL  = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge"
	deepseekChatURL = "https://chat.deepseek.com/api/v0/chat/completion"
)

var httpClient *http.Client

func init() {
	jar, _ := cookiejar.New(nil)
	httpClient = &http.Client{Timeout: 120 * time.Second}
	httpClient.Jar = jar
}

var RC = [24]uint64{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
	0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
	0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
	0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
	0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
	0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
}

func keccakF1600_23(state *[25]uint64) {
	var C, D [5]uint64

	for r := 1; r < 24; r++ {
		for x := 0; x < 5; x++ {
			C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20]
		}
		for x := 0; x < 5; x++ {
			D[x] = C[(x+4)%5] ^ bits.RotateLeft64(C[(x+1)%5], 1)
		}
		for i := 0; i < 25; i++ {
			state[i] ^= D[i%5]
		}

		x, y := 1, 0
		current := state[1]
		for t := 0; t < 24; t++ {
			nextX := y
			nextY := (2*x + 3*y) % 5
			shift := ((t + 1) * (t + 2) / 2) % 64
			temp := state[nextX+5*nextY]
			state[nextX+5*nextY] = bits.RotateLeft64(current, int(shift))
			current = temp
			x, y = nextX, nextY
		}

		for j := 0; j < 25; j += 5 {
			for i := 0; i < 5; i++ {
				C[i] = state[j+i]
			}
			for i := 0; i < 5; i++ {
				state[j+i] ^= (^C[(i+1)%5]) & C[(i+2)%5]
			}
		}

		state[0] ^= RC[r]
	}
}

func customSHA3_256(data []byte) [32]byte {
	var state [25]uint64
	rate := 136
	offset := 0
	inlen := len(data)

	for inlen >= rate {
		for i := 0; i < rate/8; i++ {
			state[i] ^= binary.LittleEndian.Uint64(data[offset+i*8:])
		}
		keccakF1600_23(&state)
		offset += rate
		inlen -= rate
	}

	var block [136]byte
	copy(block[:], data[offset:])
	block[inlen] ^= 0x06
	block[rate-1] ^= 0x80

	for i := 0; i < rate/8; i++ {
		state[i] ^= binary.LittleEndian.Uint64(block[i*8:])
	}
	keccakF1600_23(&state)

	var out [32]byte
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(out[i*8:], state[i])
	}
	return out
}

var modelMap = map[string]map[string]interface{}{
	"deepseek-chat": {
		"model":            "deepseek-chat",
		"thinking_enabled": false,
		"search_enabled":   false,
	},
	"deepseek-chat-reasoning": {
		"model":            "deepseek-chat",
		"thinking_enabled": true,
		"search_enabled":   false,
	},
	"deepseek-searcher": {
		"model":            "deepseek-chat",
		"thinking_enabled": false,
		"search_enabled":   true,
	},
	"default": {
		"model":            "deepseek-chat",
		"thinking_enabled": false,
		"search_enabled":   false,
	},
}

type OpenAIRequest struct {
	Model      string    `json:"model"`
	Messages   []Message `json:"messages"`
	Stream     bool      `json:"stream"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice any       `json:"tool_choice,omitempty"`
}

type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type ToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	Index         int            `json:"index"`
	Message       Message        `json:"message"`
	FinishReason  string         `json:"finish_reason"`
	FinishDetails *FinishDetails `json:"finish_details,omitempty"`
}

type FinishDetails struct {
	Type      string     `json:"type"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type StreamChoice struct {
	Index         int            `json:"index"`
	Delta         Delta          `json:"delta"`
	FinishReason  *string        `json:"finish_reason"`
	FinishDetails *FinishDetails `json:"finish_details,omitempty"`
}

type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCallContent struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIStreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

type DeepSeekRequest struct {
	ChatSessionID   string   `json:"chat_session_id"`
	ParentMessageID *int     `json:"parent_message_id"`
	Prompt          string   `json:"prompt"`
	RefFileIds      []string `json:"ref_file_ids"`
	ThinkingEnabled bool     `json:"thinking_enabled"`
	SearchEnabled   bool     `json:"search_enabled"`
	AudioID         *string  `json:"audio_id"`
	Preempt         bool     `json:"preempt"`
}

type PowChallenge struct {
	Algorithm   string `json:"algorithm"`
	Challenge   string `json:"challenge"`
	Salt        string `json:"salt"`
	Signature   string `json:"signature"`
	Difficulty  int    `json:"difficulty"`
	ExpireAt    int64  `json:"expire_at"`
	ExpireAfter int    `json:"expire_after"`
	TargetPath  string `json:"target_path"`
}

type PowResponse struct {
	Algorithm  string `json:"algorithm"`
	Challenge  string `json:"challenge"`
	Salt       string `json:"salt"`
	Signature  string `json:"signature"`
	Answer     int    `json:"answer"`
	TargetPath string `json:"target_path"`
}

func getLeimToken() string {
	req, _ := http.NewRequest("GET", "https://hif-leim.deepseek.com/query", nil)
	req.Header.Set("X-Client-Platform", "android")
	req.Header.Set("X-Client-Version", "1.7.10")
	req.Header.Set("X-Client-Locale", "en_US")
	req.Header.Set("X-Client-Bundle-Id", "com.deepseek.chat")
	req.Header.Set("User-Agent", "DeepSeek/1.7.10 Android/34")

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("Failed to get LEIM token: %v", err)
		return ""
	}
	defer resp.Body.Close()

	var result struct {
		Data struct {
			BizData struct {
				Value string `json:"value"`
			} `json:"biz_data"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Printf("Failed to parse LEIM response: %v", err)
		return ""
	}
	return result.Data.BizData.Value
}

func createChatSession(bearerToken, leimToken string) (string, error) {
	req, err := http.NewRequest("POST", "https://chat.deepseek.com/api/v0/chat_session/create", nil)
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "DeepSeek/1.7.10 Android/34")
	req.Header.Set("X-Client-Platform", "android")
	req.Header.Set("X-Client-Version", "1.7.10")
	req.Header.Set("X-Client-Locale", "en_US")
	req.Header.Set("X-Client-Bundle-Id", "com.deepseek.chat")
	req.Header.Set("Authorization", "Bearer "+bearerToken)
	if leimToken != "" {
		req.Header.Set("X-Hif-Leim", leimToken)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Data struct {
			BizData struct {
				ChatSession struct {
					ID string `json:"id"`
				} `json:"chat_session"`
			} `json:"biz_data"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse session response: %v", err)
	}

	log.Printf("Created chat session: %s", result.Data.BizData.ChatSession.ID)
	return result.Data.BizData.ChatSession.ID, nil
}

func solvePow(bearerToken, leimToken string) (*PowResponse, error) {
	payload := map[string]string{"target_path": "/api/v0/chat/completion"}
	data, _ := json.Marshal(payload)

	req, err := http.NewRequest("POST", deepseekPowURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "DeepSeek/1.7.10 Android/34")
	req.Header.Set("X-Client-Platform", "android")
	req.Header.Set("X-Client-Version", "1.7.10")
	req.Header.Set("X-Client-Locale", "en_US")
	req.Header.Set("X-Client-Bundle-Id", "com.deepseek.chat")
	req.Header.Set("Authorization", "Bearer "+bearerToken)
	if leimToken != "" {
		req.Header.Set("X-Hif-Leim", leimToken)
	}

	client := httpClient
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result struct {
		Data struct {
			BizData struct {
				Challenge PowChallenge `json:"challenge"`
			} `json:"biz_data"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse PoW response: %v", err)
	}

	ch := result.Data.BizData.Challenge
	log.Printf("PoW challenge: algo=%s difficulty=%d", ch.Algorithm, ch.Difficulty)

	targetBin, err := hex.DecodeString(ch.Challenge)
	if err != nil || len(targetBin) != 32 {
		return nil, fmt.Errorf("invalid challenge hex")
	}

	createdAt := ch.ExpireAt
	baseStr := []byte(fmt.Sprintf("%s_%d_", ch.Salt, createdAt))

	target0 := binary.LittleEndian.Uint64(targetBin[0:8])
	target1 := binary.LittleEndian.Uint64(targetBin[8:16])
	target2 := binary.LittleEndian.Uint64(targetBin[16:24])
	target3 := binary.LittleEndian.Uint64(targetBin[24:32])

	found := make(chan int, 1)
	stop := make(chan struct{})
	cores := runtime.NumCPU()

	for w := 0; w < cores; w++ {
		go func(id int) {
			buf := make([]byte, 0, 128)
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}

				nonce := i*cores + id
				buf = buf[:0]
				buf = append(buf, baseStr...)
				buf = strconv.AppendInt(buf, int64(nonce), 10)

				hash := customSHA3_256(buf)

				if hash[0] == targetBin[0] && hash[1] == targetBin[1] &&
					hash[2] == targetBin[2] && hash[3] == targetBin[3] &&
					binary.LittleEndian.Uint64(hash[0:8]) == target0 &&
					binary.LittleEndian.Uint64(hash[8:16]) == target1 &&
					binary.LittleEndian.Uint64(hash[16:24]) == target2 &&
					binary.LittleEndian.Uint64(hash[24:32]) == target3 {
					select {
					case found <- nonce:
					default:
					}
					return
				}
			}
		}(w)
	}

	select {
	case nonce := <-found:
		close(stop)
		log.Printf("PoW solved: nonce=%d", nonce)
		return &PowResponse{
			Algorithm:  ch.Algorithm,
			Challenge:  ch.Challenge,
			Salt:       ch.Salt,
			Signature:  ch.Signature,
			Answer:     nonce,
			TargetPath: ch.TargetPath,
		}, nil
	case <-time.After(120 * time.Second):
		close(stop)
		return nil, fmt.Errorf("PoW timeout after 120 seconds")
	}
}

func encodePowResponse(pow *PowResponse) string {
	data, _ := json.Marshal(pow)
	return base64.StdEncoding.EncodeToString(data)
}

func buildQuery(messages []Message) string {
	var parts []string
	for _, m := range messages {
		switch m.Role {
		case "system":
			parts = append(parts, "[System: "+m.Content+"]")
		case "user":
			parts = append(parts, m.Content)
		case "assistant":
			if m.Content != "" && !strings.Contains(m.Content, "tool_call") {
				parts = append(parts, m.Content)
			}
			for _, tc := range m.ToolCalls {
				parts = append(parts, fmt.Sprintf("ToolCall: {\"name\": \"%s\", \"arguments\": %s}", tc.Function.Name, tc.Function.Arguments))
			}
		case "tool":
			parts = append(parts, fmt.Sprintf("[Tool result for %s]: %s", m.Name, m.Content))
		}
	}
	return strings.Join(parts, "\n\n")
}

func formatToolsForPrompt(tools []Tool) string {
	var buf bytes.Buffer
	buf.WriteString("## Tools (YOU HAVE ACCESS TO THESE)\n")
	buf.WriteString("When user asks you to perform an action, you MUST use these tools. Respond with ONLY a JSON object:\n")
	buf.WriteString(`{"tool_call": {"name": "function_name", "arguments": {"param1": "value1"}}}`)
	buf.WriteString("\n\nTools:\n")

	for _, tool := range tools {
		if tool.Type == "function" {
			params := ""
			if tool.Function.Parameters != nil {
				if p, ok := tool.Function.Parameters.(map[string]interface{}); ok {
					if props, ok := p["properties"].(map[string]interface{}); ok {
						var paramList []string
						for name, prop := range props {
							desc := ""
							if m, ok := prop.(map[string]interface{}); ok {
								if d, ok := m["description"].(string); ok {
									desc = " - " + d
								}
							}
							paramList = append(paramList, name+desc)
						}
						params = strings.Join(paramList, ", ")
					}
				}
			}
			buf.WriteString(fmt.Sprintf("- %s(%s): %s\n", tool.Function.Name, params, tool.Function.Description))
		}
	}
	return buf.String()
}

func extractToolCalls(text string) ([]ToolCall, string) {
	cleanedText := cleanContent(text)

	positions := findAllToolCallPositions(text)
	if len(positions) == 0 {
		return nil, cleanedText
	}

	var toolCalls []ToolCall

	for _, pos := range positions {
		jsonStr := text[pos.Start:pos.End]

		// Try wrapped format: {"tool_call": {"name": ..., "arguments": ...}}
		var wrapped struct {
			ToolCall struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"tool_call"`
		}
		if json.Unmarshal([]byte(jsonStr), &wrapped) == nil && wrapped.ToolCall.Name != "" {
			toolCalls = append(toolCalls, ToolCall{
				Index: len(toolCalls),
				ID:    "call_" + uuid.New().String()[:8],
				Type:  "function",
				Function: FunctionCall{
					Name:      wrapped.ToolCall.Name,
					Arguments: string(wrapped.ToolCall.Arguments),
				},
			})
			continue
		}

		// Try bare format: {"name": ..., "arguments": ...}
		var bare struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if json.Unmarshal([]byte(jsonStr), &bare) == nil && bare.Name != "" && bare.Arguments != nil {
			toolCalls = append(toolCalls, ToolCall{
				Index: len(toolCalls),
				ID:    "call_" + uuid.New().String()[:8],
				Type:  "function",
				Function: FunctionCall{
					Name:      bare.Name,
					Arguments: string(bare.Arguments),
				},
			})
		}
	}

	return toolCalls, cleanedText
}

type toolCallPos struct {
	Start int
	End   int
}

func findAllToolCallPositions(text string) []toolCallPos {
	var positions []toolCallPos

	// Search for multiple patterns that indicate a tool call.
	patterns := []string{`{"tool_call"`, `tool_call`, `{"name"`}

	for _, pattern := range patterns {
		searchFrom := 0
		for {
			idx := strings.Index(text[searchFrom:], pattern)
			if idx == -1 {
				break
			}

			actualStart := searchFrom + idx

			// Check if this position overlaps with an already-found position.
			overlap := false
			for _, p := range positions {
				if actualStart >= p.Start && actualStart < p.End {
					overlap = true
					break
				}
			}
			if overlap {
				searchFrom = actualStart + 1
				continue
			}

			depth := 0
			startBrace := -1
			endBrace := -1

			for i := actualStart; i < len(text); i++ {
				if text[i] == '{' {
					if startBrace == -1 {
						startBrace = i
					}
					depth++
				} else if text[i] == '}' {
					depth--
					if depth == 0 {
						endBrace = i + 1
						break
					}
				}
			}

			if startBrace != -1 && endBrace != -1 {
				positions = append(positions, toolCallPos{Start: startBrace, End: endBrace})
				searchFrom = endBrace
			} else {
				searchFrom = actualStart + 1
			}
		}
	}

	return positions
}

func parseSimpleToolCall(text string) *ToolCall {
	cleaned := strings.ReplaceAll(text, "tool_call", "")
	cleaned = strings.TrimSpace(cleaned)
	cleaned = strings.Trim(cleaned, "{}")

	name := ""
	var args map[string]string

	pairsRe := regexp.MustCompile(`(\w+)\s+([^\s{},]+)`)
	matches := pairsRe.FindAllStringSubmatch(cleaned, -1)

	for _, m := range matches {
		if len(m) >= 3 {
			key := m[1]
			value := m[2]
			if key == "name" {
				name = value
			} else if key == "command" || key == "pattern" || key == "path" || key == "filePath" || key == "content" || key == "oldString" || key == "newString" || key == "include" || key == "workdir" || key == "exclude" {
				if args == nil {
					args = make(map[string]string)
				}
				args[key] = value
			}
		}
	}

	if name == "" && len(matches) > 0 && len(matches[0]) >= 3 {
		name = matches[0][2]
	}

	if name == "" {
		return nil
	}

	argsJson, _ := json.Marshal(args)

	return &ToolCall{
		Index: 0,
		ID:    "call_" + uuid.New().String()[:8],
		Type:  "function",
		Function: FunctionCall{
			Name:      name,
			Arguments: string(argsJson),
		},
	}
}

func cleanContent(text string) string {
	text = strings.ReplaceAll(text, "\\", "")
	text = strings.ReplaceAll(text, "FINISHED", "")
	text = strings.ReplaceAll(text, "finished", "")

	// Strip JSON blocks that look like tool calls.
	toolCallPrefixes := []string{`{"tool_ca`, `{"name"`}
	for _, prefix := range toolCallPrefixes {
		for {
			idx := strings.Index(text, prefix)
			if idx == -1 {
				break
			}
			depth := 0
			end := -1
			for i := idx; i < len(text); i++ {
				if text[i] == '{' {
					depth++
				} else if text[i] == '}' {
					depth--
					if depth == 0 {
						end = i + 1
						break
					}
				}
			}
			if end == -1 {
				break
			}
			text = text[:idx] + text[end:]
		}
	}

	// Strip "ToolCall:" labels.
	text = strings.ReplaceAll(text, "ToolCall:", "")

	return strings.TrimSpace(text)
}

func collectSSEResponse(resp *http.Response) (string, error) {
	var fullText string
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		_, isDone := parseDeepSeekStream(line)
		if isDone {
			break
		}
		fullText += line + "\n"
	}

	bodyBytes, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(bodyBytes), `"v":"`) {
		re := regexp.MustCompile(`"v":"([^"]*)"`)
		matches := re.FindAllStringSubmatch(string(bodyBytes), -1)
		for _, m := range matches {
			if len(m) > 1 {
				fullText += m[1]
			}
		}
	}

	re2 := regexp.MustCompile(`\{"p":"[^"]*/content","o":"APPEND","v":"([^"]*)"\}`)
	matches2 := re2.FindAllStringSubmatch(string(bodyBytes), -1)
	for _, m := range matches2 {
		if len(m) > 1 {
			fullText += m[1]
		}
	}

	return fullText, nil
}

// sseState tracks which DeepSeek fragment is currently active so that THINK
// (reasoning) tokens can be separated from RESPONSE tokens across SSE lines.
type sseState struct {
	currentFragType string // "THINK" or "RESPONSE"
}

// parseLine processes a single SSE line and returns
// (responseToken, thinkingToken, isDone).
func (s *sseState) parseLine(line string) (string, string, bool) {
	line = strings.TrimSpace(line)

	if strings.HasPrefix(line, "event:") {
		if strings.TrimSpace(strings.TrimPrefix(line, "event:")) == "close" {
			return "", "", true
		}
		return "", "", false
	}

	if !strings.HasPrefix(line, "data:") {
		return "", "", false
	}

	data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if data == "" || data == "[DONE]" {
		return "", "", false
	}

	var obj map[string]json.RawMessage
	if json.Unmarshal([]byte(data), &obj) != nil {
		return "", "", false
	}

	pRaw, hasP := obj["p"]
	oRaw, hasO := obj["o"]
	vRaw, hasV := obj["v"]
	if !hasV {
		return "", "", false
	}

	if hasP {
		var pStr string
		json.Unmarshal(pRaw, &pStr)

		switch {
		case pStr == "response/fragments":
			var fragOp string
			if hasO {
				json.Unmarshal(oRaw, &fragOp)
			}
			if fragOp == "APPEND" {
				var newFrags []struct {
					Type    string `json:"type"`
					Content string `json:"content"`
				}
				if json.Unmarshal(vRaw, &newFrags) == nil && len(newFrags) > 0 {
					s.currentFragType = newFrags[len(newFrags)-1].Type
					for _, f := range newFrags {
						if f.Type == "RESPONSE" {
							return f.Content, "", false
						} else if f.Type == "THINK" {
							return "", f.Content, false
						}
					}
				}
			}

		case strings.HasSuffix(pStr, "/content"):
			var content string
			if json.Unmarshal(vRaw, &content) == nil {
				if s.currentFragType == "RESPONSE" {
					return content, "", false
				} else if s.currentFragType == "THINK" {
					return "", content, false
				}
			}

		case pStr == "response/status":
			var status string
			if json.Unmarshal(vRaw, &status) == nil && status == "FINISHED" {
				return "", "", true
			}
		}
	} else {
		// No "p" field.
		var strV string
		if json.Unmarshal(vRaw, &strV) == nil {
			// Bare token delta — belongs to the current active fragment.
			if s.currentFragType == "RESPONSE" {
				return strV, "", false
			} else if s.currentFragType == "THINK" {
				return "", strV, false
			}
		} else {
			// v is an object — initial full-response snapshot.
			var fullResp struct {
				Response struct {
					Fragments []struct {
						Type    string `json:"type"`
						Content string `json:"content"`
					} `json:"fragments"`
				} `json:"response"`
			}
			if json.Unmarshal(vRaw, &fullResp) == nil && len(fullResp.Response.Fragments) > 0 {
				s.currentFragType = fullResp.Response.Fragments[len(fullResp.Response.Fragments)-1].Type
				var resp, think string
				for _, f := range fullResp.Response.Fragments {
					if f.Type == "RESPONSE" {
						resp += f.Content
					} else if f.Type == "THINK" {
						think += f.Content
					}
				}
				return resp, think, false
			}
		}
	}
	return "", "", false
}

// parseSSEContent parses a full SSE body, returning only RESPONSE content.
func parseSSEContent(body []byte) string {
	state := &sseState{}
	var result strings.Builder
	for _, line := range strings.Split(string(body), "\n") {
		resp, _, done := state.parseLine(line)
		if done {
			break
		}
		result.WriteString(resp)
	}
	return result.String()
}

type ToolResult struct {
	Name    string
	Content string
	Error   string
}

func executeTool(name string, args map[string]interface{}) ToolResult {
	switch name {
	case "read":
		return executeRead(args)
	case "write":
		return executeWrite(args)
	case "edit":
		return executeEdit(args)
	case "bash":
		return executeBash(args)
	case "glob":
		return executeGlob(args)
	case "grep":
		return executeGrep(args)
	default:
		return ToolResult{Error: fmt.Sprintf("Unknown tool: %s", name)}
	}
}

func getStringArg(args map[string]interface{}, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getStringSliceArg(args map[string]interface{}, key string) []string {
	if v, ok := args[key]; ok {
		if arr, ok := v.([]interface{}); ok {
			var result []string
			for _, item := range arr {
				if s, ok := item.(string); ok {
					result = append(result, s)
				}
			}
			return result
		}
	}
	return nil
}

func executeRead(args map[string]interface{}) ToolResult {
	filePath := getStringArg(args, "filePath")
	if filePath == "" {
		return ToolResult{Error: "filePath is required"}
	}

	info, err := os.Stat(filePath)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to stat path: %v", err)}
	}

	if info.IsDir() {
		entries, err := os.ReadDir(filePath)
		if err != nil {
			return ToolResult{Error: fmt.Sprintf("Failed to read directory: %v", err)}
		}
		var lines []string
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() {
				name += "/"
			}
			lines = append(lines, name)
		}
		return ToolResult{Content: strings.Join(lines, "\n")}
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to read file: %v", err)}
	}

	return ToolResult{Content: string(content)}
}

func executeWrite(args map[string]interface{}) ToolResult {
	filePath := getStringArg(args, "filePath")
	content := getStringArg(args, "content")
	if filePath == "" {
		return ToolResult{Error: "filePath is required"}
	}

	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to create directory: %v", err)}
	}

	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to write file: %v", err)}
	}

	return ToolResult{Content: fmt.Sprintf("Written %d bytes to %s", len(content), filePath)}
}

func executeEdit(args map[string]interface{}) ToolResult {
	filePath := getStringArg(args, "filePath")
	oldString := getStringArg(args, "oldString")
	newString := getStringArg(args, "newString")

	if filePath == "" || oldString == "" {
		return ToolResult{Error: "filePath and oldString are required"}
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to read file: %v", err)}
	}

	if !strings.Contains(string(content), oldString) {
		return ToolResult{Error: fmt.Sprintf("oldString not found in file")}
	}

	newContent := strings.Replace(string(content), oldString, newString, 1)
	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return ToolResult{Error: fmt.Sprintf("Failed to write file: %v", err)}
	}

	return ToolResult{Content: fmt.Sprintf("Edited %s", filePath)}
}

func executeBash(args map[string]interface{}) ToolResult {
	command := getStringArg(args, "command")
	if command == "" {
		return ToolResult{Error: "command is required"}
	}

	workdir := getStringArg(args, "workdir")
	if workdir == "" {
		workdir = "."
	}

	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = workdir
	output, err := cmd.CombinedOutput()

	if err != nil {
		return ToolResult{
			Content: string(output),
			Error:   fmt.Sprintf("Command failed: %v", err),
		}
	}

	return ToolResult{Content: string(output)}
}

func executeGlob(args map[string]interface{}) ToolResult {
	pattern := getStringArg(args, "pattern")
	path := getStringArg(args, "path")
	exclude := getStringSliceArg(args, "exclude")

	if pattern == "" {
		pattern = "*"
	}
	if path == "" {
		path = "."
	}

	matches, err := filepath.Glob(filepath.Join(path, pattern))
	if err != nil {
		return ToolResult{Error: fmt.Sprintf("Glob failed: %v", err)}
	}

	var filtered []string
	for _, m := range matches {
		skip := false
		for _, ex := range exclude {
			if matched, _ := filepath.Match(ex, filepath.Base(m)); matched {
				skip = true
				break
			}
		}
		if !skip {
			rel, _ := filepath.Rel(".", m)
			filtered = append(filtered, rel)
		}
	}

	return ToolResult{Content: strings.Join(filtered, "\n")}
}

func executeGrep(args map[string]interface{}) ToolResult {
	pattern := getStringArg(args, "pattern")
	path := getStringArg(args, "path")
	include := getStringArg(args, "include")

	if pattern == "" {
		return ToolResult{Error: "pattern is required"}
	}
	if path == "" {
		path = "."
	}

	var results []string
	var regex *regexp.Regexp
	var err error

	if strings.HasPrefix(pattern, "regex:") {
		pattern = strings.TrimPrefix(pattern, "regex:")
		regex, err = regexp.Compile(pattern)
		if err != nil {
			return ToolResult{Error: fmt.Sprintf("Invalid regex: %v", err)}
		}
	} else {
		regex = regexp.MustCompile(regexp.QuoteMeta(pattern))
	}

	err = filepath.Walk(path, func(walkPath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		if info.IsDir() {
			return nil
		}

		if include != "" && !strings.HasSuffix(walkPath, include) {
			return nil
		}

		if strings.Contains(walkPath, "/node_modules/") || strings.Contains(walkPath, "/.git/") {
			return nil
		}

		content, err := os.ReadFile(walkPath)
		if err != nil {
			return nil
		}

		lines := strings.Split(string(content), "\n")
		for i, line := range lines {
			if regex.MatchString(line) {
				rel, _ := filepath.Rel(".", walkPath)
				results = append(results, fmt.Sprintf("%s:%d: %s", rel, i+1, line))
			}
		}

		return nil
	})

	if err != nil {
		return ToolResult{Error: fmt.Sprintf("Grep failed: %v", err)}
	}

	if len(results) == 0 {
		return ToolResult{Content: "No matches found"}
	}

	return ToolResult{Content: strings.Join(results, "\n")}
}

func executeToolLoop(chatSessionID string, messages []Message, tools []Tool, thinking, search bool, bearerToken, leimToken string) (string, error) {
	maxIterations := 5
	startLen := len(messages)

	for i := 0; i < maxIterations; i++ {
		pow, err := solvePow(bearerToken, leimToken)
		if err != nil {
			return "", fmt.Errorf("PoW failed: %v", err)
		}
		powResp := encodePowResponse(pow)

		var prompt string
		if i == 0 {
			prompt = buildQuery(messages)
		} else {
			recentMessages := messages
			if len(messages) > startLen+4 {
				recentMessages = messages[len(messages)-4:]
			}
			prompt = buildQuery(recentMessages)
		}

		if len(tools) > 0 {
			toolsIntro := formatToolsForPrompt(tools)
			prompt = toolsIntro + "\n\n" + prompt
		}

		resp, err := doDeepSeekRequest(chatSessionID, prompt, thinking, search, bearerToken, leimToken, powResp)
		if err != nil {
			return "", fmt.Errorf("DeepSeek request failed: %v", err)
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("DeepSeek error %d: %s", resp.StatusCode, string(bodyBytes))
		}

		content := parseSSEContent(bodyBytes)

		toolCalls, cleanedContent := extractToolCalls(content)

		if len(toolCalls) > 0 {
			log.Printf("Executing %d tool calls", len(toolCalls))

			results := []string{}
			for _, tc := range toolCalls {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					log.Printf("Failed to parse tool args: %v", err)
					continue
				}

				log.Printf("Executing tool: %s with args: %v", tc.Function.Name, args)
				result := executeTool(tc.Function.Name, args)

				if result.Error != "" {
					results = append(results, fmt.Sprintf("Error: %s", result.Error))
				} else {
					results = append(results, result.Content)
				}
			}

			return strings.Join(results, "\n"), nil
		}

		return cleanedContent, nil
	}

	return "", fmt.Errorf("Max tool iterations exceeded")
}

func doDeepSeekRequest(chatSessionID, prompt string, thinking, search bool, bearerToken, leimToken, powResp string) (*http.Response, error) {
	req := DeepSeekRequest{
		ChatSessionID:   chatSessionID,
		ParentMessageID: nil,
		Prompt:          prompt,
		RefFileIds:      []string{},
		ThinkingEnabled: thinking,
		SearchEnabled:   search,
		AudioID:         nil,
		Preempt:         false,
	}
	data, _ := json.Marshal(req)

	httpReq, err := http.NewRequest("POST", deepseekChatURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("User-Agent", "DeepSeek/1.7.10 Android/34")
	httpReq.Header.Set("X-Client-Platform", "android")
	httpReq.Header.Set("X-Client-Version", "1.7.10")
	httpReq.Header.Set("X-Client-Locale", "en_US")
	httpReq.Header.Set("X-Client-Bundle-Id", "com.deepseek.chat")
	httpReq.Header.Set("X-Ds-Pow-Response", powResp)
	httpReq.Header.Set("Authorization", "Bearer "+bearerToken)
	if leimToken != "" {
		httpReq.Header.Set("X-Hif-Leim", leimToken)
	}

	return httpClient.Do(httpReq)
}

func parseDeepSeekStream(line string) (string, bool) {
	line = strings.TrimSpace(line)

	if strings.HasPrefix(line, "event:") {
		event := strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		if event == "close" {
			return "", true
		}
		return "", false
	}

	if !strings.HasPrefix(line, "data:") {
		return "", false
	}

	data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if data == "" {
		return "", false
	}

	var directContent struct {
		V string `json:"v"`
	}
	if json.Unmarshal([]byte(data), &directContent) == nil && directContent.V != "" {
		var check struct {
			P string `json:"p"`
			O string `json:"o"`
		}
		if json.Unmarshal([]byte(data), &check) != nil || (check.P == "" && check.O == "") {
			return directContent.V, false
		}
	}

	var genericOp struct {
		P string `json:"p"`
		O string `json:"o"`
		V string `json:"v"`
	}
	if json.Unmarshal([]byte(data), &genericOp) == nil {
		if genericOp.P == "response/status" && genericOp.V == "FINISHED" {
			return "", true
		}
		if genericOp.O == "APPEND" && strings.HasSuffix(genericOp.P, "/content") {
			return genericOp.V, false
		}
	}

	var structured struct {
		V struct {
			Response struct {
				Fragments []struct {
					Type    string `json:"type"`
					Content string `json:"content"`
				} `json:"fragments"`
			} `json:"response"`
		} `json:"v"`
	}
	if json.Unmarshal([]byte(data), &structured) == nil {
		var content string
		for _, frag := range structured.V.Response.Fragments {
			if frag.Type == "RESPONSE" {
				content += frag.Content
			}
		}
		return content, false
	}

	var errResp struct {
		Code int    `json:"code"`
		Msg  string `json:"msg"`
	}
	if json.Unmarshal([]byte(data), &errResp) == nil {
		if errResp.Code != 0 {
			log.Printf("API error: code=%d msg=%s", errResp.Code, errResp.Msg)
			return "", false
		}
	}

	var topLevelErr struct {
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(data), &topLevelErr) == nil {
		if topLevelErr.Error.Code != 0 || topLevelErr.Error.Message != "" {
			log.Printf("API top-level error: code=%d msg=%s", topLevelErr.Error.Code, topLevelErr.Error.Message)
			return "", false
		}
	}

	var bizErr struct {
		Data struct {
			BizCode    int    `json:"biz_code"`
			BizMessage string `json:"biz_msg"`
		} `json:"data"`
	}
	if json.Unmarshal([]byte(data), &bizErr) == nil {
		if bizErr.Data.BizCode != 0 {
			log.Printf("API biz error: code=%d msg=%s", bizErr.Data.BizCode, bizErr.Data.BizMessage)
			return "", false
		}
	}

	return "", false
}

// writeError writes an OpenAI-spec JSON error response.
func writeError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"message": message,
			"type":    "server_error",
			"code":    fmt.Sprintf("%d", code),
		},
	})
}

// streamResponse fires off a single DeepSeek request and forwards tokens to
// the client as they arrive (true streaming). RESPONSE tokens are sent via
// sendChunk; THINK tokens are sent as reasoning_content (only when thinking
// is enabled). Returns the full assembled RESPONSE text.
func streamResponse(chatSessionID string, messages []Message, tools []Tool, thinking, search bool, bearerToken, leimToken string, sendChunk func(Delta, *string)) (string, error) {
	pow, err := solvePow(bearerToken, leimToken)
	if err != nil {
		return "", fmt.Errorf("PoW failed: %v", err)
	}

	prompt := buildQuery(messages)
	if len(tools) > 0 {
		prompt = formatToolsForPrompt(tools) + "\n\n" + prompt
	}
	resp, err := doDeepSeekRequest(chatSessionID, prompt, thinking, search, bearerToken, leimToken, encodePowResponse(pow))
	if err != nil {
		return "", fmt.Errorf("DeepSeek request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("DeepSeek error %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	state := &sseState{}
	var fullResponse strings.Builder
	bufferContent := len(tools) > 0

	for scanner.Scan() {
		respTok, thinkTok, done := state.parseLine(scanner.Text())
		if done {
			break
		}
		if thinkTok != "" && thinking {
			sendChunk(Delta{ReasoningContent: thinkTok}, nil)
		}
		if respTok != "" {
			fullResponse.WriteString(respTok)
			if !bufferContent {
				sendChunk(Delta{Content: respTok}, nil)
			}
		}
	}

	return fullResponse.String(), scanner.Err()
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	bearerToken := os.Getenv("DS_BEARER_TOKEN")
	if bearerToken == "" {
		writeError(w, "DS_BEARER_TOKEN not configured", http.StatusInternalServerError)
		return
	}

	leimToken := os.Getenv("DS_LEIM_TOKEN")
	if leimToken == "" {
		log.Printf("Fetching LEIM token...")
		leimToken = getLeimToken()
	}

	modelConfig, ok := modelMap[req.Model]
	if !ok {
		modelConfig = modelMap["default"]
	}

	thinking := modelConfig["thinking_enabled"].(bool)
	search := modelConfig["search_enabled"].(bool)

	log.Printf("Creating chat session...")
	chatSessionID, err := createChatSession(bearerToken, leimToken)
	if err != nil {
		writeError(w, "Failed to create chat session: "+err.Error(), http.StatusInternalServerError)
		return
	}

	completionID := "chatcmpl-" + uuid.New().String()
	created := time.Now().Unix()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeError(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		sendChunk := func(delta Delta, finishReason *string) {
			chunk := OpenAIStreamChunk{
				ID: completionID, Object: "chat.completion.chunk", Created: created, Model: req.Model,
				Choices: []StreamChoice{{Index: 0, Delta: delta, FinishReason: finishReason}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		sendChunk(Delta{Role: "assistant"}, nil)

		// Stream from DeepSeek. When tools are present, content is buffered
		// so we can detect tool calls before sending to the client.
		var result string
		result, err = streamResponse(chatSessionID, req.Messages, req.Tools, thinking, search, bearerToken, leimToken, sendChunk)

		if err != nil {
			log.Printf("Stream error: %v", err)
			errChunk, _ := json.Marshal(map[string]interface{}{
				"error": map[string]string{"message": err.Error(), "type": "server_error"},
			})
			fmt.Fprintf(w, "data: %s\n\n", errChunk)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		if len(req.Tools) > 0 {
			toolCalls, cleanedContent := extractToolCalls(result)
			if len(toolCalls) > 0 {
				// Return tool calls in OpenAI format — let the client handle execution.
				log.Printf("Returning %d tool calls to client", len(toolCalls))
				sendChunk(Delta{ToolCalls: toolCalls}, nil)
				toolCallsReason := "tool_calls"
				sendChunk(Delta{}, &toolCallsReason)
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				log.Printf("query=%q model=%s stream=true tool_calls=%d", truncate(buildQuery(req.Messages), 80), req.Model, len(toolCalls))
				return
			}
			// No tool calls — send the buffered content.
			sendChunk(Delta{Content: cleanedContent}, nil)
		}

		stop := "stop"
		sendChunk(Delta{}, &stop)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		log.Printf("query=%q model=%s stream=true chars=%d", truncate(buildQuery(req.Messages), 80), req.Model, len(result))
	} else {
		pow, powErr := solvePow(bearerToken, leimToken)
		if powErr != nil {
			writeError(w, "PoW failed: "+powErr.Error(), http.StatusInternalServerError)
			return
		}

		prompt := buildQuery(req.Messages)
		if len(req.Tools) > 0 {
			prompt = formatToolsForPrompt(req.Tools) + "\n\n" + prompt
		}
		resp, reqErr := doDeepSeekRequest(chatSessionID, prompt, thinking, search, bearerToken, leimToken, encodePowResponse(pow))
		if reqErr != nil {
			writeError(w, "DeepSeek request failed: "+reqErr.Error(), http.StatusInternalServerError)
			return
		}
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			writeError(w, fmt.Sprintf("DeepSeek error %d: %s", resp.StatusCode, string(bodyBytes)), http.StatusBadGateway)
			return
		}

		content := parseSSEContent(bodyBytes)
		toolCalls, cleanedContent := extractToolCalls(content)

		promptWords := len(strings.Fields(buildQuery(req.Messages)))

		if len(toolCalls) > 0 {
			// Return tool calls to the client in OpenAI format.
			log.Printf("Returning %d tool calls to client (non-stream)", len(toolCalls))
			response := OpenAIResponse{
				ID: completionID, Object: "chat.completion", Created: created, Model: req.Model,
				Choices: []Choice{{
					Index:        0,
					Message:      Message{Role: "assistant", ToolCalls: toolCalls},
					FinishReason: "tool_calls",
				}},
				Usage: &Usage{
					PromptTokens: promptWords,
					TotalTokens:  promptWords,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
		} else {
			completionWords := len(strings.Fields(cleanedContent))
			response := OpenAIResponse{
				ID: completionID, Object: "chat.completion", Created: created, Model: req.Model,
				Choices: []Choice{{
					Index:        0,
					Message:      Message{Role: "assistant", Content: cleanedContent},
					FinishReason: "stop",
				}},
				Usage: &Usage{
					PromptTokens:     promptWords,
					CompletionTokens: completionWords,
					TotalTokens:      promptWords + completionWords,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
		}
		log.Printf("query=%q model=%s chars=%d", truncate(buildQuery(req.Messages), 80), req.Model, len(cleanedContent))
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	type ModelEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
	}
	models := struct {
		Object string       `json:"object"`
		Data   []ModelEntry `json:"data"`
	}{
		Object: "list",
		Data: []ModelEntry{
			{ID: "deepseek-chat", Object: "model", Created: 1700000000},
			{ID: "deepseek-chat-reasoning", Object: "model", Created: 1700000000},
			{ID: "deepseek-searcher", Object: "model", Created: 1700000000},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		apiKey := os.Getenv("PROXY_API_KEY")
		if apiKey != "" {
			auth := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			if auth != apiKey {
				http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
				return
			}
		}
		next(w, r)
	}
}

func getEnvOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file, reading from environment")
	}

	if os.Getenv("DS_BEARER_TOKEN") == "" {
		log.Fatal("DS_BEARER_TOKEN is required — set it in .env")
	}

	port := getEnvOrDefault("PORT", "8080")

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", authMiddleware(handleChatCompletions))
	mux.HandleFunc("/v1/models", authMiddleware(handleModels))

	log.Printf("DeepSeek OpenAI proxy listening on :%s", port)
	log.Printf("Using %d CPU cores for PoW", runtime.GOMAXPROCS(0))
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
