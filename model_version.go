package anthropic

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/wbrown/llmapi"
)

// modelDateSuffix matches the 8-digit date suffix on older Anthropic model
// IDs (e.g. "claude-3-5-sonnet-20241022"). The date does not contribute to
// the version and is stripped before parsing.
var modelDateSuffix = regexp.MustCompile(`-[0-9]{8}$`)

// parseModelVersion extracts the (major, minor) version from an Anthropic
// model ID. Returns ok=false if the string does not start with "claude-"
// or contains no numeric version component.
//
// Recognized forms:
//
//	claude-opus-4-8           -> (4, 8)
//	claude-sonnet-4-6         -> (4, 6)
//	claude-haiku-4-5-20251001 -> (4, 5)
//	claude-3-5-sonnet-...     -> (3, 5)
//	claude-3-opus-...         -> (3, 0)   no minor present
//	claude-2.1                -> (2, 1)
//	claude-instant-1.2        -> (1, 2)
//
// When only a major component is present, minor is reported as 0.
func parseModelVersion(model string) (major, minor int, ok bool) {
	if !strings.HasPrefix(model, "claude-") {
		return 0, 0, false
	}
	rest := strings.TrimPrefix(model, "claude-")
	rest = modelDateSuffix.ReplaceAllString(rest, "")

	// Split on both '-' and '.' so "claude-opus-4-8" and "claude-2.1"
	// parse uniformly.
	tokens := strings.FieldsFunc(rest, func(r rune) bool {
		return r == '-' || r == '.'
	})

	var nums []int
	for _, tok := range tokens {
		n, err := strconv.Atoi(tok)
		if err != nil {
			continue
		}
		nums = append(nums, n)
	}
	if len(nums) == 0 {
		return 0, 0, false
	}
	major = nums[0]
	if len(nums) >= 2 {
		minor = nums[1]
	}
	return major, minor, true
}

// supportsSampling reports whether the given model accepts the temperature,
// top_p, and top_k sampling parameters.
//
// Anthropic deprecated all three on Claude Opus 4.7 and applies the same
// constraint to Claude Opus 4.8; sending any of them — even at the documented
// default value — returns HTTP 400 because the check is presence-based. The
// cutoff is applied family-agnostically: any model at version 4.7 or newer
// omits the sampling parameters.
//
// Unrecognized model IDs return false. Omitting the parameters is the safer
// default against an unknown future model, since the failure mode for the
// other direction is a hard 400 rather than a quality regression.
//
// See https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
func supportsSampling(model string) bool {
	major, minor, ok := parseModelVersion(model)
	if !ok {
		return false
	}
	if major > 4 {
		return false
	}
	if major == 4 && minor >= 7 {
		return false
	}
	return true
}

// resolveSampling computes the effective sampling parameters for a request.
// It layers per-call overrides over conversation defaults, then gates the
// result by the target model's capabilities: when the model does not accept
// sampling parameters, all three return values are zeroed so JSON marshalling
// omits them from the request body (Temperature returns nil, TopP and TopK
// rely on the existing omitempty tags).
//
// The non-zero override convention matches the original three-block pattern
// at each call site: a zero value in llmapi.Sampling means "use the
// conversation's configured value."
func resolveSampling(settings *SampleSettings, override llmapi.Sampling) (temperature *float64, topP float64, topK int) {
	if !supportsSampling(settings.Model) {
		return nil, 0, 0
	}
	t := settings.Temperature
	if override.Temperature != 0 {
		t = override.Temperature
	}
	p := settings.TopP
	if override.TopP != 0 {
		p = override.TopP
	}
	k := settings.TopK
	if override.TopK != 0 {
		k = override.TopK
	}
	return &t, p, k
}
