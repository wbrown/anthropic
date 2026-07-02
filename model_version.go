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

// supportsAdaptiveThinking reports whether the given model accepts
// thinking: {type: "adaptive"} paired with output_config.effort — the
// mechanism Anthropic recommends for controlling reasoning depth on Claude
// 4.6 and later, in place of the legacy thinking: {type: "enabled",
// budget_tokens: N} shape.
//
// Adaptive thinking became available (and recommended) starting with Claude
// Opus 4.6 and Claude Sonnet 4.6, where the legacy shape is deprecated but
// still functional as a transitional escape hatch. Claude Opus 4.7/4.8 and
// any Claude N.x family where N > 4 (Claude Sonnet 5, Claude Fable 5, Claude
// Mythos 5) go further and reject the legacy shape outright with a hard 400
// — adaptive thinking is the only mechanism those models accept. Either way,
// once a model reaches this cutoff, adaptive is the correct choice, so this
// function does not distinguish "recommended" from "required".
//
// Unrecognized model IDs return false, preserving this library's pre-existing
// budget-based behavior for models it doesn't yet recognize rather than
// guessing at a request shape that may not be supported at all.
func supportsAdaptiveThinking(model string) bool {
	major, minor, ok := parseModelVersion(model)
	if !ok {
		return false
	}
	if major > 4 {
		return true
	}
	return major == 4 && minor >= 6
}

// mustAlwaysThink reports whether the given model keeps thinking on
// unconditionally: Claude Fable 5 and Claude Mythos 5 reject an explicit
// thinking: {type: "disabled"} with a hard 400, and sending no thinking
// field at all doesn't turn thinking off either — it just runs adaptive
// thinking with thinking.display defaulting to "omitted", so the
// unavoidable (and billed) reasoning comes back as real thinking blocks
// with empty text. Every other model this library recognizes can genuinely
// be made to not think, whether by omitting the field (Opus 4.6-4.8, Sonnet
// 4.6) or an explicit disable.
//
// Unlike supportsAdaptiveThinking/supportsSampling, this is not a version
// cutoff — it names a specific model family by substring. A hypothetical
// future claude-opus-5-0 is not assumed to inherit Fable/Mythos's
// always-on behavior just because it shares a major version number.
func mustAlwaysThink(model string) bool {
	return strings.Contains(model, "fable") || strings.Contains(model, "mythos")
}

// defaultsToAdaptiveThinking reports whether the given model runs adaptive
// thinking when the request OMITS the thinking field entirely. Claude Sonnet 5
// does: unlike Opus 4.7/4.8 (whose omission-default is thinking off, matching
// Opus 4.6 and Sonnet 4.6), a Sonnet 5 request with no thinking field reasons
// anyway — billed and slow — so turning thinking off there requires an
// explicit thinking: {type: "disabled"}, which Sonnet 5 accepts.
//
// This predicate only covers models that CAN be disabled; the always-thinking
// family (see mustAlwaysThink) rejects {type: "disabled"} outright and reports
// false here — callers must check mustAlwaysThink first. Like mustAlwaysThink
// this is a family fact, not a version cutoff: a hypothetical claude-opus-5-0
// is not assumed to share Sonnet 5's omission-default, while later Sonnet
// versions (5.1+) are assumed to keep their own family's behavior.
func defaultsToAdaptiveThinking(model string) bool {
	if !strings.Contains(model, "sonnet") {
		return false
	}
	major, _, ok := parseModelVersion(model)
	return ok && major >= 5
}

// modelOutputCeiling reports the model's real per-request output ceiling on
// the synchronous Messages API, per Anthropic's models overview: 128000 for
// every 4.6+ model (Fable/Mythos 5, Opus 4.6-4.8, Sonnet 4.6, Sonnet 5),
// 64000 for the 4.5 tier (Sonnet 4.5, Opus 4.5, Haiku 4.5), 32000 for Opus
// 4.1. Generations older than 4.1 are retired at the API, and unrecognized
// IDs are unknown — both report 0, meaning "no ceiling known, do not clamp",
// which preserves this library's pre-existing behavior for models it cannot
// vouch for.
//
// See https://platform.claude.com/docs/en/about-claude/models/overview
func modelOutputCeiling(model string) int {
	major, minor, ok := parseModelVersion(model)
	if !ok {
		return 0
	}
	switch {
	case major > 4 || (major == 4 && minor >= 6):
		return 128000
	case major == 4 && minor == 5:
		return 64000
	case major == 4 && minor == 1:
		return 32000
	default:
		return 0
	}
}

// resolveSampling computes the effective sampling parameters for a request.
// It layers per-call overrides over conversation defaults, then gates the
// result by two independent constraints, either of which omits all three
// (Temperature returns nil, TopP and TopK rely on the existing omitempty
// tags): the target model's capabilities (see supportsSampling), and whether
// thinking is active on this request. Anthropic rejects any explicit
// temperature/top_p/top_k the moment thinking is enabled — "enabled" or
// "adaptive" — with a 400 ("temperature may only be set to 1 when thinking
// is enabled or in adaptive mode"), independent of the 4.7+ sampling-removal
// cutoff supportsSampling tracks. A Claude 4.6 model that fully supports
// sampling still cannot combine it with thinking.
//
// The non-zero override convention matches the original three-block pattern
// at each call site: a zero value in llmapi.Sampling means "use the
// conversation's configured value."
func resolveSampling(settings *SampleSettings, override llmapi.Sampling, thinkingActive bool) (temperature *float64, topP float64, topK int) {
	if !supportsSampling(settings.Model) || thinkingActive {
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
