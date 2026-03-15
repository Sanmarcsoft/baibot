/// Response value scoring and filtering for group chat bots.
///
/// Hybrid approach: prompt-based self-scoring ([SCORE:N] prefix) combined
/// with deterministic heuristics. Prevents repetitive, low-value responses
/// in multi-agent Matrix chat rooms.

use regex::Regex;
use std::sync::LazyLock;

static SCORE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\[SCORE:(\d{1,2})\]\s*").unwrap()
});

/// Parsed LLM response with optional self-assigned score.
pub struct ParsedResponse {
    pub score: Option<u8>,
    pub body: String,
}

/// Parse a [SCORE:N] prefix from the LLM response and strip it.
pub fn parse_scored_response(raw: &str) -> ParsedResponse {
    if let Some(caps) = SCORE_RE.captures(raw) {
        let score: u8 = caps.get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0)
            .min(10);
        let body = raw[caps.get(0).unwrap().end()..].to_string();
        ParsedResponse { score: Some(score), body }
    } else {
        ParsedResponse { score: None, body: raw.to_string() }
    }
}

/// Generic filler phrases that add no value.
const GENERIC_PHRASES: &[&str] = &[
    "ok", "okay", "sure", "got it", "noted", "thanks", "thank you",
    "sounds good", "will do", "no problem", "np", "yes", "no",
    "alright", "right", "understood", "acknowledged", "ack",
    "makes sense", "fair enough", "good point", "i see",
    "interesting", "cool", "nice", "great", "awesome",
];

/// Check if text contains information signals (code, URLs, numbers, multi-line).
fn has_information_signals(text: &str) -> bool {
    text.contains("```")
        || text.contains("http://")
        || text.contains("https://")
        || text.chars().any(|c| c.is_ascii_digit())
        || text.contains('\n')
        || text.len() > 120
}

/// Apply deterministic heuristics to score a response.
/// Returns (score_override, score_delta, reason).
fn apply_heuristics(body: &str) -> (Option<u8>, i16, &'static str) {
    let trimmed = body.trim();
    let lower = trimmed.to_lowercase();
    let word_count = trimmed.split_whitespace().count();

    // Hard suppress: explicit PASS signal
    if lower.starts_with("pass") || lower.starts_with("[pass]") {
        return (Some(0), 0, "explicit-pass");
    }

    // Hard suppress: empty
    if trimmed.is_empty() {
        return (Some(0), 0, "empty-body");
    }

    let mut delta: i16 = 0;
    let mut reason = "ok";

    // Penalty: ultra-short generic phrases
    if GENERIC_PHRASES.contains(&lower.as_str())
        || (word_count <= 3 && GENERIC_PHRASES.iter().any(|p| lower.contains(p)))
    {
        delta -= 4;
        reason = "generic-short";
    }
    // Penalty: short with no information signals
    else if word_count < 15 && !has_information_signals(trimmed) {
        delta -= 2;
        reason = "short-no-signals";
    }

    // Bonus: contains code
    if trimmed.contains("```") {
        delta += 2;
        reason = "contains-code";
    }

    // Bonus: contains URL
    if trimmed.contains("http://") || trimmed.contains("https://") {
        delta += 1;
    }

    // Bonus: answer opener
    const ANSWER_OPENERS: &[&str] = &[
        "the reason", "this is because", "you can", "you should", "to fix",
        "the issue is", "the error", "try ", "use ", "run ", "check ",
    ];
    if ANSWER_OPENERS.iter().any(|o| lower.starts_with(o)) {
        delta += 2;
        if reason == "ok" { reason = "answer-opener"; }
    }

    (None, delta, reason)
}

/// Default threshold for response filtering.
const DEFAULT_THRESHOLD: u8 = 7;
const DEFAULT_SCORE_WHEN_MISSING: u8 = 5;

/// Evaluate whether an LLM response should be sent to the Matrix room.
///
/// Returns `Some(cleaned_body)` if the response should be sent,
/// or `None` if it should be suppressed.
pub fn filter_response(raw_llm_response: &str) -> Option<String> {
    let parsed = parse_scored_response(raw_llm_response);
    let body = parsed.body.trim().to_string();

    // Apply heuristics
    let (override_score, delta, reason) = apply_heuristics(&body);

    // Hard override suppresses unconditionally
    if let Some(0) = override_score {
        tracing::info!(reason, "Response suppressed by heuristic hard-rule");
        return None;
    }

    // Calculate final score
    let a_score = parsed.score.unwrap_or(DEFAULT_SCORE_WHEN_MISSING) as i16;

    // Heuristics can only lower when LLM provided a score
    let effective_delta = if parsed.score.is_some() {
        delta.min(0)
    } else {
        delta
    };

    let final_score = (a_score + effective_delta).clamp(0, 10) as u8;

    if final_score >= DEFAULT_THRESHOLD {
        tracing::debug!(
            final_score,
            a_score = parsed.score.unwrap_or(DEFAULT_SCORE_WHEN_MISSING),
            delta = effective_delta,
            reason,
            "Response passed filter"
        );
        Some(body)
    } else {
        tracing::info!(
            final_score,
            a_score = parsed.score.unwrap_or(DEFAULT_SCORE_WHEN_MISSING),
            delta = effective_delta,
            reason,
            body_preview = &body[..body.len().min(50)],
            "Response suppressed (below threshold)"
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_score_present() {
        let p = parse_scored_response("[SCORE:8] This is a helpful response.");
        assert_eq!(p.score, Some(8));
        assert_eq!(p.body, "This is a helpful response.");
    }

    #[test]
    fn test_parse_score_missing() {
        let p = parse_scored_response("Just a regular response.");
        assert_eq!(p.score, None);
        assert_eq!(p.body, "Just a regular response.");
    }

    #[test]
    fn test_parse_score_out_of_range() {
        let p = parse_scored_response("[SCORE:15] Clamped.");
        assert_eq!(p.score, Some(10));
    }

    #[test]
    fn test_filter_explicit_pass() {
        assert!(filter_response("PASS").is_none());
        assert!(filter_response("[PASS] Not needed.").is_none());
        assert!(filter_response("[SCORE:9] PASS").is_none());
    }

    #[test]
    fn test_filter_generic_phrase() {
        assert!(filter_response("Sounds good").is_none());
        assert!(filter_response("ok").is_none());
        assert!(filter_response("Got it, thanks").is_none());
    }

    #[test]
    fn test_filter_high_score_passes() {
        assert!(filter_response("[SCORE:9] The fix is to add --memory 4g to the docker run command.").is_some());
    }

    #[test]
    fn test_filter_low_score_suppressed() {
        assert!(filter_response("[SCORE:3] Sure, that works.").is_none());
    }

    #[test]
    fn test_filter_code_bonus() {
        let resp = filter_response("[SCORE:6] Try this:\n```bash\ndocker restart moneypenny\n```");
        assert!(resp.is_some()); // 6 + 2 (code bonus) = 8 >= 7
    }

    #[test]
    fn test_filter_missing_score_substantive() {
        // No score prefix, but contains code → default 5 + 2 = 7 → passes
        let resp = filter_response("Here's the fix:\n```\nnpm install\n```");
        assert!(resp.is_some());
    }

    #[test]
    fn test_filter_missing_score_short() {
        // No score, short, no signals → 5 - 2 = 3 → suppressed
        assert!(filter_response("I'll look into it.").is_none());
    }
}
