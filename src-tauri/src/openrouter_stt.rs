//! Fork-only: cloud speech-to-text through OpenRouter's
//! `/audio/transcriptions` endpoint.
//!
//! Kept in its own module so the fork touches upstream files only at a few
//! small hook points (settings fields, the `transcribe()` branch and the
//! recording-start guards in `actions.rs`). The API key is shared with the
//! OpenRouter post-processing provider.

use crate::settings::{self, AppSettings};
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::Deserialize;
use std::time::Duration;
use tauri::AppHandle;

const PROVIDER_ID: &str = "openrouter";
const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_MODEL: &str = "openai/gpt-transcribe";
/// OpenRouter gives the upstream model 60 s; leave room for the upload.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

/// Same 16 kHz mono 16-bit layout as the recordings Handy saves to history.
const SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

/// Whether recordings should be sent to OpenRouter instead of a local model.
pub fn is_enabled(settings: &AppSettings) -> bool {
    settings.openrouter_stt_enabled
}

/// Transcribe 16 kHz mono samples through OpenRouter and return the raw text.
///
/// Blocking: runs the request on a dedicated thread so it is safe to call from
/// both sync code and inside an async task (as `TranscriptionManager::transcribe`
/// is).
pub fn transcribe(settings: &AppSettings, samples: &[f32]) -> anyhow::Result<String> {
    let api_key = settings
        .post_process_api_keys
        .get(PROVIDER_ID)
        .cloned()
        .unwrap_or_default();
    if api_key.trim().is_empty() {
        anyhow::bail!("OpenRouter API key is not set. Add it under Post Processing → OpenRouter.");
    }

    let base_url = settings
        .post_process_providers
        .iter()
        .find(|p| p.id == PROVIDER_ID)
        .map(|p| p.base_url.clone())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let url = format!("{}/audio/transcriptions", base_url.trim_end_matches('/'));
    let body = build_request_body(
        effective_model(settings),
        &encode_wav(samples)?,
        language_hint(&settings.selected_language),
    );

    debug!(
        "Sending {:.1}s of audio to OpenRouter (model: {})",
        samples.len() as f64 / SAMPLE_RATE as f64,
        effective_model(settings)
    );

    std::thread::spawn(move || tauri::async_runtime::block_on(send(url, api_key, body)))
        .join()
        .map_err(|_| anyhow::anyhow!("OpenRouter request thread panicked"))?
}

async fn send(url: String, api_key: String, body: serde_json::Value) -> anyhow::Result<String> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key.trim()))?,
    );

    let response = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(REQUEST_TIMEOUT)
        .build()?
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("OpenRouter request failed: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "OpenRouter transcription failed ({}): {}",
            status,
            error_text.chars().take(500).collect::<String>()
        );
    }

    let parsed: TranscriptionResponse = response
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Unexpected OpenRouter response: {}", e))?;
    Ok(parsed.text.trim().to_string())
}

fn effective_model(settings: &AppSettings) -> &str {
    match settings.openrouter_stt_model.trim() {
        "" => DEFAULT_MODEL,
        model => model,
    }
}

/// ISO-639-1 hint for the API, or `None` to let the model auto-detect.
fn language_hint(selected_language: &str) -> Option<String> {
    match selected_language {
        "" | "auto" => None,
        other => Some(crate::managers::model::canonical_language_code(other).to_string()),
    }
}

fn build_request_body(model: &str, wav: &[u8], language: Option<String>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "input_audio": {
            "data": base64_encode(wav),
            "format": "wav",
        },
    });
    if let Some(language) = language {
        body["language"] = serde_json::Value::String(language);
    }
    body
}

fn encode_wav(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Standard base64 with padding; inlined to avoid a new direct dependency.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[tauri::command]
#[specta::specta]
pub fn change_openrouter_stt_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.openrouter_stt_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_openrouter_stt_model_setting(app: AppHandle, model: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.openrouter_stt_model = model.trim().to_string();
    settings::write_settings(&app, settings);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn wav_has_riff_header_and_pcm16_payload() {
        let wav = encode_wav(&[0.0, 0.5, -0.5, 2.0]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 4 * 2);
    }

    #[test]
    fn request_body_shape() {
        let body = build_request_body("openai/gpt-transcribe", b"foo", Some("ru".into()));
        assert_eq!(body["model"], "openai/gpt-transcribe");
        assert_eq!(body["input_audio"]["data"], "Zm9v");
        assert_eq!(body["input_audio"]["format"], "wav");
        assert_eq!(body["language"], "ru");

        let auto = build_request_body("m", b"", None);
        assert!(auto.get("language").is_none());
    }

    #[test]
    fn language_hint_skips_auto_and_strips_region() {
        assert_eq!(language_hint("auto"), None);
        assert_eq!(language_hint(""), None);
        assert_eq!(language_hint("ru").as_deref(), Some("ru"));
        assert_eq!(language_hint("zh-Hans").as_deref(), Some("zh"));
    }

    #[test]
    fn empty_model_falls_back_to_default() {
        let mut settings = settings::get_default_settings();
        settings.openrouter_stt_model = "  ".into();
        assert_eq!(effective_model(&settings), DEFAULT_MODEL);
        settings.openrouter_stt_model = "mistralai/voxtral-mini-transcribe".into();
        assert_eq!(
            effective_model(&settings),
            "mistralai/voxtral-mini-transcribe"
        );
    }
}
