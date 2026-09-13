//! dormouse-serve - OpenAI compatible inference engine.
//! Concurrent: model forward is &self (no mutable state), so the model is
//! shared behind Arc and tokio's multi-thread runtime serves many users in
//! parallel. spawn_blocking keeps generation off the async executor threads.
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::SystemTime};

use axum::{extract::State, routing::{get, post}, Json, Router};
use clap::Parser;
use burn::module::Module;
use dormouse_core::DormouseConfig;
use dormouse_core::DormouseModel;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "checkpoints")] ckpt_dir: PathBuf,
    #[arg(long, default_value = "latest", help = "checkpoint file name (<name>.bin)")] ckpt_name: String,
    #[arg(long, default_value = "small")] preset: String,
    #[arg(long, default_value = "8000")] port: u16,
}

#[derive(Clone)]
struct AppState {
    model: Arc<DormouseModel>,
}

// ---------- request/response types (OpenAI wire format) ----------

#[derive(Deserialize)]
struct ChatReq {
    model: Option<String>,
    messages: Vec<ChatMsg>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatMsg {
    #[allow(dead_code)]
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CompReq {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    model: Option<String>,
}

#[derive(Serialize)]
struct ChatResp {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMsgOut,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMsgOut {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompResp {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ModelsResp {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

fn now() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sample_token(logits: &[f32], temp: f32, rng: &mut impl Rng) -> u8 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| ((x - max) / temp).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut r = rng.gen_range(0.0..sum);
    let mut next = 0u8;
    for (i, e) in exp.iter().enumerate() {
        r -= e;
        if r <= 0.0 { next = i as u8; break; }
    }
    if next < 32 || next > 126 { next = 32 + (next % 95); }
    next
}

/// Autoregressive generation, bytes in -> bytes out.
fn generate(model: &DormouseModel, prompt: &str, max_tokens: usize, temp: f32) -> String {
    let mut bytes = prompt.as_bytes().to_vec();
    let mut rng = rand::thread_rng();
    for _ in 0..max_tokens {
        let logits = model.forward_bytes::<dormouse_train::Backend>(&bytes);
        bytes.push(sample_token(&logits, temp, &mut rng));
        if bytes.len() > model.max_seq_len() { break; }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatReq>,
) -> Json<ChatResp> {
    let prompt = req.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let max_tokens = req.max_tokens.unwrap_or(32);
    let temp = req.temperature.unwrap_or(0.8).clamp(0.1, 2.0);
    let model = state.model.clone();
    let prompt_len = prompt.len();
    let text = tokio::task::spawn_blocking(move || generate(&model, &prompt, max_tokens, temp))
        .await.unwrap_or_default();
    let content = text[prompt_len.min(text.len())..].to_string();
    let comp_tokens = content.len();
    Json(ChatResp {
        id: format!("chatcmpl-{}", rand::random::<u32>()),
        object: "chat.completion".into(),
        created: now(),
        model: req.model.unwrap_or_else(|| "dormouse".into()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMsgOut { role: "assistant".into(), content },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: comp_tokens,
            total_tokens: prompt_len + comp_tokens,
        },
    })
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompReq>,
) -> Json<CompResp> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let temp = req.temperature.unwrap_or(0.8).clamp(0.1, 2.0);
    let model = state.model.clone();
    let prompt_len = req.prompt.len();
    let text = tokio::task::spawn_blocking(move || generate(&model, &req.prompt, max_tokens, temp))
        .await.unwrap_or_default();
    let comp = text[prompt_len.min(text.len())..].to_string();
    let comp_tokens = comp.len();
    Json(CompResp {
        id: format!("cmpl-{}", rand::random::<u32>()),
        object: "text_completion".into(),
        created: now(),
        model: req.model.unwrap_or_else(|| "dormouse".into()),
        choices: vec![CompChoice { text: comp, index: 0, finish_reason: "stop".into() }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: comp_tokens,
            total_tokens: prompt_len + comp_tokens,
        },
    })
}

async fn models() -> Json<ModelsResp> {
    Json(ModelsResp {
        object: "list".into(),
        data: vec![
            ModelInfo { id: "dormouse-small".into(), object: "model".into(), created: now(), owned_by: "dormouse".into() },
            ModelInfo { id: "dormouse-base".into(), object: "model".into(), created: now(), owned_by: "dormouse".into() },
            ModelInfo { id: "dormouse-one_b".into(), object: "model".into(), created: now(), owned_by: "dormouse".into() },
        ],
    })
}

#[tokio::main]
async fn main() {
    let a = Args::parse();
    let cfg: DormouseConfig = match a.preset.as_str() {
        "nano" => DormouseConfig::nano(),
        "base" => DormouseConfig::base(),
        "one_b" => DormouseConfig::one_b(),
        _ => DormouseConfig::small(),
    };
    let model = dormouse_train::load_model_weights(&a.ckpt_dir, &a.ckpt_name, cfg)
        .unwrap_or_else(|| {
            eprintln!("ckpt not found: {}/{}.bin", a.ckpt_dir.display(), a.ckpt_name);
            std::process::exit(1);
        });
    println!(
        "dormouse-serve {}/{}.bin preset={} params={} -> http://0.0.0.0:{}/v1 (concurrent, {} runtime threads)",
        a.ckpt_dir.display(), a.ckpt_name, a.preset, model.num_params(), a.port,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    );
    let state = AppState { model: Arc::new(model) };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .with_state(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], a.port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
