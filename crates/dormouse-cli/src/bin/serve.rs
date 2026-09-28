//! dormouse serve - OpenAI compatible API, concurrent (model forward is &self)
//!
//! It reads an INFERENCE EXPORT (`.dmexp`), never a training checkpoint: see
//! `generate.rs` for why that refusal is the feature. Pointing this at a
//! training checkpoint (or its 34 GB `.ngram` sidecar) prints why it will not
//! and exits; it never train-shaped-loads the optimizer section.
use axum::{extract::Json, routing::{get, post}, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, help = "inference export written by `dormouse export` (<name>.dmexp)")] export: PathBuf,
    #[arg(long, default_value = "8000")] port: u16,
}

#[derive(Clone)]
struct AppState {
    model: Arc<dormouse_core::DormouseModel>,
    /// What is actually loaded. `/v1/models` used to answer with four hardcoded
    /// names of models that may not be the one on the GPU, which is the
    /// wrong-but-plausible answer ADR-0011 calls the cardinal sin.
    model_id: String,
}

#[derive(Deserialize)]
struct ChatReq {
    model: Option<String>,
    messages: Vec<ChatMsg>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMsg {
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
    message: ChatMsg,
    finish_reason: String,
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

fn generate(model: &dormouse_core::DormouseModel, prompt: &str, max_tokens: usize, temp: f32) -> String {
    use rand::Rng;
    let mut bytes = prompt.as_bytes().to_vec();
    let mut rng = rand::thread_rng();
    for _ in 0..max_tokens {
        let logits = dormouse_train::decode::next_byte_logits::<dormouse_train::Backend>(model, &bytes);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| ((x - max)/temp).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut r = rng.gen_range(0.0..sum);
        let mut next = 0u8;
        for (i, e) in exp.iter().enumerate() {
            r -= e;
            if r <= 0.0 { next = i as u8; break; }
        }
        if next < 32 || next > 126 { next = 32 + (next % 95); }
        bytes.push(next);
        if bytes.len() > model.max_seq_len() { break; }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

async fn chat_completions(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<ChatReq>,
) -> Json<ChatResp> {
    let prompt = req.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let max_tokens = req.max_tokens.unwrap_or(32);
    let temp = req.temperature.unwrap_or(0.8);
    let model = state.model.clone();
    let prompt_len = prompt.len();
    // LOUD, not `unwrap_or_default()`: a panicking decode task used to become
    // an EMPTY string, i.e. a 200 carrying a confident empty completion. The
    // decode seam asserts on an inert memory arm and on a failed logit
    // readback, so those two checks were being answered with a valid-looking
    // response and a line in the log nobody reads (ADR-0019).
    let text = tokio::task::spawn_blocking(move || generate(&model, &prompt, max_tokens, temp))
        .await
        .expect("serve: the decode task panicked - refusing to answer with an empty completion");
    let content = text[prompt_len..].to_string();
    let prompt_tokens = prompt_len;
    let comp_tokens = content.len();
    Json(ChatResp {
        id: format!("chatcmpl-{}", rand::random::<u32>()),
        object: "chat.completion".into(),
        created: 0,
        model: req.model.unwrap_or("dormouse".into()),
        choices: vec![ChatChoice { index: 0, message: ChatMsg { role: "assistant".into(), content }, finish_reason: "stop".into() }],
        usage: Usage { prompt_tokens, completion_tokens: comp_tokens, total_tokens: prompt_tokens+comp_tokens },
    })
}

async fn completions(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<CompReq>,
) -> Json<CompResp> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let temp = req.temperature.unwrap_or(0.8);
    let model = state.model.clone();
    let prompt = req.prompt.clone();
    let text = tokio::task::spawn_blocking(move || generate(&model, &prompt, max_tokens, temp))
        .await
        .expect("serve: the decode task panicked - refusing to answer with an empty completion");
    let comp = text[req.prompt.len()..].to_string();
    let prompt_tokens = req.prompt.len();
    let comp_tokens = comp.len();
    Json(CompResp {
        id: format!("cmpl-{}", rand::random::<u32>()),
        object: "text_completion".into(),
        created: 0,
        model: req.model.unwrap_or("dormouse".into()),
        choices: vec![CompChoice { text: comp, index: 0, finish_reason: "stop".into() }],
        usage: Usage { prompt_tokens, completion_tokens: comp_tokens, total_tokens: prompt_tokens+comp_tokens },
    })
}

async fn models(axum::extract::State(state): axum::extract::State<AppState>) -> Json<ModelsResp> {
    Json(ModelsResp {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".into(),
            created: 0,
            owned_by: "dormouse".into(),
        }],
    })
}

#[tokio::main]
async fn main() {
    let a = Args::parse();
    // One load path, and it is the export. The config is not a flag here: it
    // travels inside the file, so the served model cannot be a preset the
    // operator mistyped (ADR-0011: a wrong-but-plausible answer is the sin).
    let (model, cfg, h) = dormouse_train::export::read(&a.export)
        .unwrap_or_else(|e| { eprintln!("serve: {e}"); std::process::exit(1); });
    // LOUD before the socket is bound: a --engram-ram model has its memory in
    // a .ngram sidecar this export does not ship, so serving it would answer
    // every request from a memory arm that never ran.
    dormouse_train::decode::refuse_unservable_memory(&cfg)
        .unwrap_or_else(|e| { eprintln!("serve: {e}"); std::process::exit(1); });
    println!(
        "loaded {} ({:?} weights, {} params, step {}, d_model {}, vocab {})",
        a.export.display(), h.dtype, h.num_params, h.step, cfg.d_model, cfg.vocab
    );
    let state = AppState {
        model: Arc::new(model),
        model_id: a
            .export
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dormouse")
            .to_string(),
    };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .with_state(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], a.port));
    println!("dormouse openai serve {} -> http://{}/v1 (concurrent)", a.export.display(), addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
