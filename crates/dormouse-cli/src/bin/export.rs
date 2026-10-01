//! dormouse export - a training checkpoint becomes a standalone inference
//! artifact: no optimizer section, no `.ngram` sidecar, weights narrowed.
//!
//! `dormouse export --info <file>` prints what a file IS without loading it,
//! for someone handed one 15 MB file on a machine that has never seen this
//! repository: the format, the dtype, the model shape, the config, the
//! parameter count, and whether the checksum is intact.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use dormouse_train::export::{self, DType};

#[derive(Parser, Debug)]
#[command(about = "export a training checkpoint as a standalone inference file")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Write `<ckpt-dir>/<ckpt-name>.bin` as an inference export.
    Run {
        #[arg(long, default_value = "checkpoints")]
        ckpt_dir: PathBuf,
        #[arg(
            long,
            default_value = "latest",
            help = "checkpoint file name (<name>.bin)"
        )]
        ckpt_name: String,
        /// Weight format: bf16 | f16 | f32. See README "The model file".
        #[arg(long, default_value = "bf16")]
        dtype: String,
        /// Output path. Defaults to `<ckpt-dir>/<ckpt-name>.<dtype>.dmexp`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only used when the run left no `<ckpt-name>.config.toml` next to the
        /// checkpoint. A preset is a guess about a shape; the snapshot is not.
        #[arg(long, default_value = "small")]
        preset: String,
        #[arg(long)]
        config: Option<String>,
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Describe an export, and verify its checksum.
    Info { path: PathBuf },
}

fn die(e: String) -> ! {
    eprintln!("export: {e}");
    std::process::exit(1);
}

fn main() {
    let a = Args::parse();
    match a.cmd {
        Cmd::Run {
            ckpt_dir,
            ckpt_name,
            dtype,
            out,
            preset,
            config,
            set,
        } => {
            let dtype = DType::parse(&dtype).unwrap_or_else(|e| die(e));
            let ckpt = ckpt_dir.join(format!("{ckpt_name}.bin"));
            let out = out.unwrap_or_else(|| {
                ckpt_dir.join(format!(
                    "{ckpt_name}.{}.dmexp",
                    format!("{dtype:?}").to_lowercase()
                ))
            });
            let preset = config.as_deref().unwrap_or(&preset);
            let r = export::export_ckpt(&ckpt, Some(preset), &set, dtype, &out)
                .unwrap_or_else(|e| die(e));
            let kb = r.file_bytes as f64 / 1024.0;
            println!("export: {} ({dtype:?})", r.path);
            println!("  params      {}", r.num_params);
            println!("  size        {} B ({kb:.1} KiB)", r.file_bytes);
            if r.source_bytes > 0 {
                println!(
                    "  from        {} B training checkpoint ({:.1}x smaller)",
                    r.source_bytes,
                    r.source_bytes as f64 / r.file_bytes as f64
                );
            }
            println!("  max |w|     {:.6e}", r.max_abs);
            println!("  min |w|>0   {:.6e}", r.min_nonzero_abs);
            println!(
                "  flushed     {} (below the format's smallest normal, now 0)",
                r.flushed
            );
            println!("  load with   generate --export {} ...", r.path);
        }
        Cmd::Info { path } => {
            let (h, bytes) = export::inspect(&path).unwrap_or_else(|e| die(e));
            println!("file      {} ({bytes} B)", path.display());
            println!("format    DMEXPRT v{}", h.format);
            println!("dtype     {:?}", h.dtype);
            println!("source    {}", h.source);
            println!("step      {}", h.step);
            println!("params    {} in {} tensors", h.num_params, h.num_tensors);
            println!("max |w|   {:.6e}", h.max_abs);
            println!("min |w|>0 {:.6e}", h.min_nonzero_abs);
            println!("flushed   {}", h.flushed);
            println!("checksum  ok (CRC-32 over the payload)");
            println!("config    ---");
            print!("{}", h.config);
        }
    }
}
