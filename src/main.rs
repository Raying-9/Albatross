use std::fs::File;
use std::io::Write;
use std::path::Path;
mod reference;
use ndarray::Array2;
use regex::Regex;
use tch::{nn, nn::ModuleT, Tensor, Device};



use reference::rwkv7::RWKVArgs;
use reference::rwkv7::RWKV_x070;
use reference::utils::{TRIE_TOKENIZER, sampler_simple_batch};

fn main() -> anyhow::Result<()> {
    let prompt = "User: Evaluate $(1+2i)6-3i$.\n\nAssistant: <think";
    const BATCH_SIZE: i64 = 4;
    const GENERATION_LENGTH: i64 = 100;
    const DECODE_NOISE: f64 = 1.0;
    const DECODE_TEMP: f64 = 0.5;

    
    // 初始化参数
    let args= RWKVArgs {
        vocab_size: 65536,
        head_size: 64,
        model_name: "/mnt/d/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096".to_string(),
    };

    
    // 获取设备
    let device = Device::cuda_if_available();


    let mut log_file = File::create("rollout.log")?;

    let model = RWKV_x070::new(args);
    let tokenizer = TRIE_TOKENIZER::new("reference/rwkv_vocab_v20230424.txt");

    // init state
    let mut state = model.generate_zero_state(BATCH_SIZE);
    let mut out = model.forward_batch(vec![tokenizer.encode(prompt); BATCH_SIZE], &mut state, true);

    let mut all_out: Vec<Vec<i64>> = Vec::new();

    println!("rollout {} tokens...", GENERATION_LENGTH);

    for i in 0..GENERATION_LENGTH {
        let batch_logits = extract_logits_for_batch(&out, 0);
        let token = sampler_simple_batch(&batch_logits, DECODE_NOISE, DECODE_TEMP);

        // 转成 Vec<Vec<i64>> 给 forward_batch
        let token_vec = vec![token.iter::<i64>().unwrap().collect::<Vec<i64>>()];

        // 保存生成的 token
        all_out.push(token_vec[0].clone());

        // forward_batch 调用
        out = model.forward_batch(token_vec, &mut state, true);

        if i % 10 == 0 {
            print!("{} ", i);
        }
    }

    println!("\n{}", "#".repeat(120));// 用于收集每个时间步的 token
        
    let mut all_out: Vec<Vec<i64>> = vec![];

    // 生成循环：按时间步收集 token
    for _t in 0..GENERATION_LENGTH {
        // 每个 batch 当前时间步的 token
        let token_batch: Vec<i64> = (0..BATCH_SIZE)
            .map(|b| {
                // 从 ForwardBatchOutput 中取出第 b 个 batch 的 token
                out[b].int64_value(&[0])
            })
            .collect();

        all_out.push(token_batch);

        // 更新 out：forward_batch 需要 Vec<Vec<i64>> 并加上 full_output 参数
        out = model.forward_batch(vec![token_batch.clone(); BATCH_SIZE], &mut state, true);
    }

    // Python 的 transpose 等效操作
    // Rust 中直接构造 [BATCH_SIZE][GENERATION_LENGTH]
    let mut all_out_t: Vec<Vec<i64>> = vec![vec![0i64; GENERATION_LENGTH]; BATCH_SIZE];
    for (t, token_row) in all_out.iter().enumerate() {
        for (b, &tok) in token_row.iter().enumerate() {
            all_out_t[b][t] = tok;
        }
    }

    // 按 batch 输出
    for n in 0..BATCH_SIZE {
        let tokens = &all_out_t[n];

        // 找到第一个 0 作为 eod
        let eod_pos = tokens.iter().position(|&x| x == 0).unwrap_or(tokens.len());
        let tokens_trimmed = &tokens[..eod_pos];

        let out_str = tokenizer.decode(tokens_trimmed);
        writeln!(log_file, "{}\n{}", out_str, "#".repeat(120))?;
        println!("{}", out_str);
    }
    Ok(())
}
