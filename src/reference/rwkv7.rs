use std::collections::HashMap;
use tch::{Tensor, Device, Kind, no_grad, IndexOp, nn};
use std::path::Path;

const HEAD_SIZE: i64 = 64;
const DTYPE: Kind = Kind::Half;

#[link(name = "rwkv7_state_fwd_fp16")]
unsafe extern "C" {
    unsafe fn rwkv7_state_fwd_fp16_forward(
        B: i32,
        T: i32,
        C: i32,
        H: i32,
        state: *mut std::ffi::c_void,
        r: *const std::ffi::c_void,
        w: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        y: *mut std::ffi::c_void,
    );
}

//########################################################################################################

// 枚举封装输入
pub enum IdxInput {
    Single(i64),
    Seq(Vec<i64>),
}


// 为了同时支持单个 Tensor 或 Vec<Tensor> 输出，定义一个枚举
pub enum ForwardBatchOutput {
    Single(Tensor),
    Multi(Vec<Tensor>),
}

// 假设你的 state 是类似 `[Tensor; 2]` 或者自定义 struct
pub type State = [Tensor; 2];


// 定义参数结构体
pub struct RWKVArgs {
    pub vocab_size: i64,
    pub head_size: i64,
    pub model_name: String,
}


//########################################################################################################

// 对应 Python 的 WKV_7 系列类
pub struct WKV_7;

impl WKV_7 {
    pub fn forward(state: &Tensor, r: &Tensor, w: &Tensor, k: &Tensor, v: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        no_grad(|| {
            let (t, c) = r.size2().unwrap();  // r: [T, C]
            let h = c / HEAD_SIZE;
            let n = HEAD_SIZE;
            assert_eq!(c, h * HEAD_SIZE);

            for x in &[r, w, k, v, a, b] {
                assert_eq!(x.kind(), DTYPE);
                assert!(x.is_contiguous());
            }

            let device = k.device();
            let y = Tensor::empty(&[t, c], (DTYPE, device));

            y
        })
    }
}

pub fn RWKV7_OP(state: &Tensor, r: &Tensor, w: &Tensor, k: &Tensor, v: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    WKV_7::forward(state, r, w, k, v, a, b)
}

pub struct WKV_7_one;

impl WKV_7_one {
    pub fn forward(state: &Tensor, r: &Tensor, w: &Tensor, k: &Tensor, v: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        no_grad(|| {
            let c = r.size()[0];
            let h = c / HEAD_SIZE;
            let n = HEAD_SIZE;
            assert_eq!(c, h * HEAD_SIZE);

            for x in &[r, w, k, v, a, b] {
                assert_eq!(x.kind(), DTYPE);
                assert!(x.is_contiguous());
            }

            let device = k.device();
            let y = Tensor::empty(&[c], (DTYPE, device));

            y
        })
    }
}

pub fn RWKV7_ONE_OP(state: &Tensor, r: &Tensor, w: &Tensor, k: &Tensor, v: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    WKV_7_one::forward(state, r, w, k, v, a, b)
}

pub struct WKV_7_batch;

impl WKV_7_batch {
    pub fn forward(
        state: &Tensor,
        r: &Tensor,
        w: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
    ) -> Tensor {
        no_grad(|| {
            let (B, T, C) = r.size3().unwrap();
            let H = C / HEAD_SIZE;
            assert_eq!(C, H * HEAD_SIZE);

            for x in &[r, w, k, v, a, b] {
                assert_eq!(x.kind(), DTYPE);
                assert!(x.is_contiguous());
            }

            let device = k.device();
            let mut y = Tensor::empty(&[B, T, C], (DTYPE, device));

            unsafe {
                rwkv7_state_fwd_fp16_forward(
                    B as i32,
                    T as i32,
                    C as i32,
                    H as i32,
                    state.data_ptr() as *mut std::ffi::c_void,
                    r.data_ptr() as *const std::ffi::c_void,
                    w.data_ptr() as *const std::ffi::c_void,
                    k.data_ptr() as *const std::ffi::c_void,
                    v.data_ptr() as *const std::ffi::c_void,
                    a.data_ptr() as *const std::ffi::c_void,
                    b.data_ptr() as *const std::ffi::c_void,
                    y.data_ptr() as *mut std::ffi::c_void,
                );
            }

            y
        })
    }
}

pub fn RWKV7_BATCH_OP(
    state: &Tensor,
    r: &Tensor,
    w: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Tensor {
    WKV_7_batch::forward(state, r, w, k, v, a, b)
}



//########################################################################################################


// RWKV_x070 struct 对应 Python 类
pub struct RWKV_x070 {
    pub args: RWKVArgs,
    pub z: HashMap<String, Tensor>, // 模型权重
    pub n_layer: usize,
    pub n_embd: i64,
    pub n_head: i64,
    pub head_size: i64,
}

impl RWKV_x070 {
    pub fn new(mut args: RWKVArgs) -> Self {
        // 自动选择设备
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        

        // 加载模型权重
        let mut vs = tch::nn::VarStore::new(device);
        vs.load(Path::new(&(args.MODEL_NAME.clone() + ".pth"))).expect("Failed to load model");
        let z: HashMap<String, Tensor> = vs.variables().iter().map(|(k,v)| (k.clone(), v.to_device(device).to_kind(Kind::Float))).collect();
            .expect("Failed to load model")
            .variables()
            .into_iter()
            .map(|(k,v)| (k, v.to_device(device).to_kind(kind::Float)))
            .collect();

        // 设置 head_size
        args.head_size = 64;

        // 自动计算 n_head 和 n_embd
        let r_k = z.get("blocks.0.att.r_k").expect("Missing r_k tensor");
        let (n_head, head_size) = {
            let sizes = r_k.size();
            (sizes[0], sizes[1])
        };
        args.n_embd = n_head * head_size;

        // 检查 head_size 是否匹配
        assert_eq!(head_size, args.head_size, "Head size mismatch");

        // 计算 n_layer
        let mut max_layer: isize = -1;
        for key in z.keys() {
            if key.starts_with("blocks.") {
                let parts: Vec<&str> = key.split('.').collect();
                if let Ok(layer_id) = parts[1].parse::<isize>() {
                    max_layer = max_layer.max(layer_id);
                }
            }
            // 对部分权重做转置（和 Python 一样）
            if key.contains("key.weight") || key.contains("value.weight") || key.contains("receptance.weight") || key.contains("output.weight") || key.contains("head.weight") {
                z.get_mut(key).unwrap().transpose_(0, 1);
            }
            z.get_mut(key).unwrap().contiguous_();
            if key.ends_with("att.r_k") {
                z.get_mut(key).unwrap().flatten_();
            }
        }
        args.n_layer = (max_layer + 1) as usize;

        // 对 embedding 做 LayerNorm
        let ln0_w = z.get("blocks.0.ln0.weight").expect("Missing ln0.weight");
        let ln0_b = z.get("blocks.0.ln0.bias").expect("Missing ln0.bias");
        let emb_w = z.get_mut("emb.weight").unwrap();
        *emb_w = emb_w.layer_norm(&[args.n_embd], Some(ln0_w), Some(ln0_b), 1e-5, false);

        // 复制忽略参数（Python 原逻辑）
        z.insert("blocks.0.att.v0".to_string(), z.get("blocks.0.att.a0").unwrap().shallow_clone());
        z.insert("blocks.0.att.v1".to_string(), z.get("blocks.0.att.a1").unwrap().shallow_clone());
        z.insert("blocks.0.att.v2".to_string(), z.get("blocks.0.att.a2").unwrap().shallow_clone());

        // 构造对象
        RWKV_x070 {
            args,
            z,
            n_layer: args.n_layer,
            n_embd: args.n_embd,
            n_head,
            head_size,
        }
    }

    pub fn generate_zero_state(&self, bsz: i64) -> [Tensor; 2] {
        if bsz >= 1 {
            let state0 = Tensor::zeros(&[self.n_layer as i64, 2, bsz, self.n_embd], (DTYPE, Device::Cuda(0)));
            let state1 = Tensor::zeros(&[self.n_layer as i64, bsz, self.n_embd / HEAD_SIZE, HEAD_SIZE, HEAD_SIZE], (Kind::Float, Device::Cuda(0)));
            (state0, state1)
        } else {
            let state0 = Tensor::zeros(&[self.n_layer as i64, 2, self.n_embd], (DTYPE, Device::Cuda(0)));
            let state1 = Tensor::zeros(&[self.n_layer as i64, self.n_embd / HEAD_SIZE, HEAD_SIZE, HEAD_SIZE], (Kind::Float, Device::Cuda(0)));
            (state0, state1)
        }
    }

    pub fn forward(
        &self,
        idx: IdxInput,        // 自定义枚举，表示单个或多个 token
        state: &mut State,    // 根据你原来的类型
        full_output: bool,
    ) -> Tensor {
        match idx {
            IdxInput::Single(i) => self.forward_one(i, state),
            IdxInput::Seq(ref v) => {
                if v.len() > 1 {
                    self.forward_seq(v, state, full_output)
                } else {
                    self.forward_one(v[0], state)
                }
            }
        }
    }

    pub fn forward_batch(
        &self,
        tokens: Vec<Vec<i64>>,
        state: &mut [Tensor; 2], // state = (state0, state1)
        full_output: bool,
    ) -> ForwardBatchOutput {
        let bsz = tokens.len();
        assert!(bsz > 0);

        let lengths: Vec<usize> = tokens.iter().map(|x| x.len()).collect();
        if lengths.iter().all(|&l| l == lengths[0]) && !full_output {
            return self.forward_batch_same_length(tokens, state, full_output);
        }

        let mut pos = vec![0; bsz];

        // 输出初始化
        let mut out: ForwardBatchOutput = if !full_output {
            ForwardBatchOutput::Single(Tensor::empty(
                &[bsz as i64, self.args.vocab_size],
                (tch::Kind::Float, Device::Cuda(0)),
            ))
        } else {
            ForwardBatchOutput::Multi(
                (0..bsz)
                    .map(|_| Tensor::empty(&[0, self.args.vocab_size], (tch::Kind::Float, Device::Cuda(0))))
                    .collect(),
            )
        };

        loop {
            // 当前活跃的样本
            let active: Vec<usize> = (0..bsz).filter(|&i| pos[i] < lengths[i]).collect();
            if active.is_empty() {
                break;
            }

            // 本次 step 长度
            let step = active
                .iter()
                .map(|&i| lengths[i] - pos[i])
                .min()
                .unwrap();

            // 构造 batch tokens
            let batch_tokens: Vec<Vec<i64>> = active
                .iter()
                .map(|&i| tokens[i][pos[i]..pos[i] + step].to_vec())
                .collect();

            // 构造 batch state，切片类似 Python [:,:,active]
            let active_tensor = Tensor::f_from_slice(&active)
                .expect("Tensor conversion failed")
                .to_device(state.0.device());


            let batch_state = (
                state.0.index_select(2, &active_tensor),
                state.1.index_select(1, &active_tensor),
            );


            // 调用 forward_batch_same_length
            let new_out = self.forward_batch_same_length(batch_tokens, &batch_state, full_output);

            // 更新输出和 state
            for (k, &i) in active.iter().enumerate() {
                match &mut out {
                    ForwardBatchOutput::Single(tensor_out) => {
                        tensor_out.i(i as i64).copy_(&new_out.i(k as i64));
                    }
                    ForwardBatchOutput::Multi(vec_out) => {
                        vec_out[i] = Tensor::cat(&[&vec_out[i], &new_out[k]], 0);
                    }
                }
                state.0.i((.., .., i)).copy_(&batch_state.0.i((.., .., k)));
                state.1.i((.., i)).copy_(&batch_state.1.i((.., k)));
                pos[i] += step;
            }
        }

        out
    }

    pub fn forward_batch_same_length(
        &self,
        tokens: Vec<Vec<i64>>,
        state: &mut [Tensor; 2],
        full_output: bool,
    ) -> ForwardBatchOutput {
        // 检查 tokens 长度一致
        let lengths: Vec<usize> = tokens.iter().map(|x| x.len()).collect();
        let first_len = lengths[0];
        if !lengths.iter().all(|&l| l == first_len) {
            panic!("here all sequences must have the same length");
        }

        // 调用 forward_seq_batch
        ForwardBatchOutput::Single(self.forward_seq_batch(tokens, state, full_output))
    }

    pub fn forward_one(&self, idx: i64, state: &mut [Tensor; 2]) -> Tensor {
        no_grad(|| {
            let z = &self.z;
            let mut x = z.get("emb.weight").unwrap().get(idx);
            let mut v_first = Tensor::empty_like(&x);

            for i in 0..self.n_layer {
                let bbb = format!("blocks.{}.", i);
                let att = format!("blocks.{}.att.", i);
                let ffn = format!("blocks.{}.ffn.", i);

                // Layer Norm 1
                let xx = x.layer_norm(
                    &[self.n_embd],
                    Some(z.get(&(bbb.clone()+"ln1.weight")).unwrap()),
                    Some(z.get(&(bbb.clone()+"ln1.bias")).unwrap()),
                    1e-5,
                    false,
                );

                // Attention Mix
                let get = |name: &str| z.get(&(att.clone() + name)).unwrap();

                let xx_att = RWKV_x070_TMix_one(
                    i as i64, self.n_head as i64, self.head_size as i64, &xx,
                    &state[0].get(i.try_into().unwrap()), &v_first, &state[1].get(i.try_into().unwrap()),
                    get("x_r"), get("x_w"), get("x_k"), get("x_v"), get("x_a"), get("x_g"),
                    get("w0"), get("w1"), get("w2"),
                    get("a0"), get("a1"), get("a2"),
                    get("v0"), get("v1"), get("v2"),
                    get("g1"), get("g2"),
                    get("k_k"), get("k_a"), get("r_k"),
                    get("receptance.weight"), get("key.weight"), get("value.weight"), get("output.weight"),
                    get("ln_x.weight"), get("ln_x.bias")
                );

                x = &x + &xx_att;
                v_first = xx_att.1; // 假设 FFI 返回 (xx_att, v_new)

                // Layer Norm 2
                let xx2 = x.layer_norm(
                    &[self.n_embd],
                    Some(z.get(&(bbb.clone()+"ln2.weight")).unwrap()),
                    Some(z.get(&(bbb.clone()+"ln2.bias")).unwrap()),
                    1e-5,
                    false,
                );

                // FFN Mix
                let xx_ffn = RWKV_x070_CMix_one(
                    &xx2,
                    &state[0].get(i.try_into().unwrap()),
                    z.get(&(ffn.clone()+"x_k")).unwrap(),
                    z.get(&(ffn.clone()+"key.weight")).unwrap(),
                    z.get(&(ffn.clone()+"value.weight")).unwrap(),
                );
                x = &x + &xx_ffn;
            }

            // Final Layer Norm + Head
            let x = x.layer_norm(
                &[self.n_embd],
                Some(z.get("ln_out.weight").unwrap()),
                Some(z.get("ln_out.bias").unwrap()),
                1e-5,
                false,
            );
            let head_weight = z.get("head.weight").unwrap();
            x.matmul(head_weight)
        })
    }

    pub fn forward_one_alt(&self, mut x: Tensor, state: &mut [Tensor; 2]) -> Tensor {
        no_grad(|| {
            let mut v_first = Tensor::empty_like(&x);

            for i in 0..self.n_layer {
                let bbb = format!("blocks.{}.", i);
                let att = format!("blocks.{}.att.", i);
                let ffn = format!("blocks.{}.ffn.", i);

                // LayerNorm 1
                let ln1_weight = self.z.get(&(bbb.clone() + "ln1.weight")).unwrap();
                let ln1_bias = self.z.get(&(bbb.clone() + "ln1.bias")).unwrap();
                let mut xx = x.layer_norm(&[self.n_embd], Some(ln1_weight), Some(ln1_bias), 1e-5, false);

                // Attention Mix
                let (xx_att, new_v_first) = RWKV_x070_TMix_one(
                    i,
                    self.n_head,
                    self.head_size,
                    &xx,
                    &state[0].get(i.try_into().unwrap()),
                    &v_first,
                    &state[1].get(i.try_into().unwrap()),
                    &self.z,
                    &att,
                );
                x = x + xx_att;
                v_first = new_v_first;

                // LayerNorm 2
                let ln2_weight = self.z.get(&(bbb.clone() + "ln2.weight")).unwrap();
                let ln2_bias = self.z.get(&(bbb.clone() + "ln2.bias")).unwrap();
                let xx2 = x.layer_norm(&[self.n_embd], Some(ln2_weight), Some(ln2_bias), 1e-5, false);

                // FFN Mix
                let xx_ffn = RWKV_x070_CMix_one(&xx2, &state[0].get(i.try_into().unwrap()), &self.z, &ffn);
                x = x + xx_ffn;
            }

            // Final LayerNorm + Head
            let ln_out_weight = self.z.get("ln_out.weight").unwrap();
            let ln_out_bias = self.z.get("ln_out.bias").unwrap();
            let x = x.layer_norm(&[self.n_embd], Some(ln_out_weight), Some(ln_out_bias), 1e-5, false);

            let head_weight = self.z.get("head.weight").unwrap();
            x.matmul(head_weight)
        })
    }

    pub fn forward_seq(&self, idx: &[i64], state: &mut [Tensor; 2], full_output: bool) -> Tensor {
        no_grad(|| {
            // idx 是序列索引列表，取 embedding
            let mut x = self.z.get("emb.weight").unwrap().index_select(0, &Tensor::f_from_slice(idx));

            let mut v_first = Tensor::empty_like(&x);

            for i in 0..self.n_layer {
                let bbb = format!("blocks.{}.", i);
                let att = format!("blocks.{}.att.", i);
                let ffn = format!("blocks.{}.ffn.", i);

                // LayerNorm 1
                let ln1_weight = self.z.get(&(bbb.clone() + "ln1.weight")).unwrap();
                let ln1_bias = self.z.get(&(bbb.clone() + "ln1.bias")).unwrap();
                let mut xx = x.layer_norm(&[self.n_embd], Some(ln1_weight), Some(ln1_bias), 1e-5, false);

                // Attention Seq
                let (xx_att, new_v_first) = RWKV_x070_TMix_seq(
                    i,
                    self.n_head,
                    self.head_size,
                    &xx,
                    &state[0].get(i.try_into().unwrap()),
                    &v_first,
                    &state[1].get(i.try_into().unwrap()),
                    &self.z,
                    &att,
                );
                x = x + xx_att;
                v_first = new_v_first;

                // LayerNorm 2
                let ln2_weight = self.z.get(&(bbb.clone() + "ln2.weight")).unwrap();
                let ln2_bias = self.z.get(&(bbb.clone() + "ln2.bias")).unwrap();
                let xx2 = x.layer_norm(&[self.n_embd], Some(ln2_weight), Some(ln2_bias), 1e-5, false);

                // FFN Seq
                let xx_ffn = rwkv_x070_Cmix_seq(
                    &xx2,
                    &state[0].get(i.try_into().unwrap()),
                    &self.z,
                    &ffn,
                );
                x = x + xx_ffn;
            }

            if !full_output {
                x = x.get(-1); // 只保留最后一个时间步
            }

            // Final LayerNorm + Head
            let ln_out_weight = self.z.get("ln_out.weight").unwrap();
            let ln_out_bias = self.z.get("ln_out.bias").unwrap();
            let x = x.layer_norm(&[self.n_embd], Some(ln_out_weight), Some(ln_out_bias), 1e-5, false);

            let head_weight = self.z.get("head.weight").unwrap();
            x.matmul(head_weight)
        })
    }

    pub fn forward_seq_batch(&self, idxs: &[Vec<i64>], state: &mut [Tensor; 2], full_output: bool) -> Tensor {
        no_grad(|| {
            let device = self.z.get("emb.weight").unwrap().device();
            // idxs 转成 Tensor 并取 embedding
            let idx_tensor = Tensor::f_from_slice(&active)
                .expect("Tensor conversion failed")
                .to_device(state.0.device());
            let mut x = self.z.get("emb.weight").unwrap().index_select(0, &idx_tensor.view(-1));

            // reshape x 为 [B, seq_len, C]
            let batch_size = idxs.len();
            let seq_len = idxs[0].len();
            x = x.view([batch_size as i64, seq_len as i64, self.n_embd]);

            let mut v_first = Tensor::empty_like(&x);

            for i in 0..self.n_layer {
                let bbb = format!("blocks.{}.", i);
                let att = format!("blocks.{}.att.", i);
                let ffn = format!("blocks.{}.ffn.", i);

                // LayerNorm 1
                let ln1_weight = self.z.get(&(bbb.clone() + "ln1.weight")).unwrap();
                let ln1_bias = self.z.get(&(bbb.clone() + "ln1.bias")).unwrap();
                let mut xx = x.layer_norm(&[self.n_embd], Some(ln1_weight), Some(ln1_bias), 1e-5, false);

                // Attention Seq Batch
                let (xx_att, new_v_first) = RWKV_x070_TMix_seq_batch(
                    i,
                    self.n_head,
                    self.head_size,
                    &xx,
                    &state[0].get(i.try_into().unwrap()),
                    &v_first,
                    &state[1].get(i.try_into().unwrap()),
                    &self.z.get(&(att.clone()+"x_r")).unwrap(),
                    &self.z.get(&(att.clone()+"x_w")).unwrap(),
                    &self.z.get(&(att.clone()+"x_k")).unwrap(),
                    &self.z.get(&(att.clone()+"x_v")).unwrap(),
                    &self.z.get(&(att.clone()+"x_a")).unwrap(),
                    &self.z.get(&(att.clone()+"x_g")).unwrap(),
                    &self.z.get(&(att.clone()+"w0")).unwrap(),
                    &self.z.get(&(att.clone()+"w1")).unwrap(),
                    &self.z.get(&(att.clone()+"w2")).unwrap(),
                    &self.z.get(&(att.clone()+"a0")).unwrap(),
                    &self.z.get(&(att.clone()+"a1")).unwrap(),
                    &self.z.get(&(att.clone()+"a2")).unwrap(),
                    &self.z.get(&(att.clone()+"v0")).unwrap(),
                    &self.z.get(&(att.clone()+"v1")).unwrap(),
                    &self.z.get(&(att.clone()+"v2")).unwrap(),
                    &self.z.get(&(att.clone()+"g1")).unwrap(),
                    &self.z.get(&(att.clone()+"g2")).unwrap(),
                    &self.z.get(&(att.clone()+"k_k")).unwrap(),
                    &self.z.get(&(att.clone()+"k_a")).unwrap(),
                    &self.z.get(&(att.clone()+"r_k")).unwrap(),
                    &self.z.get(&(att.clone()+"receptance.weight")).unwrap(),
                    &self.z.get(&(att.clone()+"key.weight")).unwrap(),
                    &self.z.get(&(att.clone()+"value.weight")).unwrap(),
                    &self.z.get(&(att.clone()+"output.weight")).unwrap(),
                    &self.z.get(&(att.clone()+"ln_x.weight")).unwrap(),
                    &self.z.get(&(att.clone()+"ln_x.bias")).unwrap()
                );

                x = x + xx_att;
                v_first = new_v_first;

                // LayerNorm 2
                let ln2_weight = self.z.get(&(bbb.clone() + "ln2.weight")).unwrap();
                let ln2_bias = self.z.get(&(bbb.clone() + "ln2.bias")).unwrap();
                let xx2 = x.layer_norm(&[self.n_embd], Some(ln2_weight), Some(ln2_bias), 1e-5, false);

                // FFN Seq Batch
                let xx_ffn = RWKV_x070_CMix_seq_batch(
                    &xx2,
                    &state[0].get(i.try_into().unwrap()),
                    &self.z,
                    &ffn,
                );
                x = x + xx_ffn;
            }

            // full_output=false 时只保留最后时间步
            if !full_output {
                x = x.select(1, -1);
            }

            // Final LayerNorm + Head
            let ln_out_weight = self.z.get("ln_out.weight").unwrap();
            let ln_out_bias = self.z.get("ln_out.bias").unwrap();
            let x = x.layer_norm(&[self.n_embd], Some(ln_out_weight), Some(ln_out_bias), 1e-5, false);

            let head_weight = self.z.get("head.weight").unwrap();
            x.matmul(head_weight)
        })
    }

    pub fn forward_one_batch_alt(
        &self,
        x: &Tensor,
        state: &mut [Tensor; 2], // 假设 state[0] = [Layer][B][C], state[1] = [Layer][B][H][N][N]
        full_output: bool,
    ) -> Tensor {
        no_grad(|| {
            let mut x = x.shallow_clone();
            let mut v_first = Tensor::empty_like(&x);

            for i in 0..self.n_layer {
                let bbb = format!("blocks.{}.", i);
                let att = format!("blocks.{}.att.", i);
                let ffn = format!("blocks.{}.ffn.", i);

                // LayerNorm 1
                let ln1_weight = self.z.get(&(bbb.clone() + "ln1.weight")).unwrap();
                let ln1_bias = self.z.get(&(bbb.clone() + "ln1.bias")).unwrap();
                let mut xx = x.layer_norm(&[self.n_embd], Some(ln1_weight), Some(ln1_bias), 1e-5, false);

                // Attention Seq Batch (Python 那堆参数完全传入)
                let [xx_att, new_v_first] = RWKV_x070_TMix_seq_batch(
                    i,
                    self.n_head,
                    self.head_size,
                    &xx,
                     &mut state,
                    &v_first,
                    &state[1].get(i.try_into().unwrap()),
                    &self.z.get(&(att.clone() + "x_r")).unwrap(),
                    &self.z.get(&(att.clone() + "x_w")).unwrap(),
                    &self.z.get(&(att.clone() + "x_k")).unwrap(),
                    &self.z.get(&(att.clone() + "x_v")).unwrap(),
                    &self.z.get(&(att.clone() + "x_a")).unwrap(),
                    &self.z.get(&(att.clone() + "x_g")).unwrap(),
                    &self.z.get(&(att.clone() + "w0")).unwrap(),
                    &self.z.get(&(att.clone() + "w1")).unwrap(),
                    &self.z.get(&(att.clone() + "w2")).unwrap(),
                    &self.z.get(&(att.clone() + "a0")).unwrap(),
                    &self.z.get(&(att.clone() + "a1")).unwrap(),
                    &self.z.get(&(att.clone() + "a2")).unwrap(),
                    &self.z.get(&(att.clone() + "v0")).unwrap(),
                    &self.z.get(&(att.clone() + "v1")).unwrap(),
                    &self.z.get(&(att.clone() + "v2")).unwrap(),
                    &self.z.get(&(att.clone() + "g1")).unwrap(),
                    &self.z.get(&(att.clone() + "g2")).unwrap(),
                    &self.z.get(&(att.clone() + "k_k")).unwrap(),
                    &self.z.get(&(att.clone() + "k_a")).unwrap(),
                    &self.z.get(&(att.clone() + "r_k")).unwrap(),
                    &self.z.get(&(att.clone() + "receptance.weight")).unwrap(),
                    &self.z.get(&(att.clone() + "key.weight")).unwrap(),
                    &self.z.get(&(att.clone() + "value.weight")).unwrap(),
                    &self.z.get(&(att.clone() + "output.weight")).unwrap(),
                    &self.z.get(&(att.clone() + "ln_x.weight")).unwrap(),
                    &self.z.get(&(att.clone() + "ln_x.bias")).unwrap(),
                );

                x = x + xx_att;
                v_first = new_v_first;

                // LayerNorm 2
                let ln2_weight = self.z.get(&(bbb.clone() + "ln2.weight")).unwrap();
                let ln2_bias = self.z.get(&(bbb.clone() + "ln2.bias")).unwrap();
                let xx2 = x.layer_norm(&[self.n_embd], Some(ln2_weight), Some(ln2_bias), 1e-5, false);

                // FFN Seq Batch
                let xx_ffn = RWKV_x070_CMix_seq_batch(
                    &xx2,
                    &mut state,
                    &self.z.get(&(ffn.clone() + "key.weight")).unwrap(),
                    &self.z.get(&(ffn.clone() + "value.weight")).unwrap(),
                );


                x = x + xx_ffn;
            }

            // full_output = false 时只保留最后时间步
            if !full_output {
                x = x.select(1, -1);
            }

            // Final LayerNorm + Head
            let ln_out_weight = self.z.get("ln_out.weight").unwrap();
            let ln_out_bias = self.z.get("ln_out.bias").unwrap();
            let x = x.layer_norm(&[self.n_embd], Some(ln_out_weight), Some(ln_out_bias), 1e-5, false);

            let head_weight = self.z.get("head.weight").unwrap();
            x.matmul(head_weight)
        })
    }
}



//########################################################################################################




pub fn RWKV_x070_TMix_one(
    layer_id: i64,
    H: i64,
    N: i64,
    x: &Tensor,
    x_prev: &mut [Tensor;2],
    mut v_first: Tensor,
    state: &Tensor,
    x_r: &Tensor, x_w: &Tensor, x_k: &Tensor, x_v: &Tensor, x_a: &Tensor, x_g: &Tensor,
    w0: &Tensor, w1: &Tensor, w2: &Tensor,
    a0: &Tensor, a1: &Tensor, a2: &Tensor,
    v0: &Tensor, v1: &Tensor, v2: &Tensor,
    g1: &Tensor, g2: &Tensor,
    k_k: &Tensor, k_a: &Tensor, r_k: &Tensor,
    R_: &Tensor, K_: &Tensor, V_: &Tensor, O_: &Tensor,
    ln_w: &Tensor, ln_b: &Tensor,) -> [Tensor; 2] {
    no_grad(|| {
        let mut xx = &x_prev[0] - x;
        x_prev[0] = x.shallow_clone();

        let xr = x + &(&xx * x_r);
        let xw = x + &(&xx * x_w);
        let xk = x + &(&xx * x_k);
        let xv = x + &(&xx * x_v);
        let xa = x + &(&xx * x_a);
        let xg = x + &(&xx * x_g);

        let r = xr.matmul(R_);
        let w = xw.matmul(w1).tanh().matmul(w2);
        let mut k = xk.matmul(K_);
        let mut v = xv.matmul(V_);
        let a = (&(&xa.matmul(a1)).matmul(a2) + a0).sigmoid();
        let g = (&xg.matmul(g1)).sigmoid().matmul(g2);

        let kk = k.view([H,N]) * k_k.view([H,N]);
        let kk = &kk / (&kk.norm_scalaropt_dim(2.0, &[-1], true) + 1e-8);
        let kk = kk.view([H*N]);
        k = k * &(1.0 + (&a - 1.0) * k_a);

        if layer_id == 0 { 
            v_first = v.shallow_clone(); 
        } else {
            v = &v + &((v0 + &xv.matmul(&v1).matmul(&v2)).sigmoid() * (&v_first - &v));
        }

        let w = (w + w0).sigmoid();

        // 调用 CUDA kernel 修改 state
        let mut xx_out = RWKV7_ONE_OP(state, &r, &w, &k, &v, &(-&kk), &(kk * &a));

        // group_norm: 使用 view(H*N)，注意 eps
        xx_out = xx_out.view([1,H*N]).group_norm(H as i64, Some(ln_w), Some(ln_b), 64e-5, false).view([H*N]);

        // 加上 r*k*r_k*v 的 residual
        xx_out = &xx_out + &(((&r * &k * r_k).view([H,N]).sum_dim_intlist(&[-1][..], true, Kind::Float) * v.view([H,N])).view([H*N]));

        let output = &xx_out * &g.matmul(O_);
        [output, v_first]
    })
}

pub fn RWKV_x070_TMix_seq(
    layer_id: i64, H: i64, N: i64,
    x: &Tensor,
    x_prev: &mut [Tensor;2],
    mut v_first: Tensor,
    state: &Tensor,
    x_r: &Tensor, x_w: &Tensor, x_k: &Tensor, x_v: &Tensor, x_a: &Tensor, x_g: &Tensor,
    w0: &Tensor, w1: &Tensor, w2: &Tensor,
    a0: &Tensor, a1: &Tensor, a2: &Tensor,
    v0: &Tensor, v1: &Tensor, v2: &Tensor,
    g1: &Tensor, g2: &Tensor,
    k_k: &Tensor, k_a: &Tensor, r_k: &Tensor,
    R_: &Tensor, K_: &Tensor, V_: &Tensor, O_: &Tensor,
    ln_w: &Tensor, ln_b: &Tensor,
) -> [Tensor; 2] {
    no_grad(|| {
        let T = x.size()[0];
        let mut xx = Tensor::cat(&[x_prev[0].unsqueeze(0), x.narrow(0,0,T-1)], 0) - x;
        x_prev[0] = x.narrow(0,T-1,1).squeeze();

        let xr = x + &(&xx * x_r);
        let xw = x + &(&xx * x_w);
        let xk = x + &(&xx * x_k);
        let xv = x + &(&xx * x_v);
        let xa = x + &(&xx * x_a);
        let xg = x + &(&xx * x_g);

        let r = xr.matmul(R_);
        let w = xw.matmul(w1).tanh().matmul(w2);
        let mut k = xk.matmul(K_);
        let mut v = xv.matmul(V_);
        let a = (&(&xa.matmul(a1)).matmul(a2) + a0).sigmoid();
        let g = (&xg.matmul(g1)).sigmoid().matmul(g2);

        let kk = k.view([T,H,N]) * k_k.view([1,H,N]);
        let kk = &kk / (&kk.norm_scalaropt_dim(2.0, &[-1], true) + 1e-8);
        let kk = kk.view([T,H*N]);
        k = k * &(1.0 + (&a-1.0) * k_a);

        if layer_id == 0 { v_first = v.shallow_clone(); }
        else { v = &v + &((v0 + &xv.matmul(&v1).matmul(&v2)).sigmoid() * (&v_first - &v)); }

        let w = (w + w0).sigmoid();

        let mut xx_out = RWKV7_OP(state, &r, &w, &k, &v, &(-&kk), &(kk * &a));

        xx_out = xx_out.view([T,H*N]).group_norm(H as i64, Some(ln_w), Some(ln_b), 64e-5, false).view([T*H*N]);
        xx_out = &xx_out + &(((&r * &k * r_k).view([T,H,N]).sum_dim_intlist(&[-1i64][..], true, Kind::Float) * v.view([T,H,N])).view([T*H*N]));
        let output = &xx_out * &g.matmul(O_);
        [output, v_first]
    })
}

pub fn RWKV_x070_TMix_seq_batch(
    layer_id: i64,
    H: i64,
    N: i64,
    x: &Tensor,          // [B, T, C]
    x_prev: &mut Tensor, // [B, C]
    mut v_first: Tensor, // [B, T, C]
    state: &Tensor,      // [LayerState], 在 CUDA 上原地修改
    x_r: &Tensor, x_w: &Tensor, x_k: &Tensor, x_v: &Tensor, x_a: &Tensor, x_g: &Tensor,

    w0: &Tensor, w1: &Tensor, w2: &Tensor,
    a0: &Tensor, a1: &Tensor, a2: &Tensor,
    v0: &Tensor, v1: &Tensor, v2: &Tensor,
    g1: &Tensor, g2: &Tensor,
    k_k: &Tensor, k_a: &Tensor, r_k: &Tensor,

    R_: &Tensor, K_: &Tensor, V_: &Tensor, O_: &Tensor,

    ln_w: &Tensor, ln_b: &Tensor,
) -> [Tensor; 2] {
    no_grad(|| {
        let (B, T, C) = (x.size()[0], x.size()[1], x.size()[2]);

        // xx = concat(x_prev, x[:,:-1,:]) - x
        let xx_cat = Tensor::cat(&[&x_prev.unsqueeze(1), &x.narrow(1, 0, T-1)], 1);
        let mut xx = &xx_cat - x;
        x_prev.copy_(&x.select(1, T-1)); // x_prev[0] = x[:,-1,:]

        let xr = x + &(&xx * x_r);
        let xw = x + &(&xx * x_w);
        let xk = x + &(&xx * x_k);
        let xv = x + &(&xx * x_v);
        let xa = x + &(&xx * x_a);
        let xg = x + &(&xx * x_g);

        let r = xr.matmul(R_);
        let w = xw.matmul(w1).tanh().matmul(w2);
        let mut k = xk.matmul(K_);
        let mut v = xv.matmul(V_);
        let a = (a0 + xv.matmul(a1).matmul(a2)).sigmoid();
        let g = (xg.matmul(g1)).sigmoid().matmul(g2);

        // kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
        let kk = (&k * k_k).view([B, T, H, N]);

        // 计算 L2 范数，沿最后一维（即 N 维）求 norm
        let norm = kk.norm_scalaropt_dim(2.0, &[-1i64], true);

        // 防止除以零（PyTorch 内部 normalize 也会加 epsilon）
        let kk = &kk / (&norm + 1e-8);

        // reshape 回原来的 [B, T, H*N]
        let kk = kk.view([B, T, H * N]);

        // k = k * (1.0 + (a - 1.0) * k_a)
        k = &k * (1.0 + (&a - 1.0) * k_a);

        if layer_id == 0 {
            v_first = v.shallow_clone();
        } else {
            v = &v + &((v0 + &xv.matmul(&v1).matmul(&v2)).sigmoid() * (&v_first - &v));
        }

        let w = (w0 + w).sigmoid();

        // 调用 CUDA 原地操作
        xx = RWKV7_BATCH_OP(state, &r, &w, &k, &v, &(-&kk), &(&kk * &a));

        xx = xx
            .view([B*T, H*N])
            .group_norm(H as i64, Some(ln_w), Some(ln_b), 64e-5, false)
            .view([B, T, H*N]);

        xx = &xx + &((&r * &k * r_k).view([B, T, H, N]).sum_dim_intlist(&[-1i64][..], true, Kind::Float) * v.view([B, T, H, N])).view([B, T, H*N]);

        let out = &xx * g;
        let out = out.matmul(O_);

        [out, v_first]
    })
}

pub fn RWKV_x070_CMix_one(
    x: &Tensor, x_prev: &mut [Tensor;2], x_k: &Tensor, K_: &Tensor, V_: &Tensor
) -> Tensor {
    no_grad(|| {
        let xx = &x_prev[1] - x;
        x_prev[1] = x.shallow_clone();
        let mut k = x + &(&xx * x_k);
        k = k.relu().pow_tensor_scalar(2.0);
        k.matmul(V_)
    })
}

pub fn RWKV_x070_CMix_seq(
    x: &Tensor, x_prev: &mut [Tensor;2], x_k: &Tensor, K_: &Tensor, V_: &Tensor
) -> Tensor {
    no_grad(|| {
        let xx = Tensor::cat(&[x_prev[1].unsqueeze(0), x.narrow(0,0,x.size()[0]-1)], 0) - x;
        x_prev[1] = x.narrow(0,x.size()[0]-1,1).squeeze();
        let mut k = x + &(&xx * x_k);
        k = k.relu().pow_tensor_scalar(2.0);
        k.matmul(V_)
    })
}

pub fn RWKV_x070_CMix_seq_batch(
    x: &Tensor, x_prev: &mut [Tensor;2], x_k: &Tensor, K_: &Tensor, V_: &Tensor
) -> Tensor {
    no_grad(|| {
        let B = x.size()[0];
        let T = x.size()[1];
        let xx = Tensor::cat(&[x_prev[1].unsqueeze(1), x.narrow(1,0,T-1)], 1) - x;
        x_prev[1].copy_(&x.narrow(1, T-1, 1).squeeze_dim(1));
        let mut k = x + &(&xx * x_k);
        k = k.relu().pow_tensor_scalar(2.0);
        k.matmul(V_)
    })
}