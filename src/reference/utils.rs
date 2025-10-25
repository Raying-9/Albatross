use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::Tensor;

/// Sampling functions
pub fn sampler_simple_batch(logits: &Tensor, noise: f64, temp: f64) -> Tensor {
    assert!(temp > 0.0, "use noise=0 for greedy decoding");
    let mut logits = logits.shallow_clone();
    if temp != 1.0 {
        logits /= temp;
    }
    if noise != 0.0 {
        let noise_tensor = Tensor::rand_like(&logits) * noise;
        logits += noise_tensor;
    }
    logits.argmax(-1, false).unsqueeze(-1)
}

pub fn sampler_simple(logits: &Tensor, noise: f64, temp: f64) -> Tensor {
    assert!(temp > 0.0, "use noise=0 for greedy decoding");
    let mut logits = logits.shallow_clone();
    if temp != 1.0 {
        logits /= temp;
    }
    if noise != 0.0 {
        let noise_tensor = Tensor::rand_like(&logits) * noise;
        logits += noise_tensor;
    }
    logits.argmax(-1, false)
}

/// TRIE structure
pub struct TRIE {
    pub ch: Option<u8>,
    pub to: Vec<Option<Box<TRIE>>>,
    pub values: HashSet<(Vec<u8>, i64)>,
    pub front: Option<*const TRIE>,
}

impl TRIE {
    pub fn new(front: Option<*const TRIE>, ch: Option<u8>) -> Self {
        let mut to = Vec::with_capacity(256);
        for _ in 0..256 {
            to.push(None);
        }

        TRIE {
            ch,
            to,
            values: HashSet::new(),
            front,
        }
    }


    pub fn add(&mut self, key: &[u8], idx: usize, val: Option<(Vec<u8>, i64)>) -> &mut Self {
        if idx == key.len() {
            let value = val.unwrap_or((key.to_vec(), 0));
            self.values.insert(value);
            return self;
        }
        let ch = key[idx] as usize;
        if self.to[ch].is_none() {
            self.to[ch] = Some(Box::new(TRIE::new(Some(self as *const TRIE), Some(key[idx]))));
        }
        self.to[ch].as_mut().unwrap().add(key, idx + 1, val)
    }

    pub fn find_longest(&self, key: &[u8], mut idx: usize) -> (usize, &TRIE, &HashSet<(Vec<u8>, i64)>) {
        let mut u = self;
        let mut ret = (idx, u, &u.values);
        while idx < key.len() {
            let ch = key[idx] as usize;
            if u.to[ch].is_none() {
                break;
            }
            u = u.to[ch].as_ref().unwrap();
            idx += 1;
            if !u.values.is_empty() {
                ret = (idx, u, &u.values);
            }
        }
        ret
    }
}

/// TRIE_TOKENIZER
pub struct TRIE_TOKENIZER {
    idx2token: HashMap<i64, Vec<u8>>,
    token2idx: HashMap<Vec<u8>, i64>,
    root: TRIE,
}

impl TRIE_TOKENIZER {
    pub fn new(file_name: &str) -> Self {
        let mut idx2token: HashMap<i64, Vec<u8>> = HashMap::new();
        idx2token.insert(0, b"<|endoftext|>".to_vec());

        let file = File::open(file_name).expect("failed to open vocab file");
        let reader = BufReader::new(file);
        let mut sorted_tokens: Vec<Vec<u8>> = Vec::new();

        let mut token2idx: HashMap<Vec<u8>, i64> = HashMap::new();

        for line in reader.lines() {
            let l = line.unwrap();
            let first_space = l.find(' ').unwrap();
            let last_space = l.rfind(' ').unwrap();

            let idx: i64 = l[..first_space].parse().unwrap();
            let mut x = &l[first_space..last_space];
            let x = x.trim();
            let bytes = if x.starts_with("b'") || x.starts_with("b\"") {
                let s = &x[2..x.len() - 1];
                s.as_bytes().to_vec()
            } else {
                x.as_bytes().to_vec()
            };

            sorted_tokens.push(bytes.clone());
            idx2token.insert(idx, bytes.clone());
            if idx != 0 {
                token2idx.insert(bytes.clone(), idx);
            }
        }

        let mut root = TRIE::new(None, None);
        for (token, &idx) in &token2idx {
            let _ = root.add(token, 0, Some((token.clone(), idx)));
        }

        TRIE_TOKENIZER {
            idx2token,
            token2idx,
            root,
        }
    }

    pub fn encode_bytes(&self, src: &[u8]) -> Vec<i64> {
        let mut idx = 0;
        let mut tokens = Vec::new();
        while idx < src.len() {
            let (_next_idx, _node, values) = self.root.find_longest(src, idx);
            idx = _next_idx;
            let &(ref _tok_bytes, ref token) = values.iter().next().unwrap();
            tokens.push(*token);
        }
        tokens
    }

    pub fn decode_bytes(&self, tokens: &[i64]) -> Vec<u8> {
        tokens.iter().map(|t| self.idx2token[t].clone()).flatten().collect()
    }

    pub fn encode(&self, src: &str) -> Vec<i64> {
        self.encode_bytes(src.as_bytes())
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        String::from_utf8(self.decode_bytes(tokens)).unwrap_or_else(|_| "<invalid utf8>".to_string())
    }
}
