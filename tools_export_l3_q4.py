"""Export Llama-3.2-1B-Instruct to colibri v4 int8 + tiktoken tokenizer.

v4 checkpoint = v2 (ak42/Q8_0) with an extended header (magic ak44):
  u32 magic 0x616b3434, i32 version=4,
  7x i32: dim hidden layers heads kv_heads vocab seq_len(CLAMPED),
  u8 shared_classifier, i32 group_size,
  f32 rope_theta, u8 llama3_scaling, f32 factor, f32 low_freq_factor,
  f32 high_freq_factor, i32 original_ctx,  pad to 256.
Weights: fp32 norms (att per layer, ffn per layer, final), then Q8_0 tensors
(tokens, wq*, wk*, wv*, wo*, w1*, w2*, w3*, [wcls if untied]) — v2 order.

tokenizer-l3.bin: i32 n_vocab(=128256), then per id: i32 len + raw bytes
(specials >= 128000 stored as their literal names).
"""
import struct, sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

GS = 64
SEQ_CLAMP = 4096

def bytes_to_unicode():
    # GPT-2 byte<->unicode table (transformers convention for tokenizer strings)
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\xa1"), ord("\xac")+1)) + list(range(ord("\xae"), ord("\xff")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip([chr(c) for c in cs], bs))

def quantize_q80(w):
    # int4 symmetric, split-nibble packing (matches colibri dot_q4): per GS group,
    # 32 bytes; byte k = (q[k]+8) | (q[k+32]+8)<<4, q in [-8,7].
    w = w.detach().float().numpy().reshape(-1, GS)
    wmax = np.abs(w).max(axis=1)
    scale = wmax / 7.0
    scale[scale == 0] = 1e-10
    q = np.round(w / scale[:, None]).clip(-8, 7).astype(np.int8)
    err = np.abs(q * scale[:, None] - w).max()
    qs = (q + 8).astype(np.uint8)               # 0..15, shape (rows, GS)
    half = GS // 2
    packed = (qs[:, :half] | (qs[:, half:] << 4)).astype(np.uint8)  # (rows, GS/2)
    return packed.tobytes(), scale.astype(np.float32).tobytes(), err

def serialize_fp32(f, t):
    f.write(t.detach().float().numpy().astype(np.float32).tobytes())

def main(repo, out_bin, out_tok):
    print("loading", repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    cfg = model.config
    sd = model.state_dict()
    dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head = dim // n_heads
    rs = getattr(cfg, 'rope_parameters', None) or getattr(cfg, 'rope_scaling', None) or {}
    theta = float(rs.get('rope_theta', getattr(cfg, 'rope_theta', 10000.0)))
    print("cfg:", dim, cfg.intermediate_size, cfg.num_hidden_layers, n_heads, n_kv, cfg.vocab_size, "theta", theta, "scaling", rs)

    def permrev(w, nh, d1):
        return w.view(nh, 2, d1 // nh // 2, dim).transpose(1, 2).reshape(d1, dim)

    emb = sd['model.embed_tokens.weight']
    tied = 'lm_head.weight' not in sd or torch.equal(sd.get('lm_head.weight', emb), emb)
    weights = [emb]
    L = cfg.num_hidden_layers
    for pfx, per in [('self_attn.q_proj', 'q'), ('self_attn.k_proj', 'k'), ('self_attn.v_proj', None),
                     ('self_attn.o_proj', None), ('mlp.gate_proj', None), ('mlp.down_proj', None), ('mlp.up_proj', None)]:
        for i in range(L):
            w = sd[f'model.layers.{i}.{pfx}.weight']
            if per == 'q': w = permrev(w, n_heads, dim)
            if per == 'k': w = permrev(w, n_kv, head * n_kv)
            weights.append(w)
    if not tied:
        weights.append(sd['lm_head.weight'])

    f = open(out_bin, 'wb')
    f.write(struct.pack('I', 0x616b3435))  # "ak45" (int4 + rope header)
    f.write(struct.pack('i', 4))
    f.write(struct.pack('iiiiiii', dim, cfg.intermediate_size, L, n_heads, n_kv,
                        cfg.vocab_size, min(cfg.max_position_embeddings, SEQ_CLAMP)))
    f.write(struct.pack('B', int(tied)))
    f.write(struct.pack('i', GS))
    f.write(struct.pack('f', theta))
    is_l3 = 1 if rs.get('rope_type', rs.get('type', '')) == 'llama3' else 0
    f.write(struct.pack('B', is_l3))
    f.write(struct.pack('fff', float(rs.get('factor', 1.0)), float(rs.get('low_freq_factor', 1.0)),
                        float(rs.get('high_freq_factor', 1.0))))
    f.write(struct.pack('i', int(rs.get('original_max_position_embeddings', cfg.max_position_embeddings))))
    f.write(b'\0' * (256 - f.tell()))
    for i in range(L):
        serialize_fp32(f, sd[f'model.layers.{i}.input_layernorm.weight'])
    for i in range(L):
        serialize_fp32(f, sd[f'model.layers.{i}.post_attention_layernorm.weight'])
    serialize_fp32(f, sd['model.norm.weight'])
    maxerr = 0.0
    for i, w in enumerate(weights):
        qb, sb, err = quantize_q80(w)
        f.write(qb); f.write(sb)
        maxerr = max(maxerr, err)
        if i % 20 == 0: print(f"{i+1}/{len(weights)} quantized")
    f.close()
    print(f"tied={tied} max q8 err {maxerr:.5f} wrote {out_bin}")

    # tokenizer: id -> raw bytes
    tok = AutoTokenizer.from_pretrained(repo)
    u2b = bytes_to_unicode()
    vocab = tok.get_vocab()  # token-string (GPT2-unicode) -> id
    n = cfg.vocab_size
    table = [b''] * n
    for t, i in vocab.items():
        if i >= n: continue
        if i >= 128000:
            table[i] = t.encode('utf-8')  # special: literal name
        else:
            table[i] = bytes(u2b[ch] for ch in t)
    tf = open(out_tok, 'wb')
    tf.write(struct.pack('i', n))
    for b in table:
        tf.write(struct.pack('i', len(b)))
        tf.write(b)
    tf.close()
    print("wrote", out_tok)
    # sanity: HF encodes for a probe string (specials off)
    probe = "Hello world! The 3 quick brown foxes' jump."
    print("HF probe ids:", tok.encode(probe, add_special_tokens=False))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else "unsloth/Llama-3.2-1B-Instruct",
         sys.argv[2] if len(sys.argv) > 2 else "/home/jarancibia/ai/machin-colibri/models/llama32-1b-q80.bin",
         sys.argv[3] if len(sys.argv) > 3 else "/home/jarancibia/ai/machin-colibri/models/tokenizer-l3.bin")
