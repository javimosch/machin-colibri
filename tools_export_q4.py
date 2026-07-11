"""Export HF llama-arch model to colibri v3 int4 (Q4_0-style, symmetric, group=64).

Layout per quantized tensor: packed nibbles (size/2 bytes; byte i of each
32-byte group holds q[k=i] in the low nibble and q[k=i+32] in the high nibble,
each stored as q+8 in 1..15), then fp32 scales (size/gs). Header = v2's with
magic ak43/version 3. Norms stay fp32, same order as v2.
"""
import struct, sys
import torch
import numpy as np
from export import load_hf_model, serialize_fp32

GS = 64

def quantize_q40(w):
    w = w.detach().float().numpy().reshape(-1, GS)
    wmax = np.abs(w).max(axis=1)
    scale = wmax / 7.0
    scale[scale == 0] = 1e-10
    q = np.clip(np.round(w / scale[:, None]), -7, 7).astype(np.int8)
    err = np.abs(q * scale[:, None] - w).max()
    # split-nibble pack: byte i of a group = (q[i]+8) | (q[i+32]+8)<<4
    qs = (q + 8).astype(np.uint8).reshape(-1, GS)
    packed = (qs[:, :32] | (qs[:, 32:] << 4)).astype(np.uint8)
    return packed.tobytes(), scale.astype(np.float32).tobytes(), err

def main(out_path, hf_path):
    model = load_hf_model(hf_path)
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    weights = [
        model.tok_embeddings.weight,
        *[l.attention.wq.weight for l in model.layers],
        *[l.attention.wk.weight for l in model.layers],
        *[l.attention.wv.weight for l in model.layers],
        *[l.attention.wo.weight for l in model.layers],
        *[l.feed_forward.w1.weight for l in model.layers],
        *[l.feed_forward.w2.weight for l in model.layers],
        *[l.feed_forward.w3.weight for l in model.layers],
    ]
    shared = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared:
        weights.append(model.output.weight)
    f = open(out_path, 'wb')
    f.write(struct.pack('I', 0x616b3433))  # "ak43"
    f.write(struct.pack('i', 3))
    f.write(struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                        n_kv_heads, p.vocab_size, p.max_seq_len))
    f.write(struct.pack('B', int(shared)))
    f.write(struct.pack('i', GS))
    f.write(b'\0' * (256 - f.tell()))
    for l in model.layers:
        serialize_fp32(f, l.attention_norm.weight)
    for l in model.layers:
        serialize_fp32(f, l.ffn_norm.weight)
    serialize_fp32(f, model.norm.weight)
    maxerr = 0.0
    for i, w in enumerate(weights):
        qb, sb, err = quantize_q40(w)
        f.write(qb); f.write(sb)
        maxerr = max(maxerr, err)
        print(f"{i+1}/{len(weights)} q4 {tuple(w.shape)} groupmax-err {err:.5f}")
    f.close()
    print(f"max q4 error: {maxerr:.5f}  wrote {out_path}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
