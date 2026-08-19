# gguf2ov

Convert a GGUF checkpoint — local or on the Hugging Face Hub — into **uncompressed**
OpenVINO IR.

## Install

```bash
pip install git+https://github.com/anzr299/gguf2ov.git
```

If your environment has pinned versions of torch / transformers / openvino that you do not
want pip to resolve against, install without touching them:

```bash
pip install --no-deps git+https://github.com/anzr299/gguf2ov.git
```

A specific branch, tag or commit:

```bash
pip install git+https://github.com/anzr299/gguf2ov.git@main
```

## Use

```bash
# local file -- one argument
gguf2ov convert ./Qwen3-8B-Q4_K_M.gguf -o out/qwen3-q4k

# Hub repo id + exact filename
gguf2ov convert unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf -o out/q4k
```

`python -m gguf2ov ...` is equivalent to the `gguf2ov` command.

## Layout

```
pyproject.toml
README.md
src/                 <- this directory IS the gguf2ov package
├── __init__.py
├── __main__.py
├── cli.py
└── convert.py
```

`pyproject.toml` maps it with `package-dir = { gguf2ov = "src" }`, so the modules sit directly
in `src/` with no redundant `src/gguf2ov/` nesting, and still install and import as `gguf2ov`.

Because the directory name does not match the package name, the package must be installed to
be importable — there is no run-from-checkout shortcut. For local development:

```bash
git clone https://github.com/anzr299/gguf2ov.git && cd gguf2ov
pip install -e .          # or: pip install -e . --no-deps
```

## Specifying the GGUF

You name the file; nothing is guessed and no repo is searched.

| invocation | meaning |
|---|---|
| `convert ./model.gguf` | local file — the second argument is not needed, and passing it is an error |
| `convert owner/repo model.gguf` | Hub repo id + exact filename |

The arguments are passed straight through as
`from_pretrained(model_id, gguf_file=...)`, which resolves local-vs-Hub itself: a `gguf_file`
that is an existing path is used directly, otherwise it goes through
`cached_file(model_id, gguf_file)`, which downloads and caches
([`modeling_utils.py:771-792`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py)).
There is no download or discovery code here — that would be a second, worse cache in front of
a working one.

## How it works

GGUF weights are dequantized to fp16/bf16, saved as a plain torch checkpoint, then exported
to OV IR.

**This tool performs no weight manipulation of its own.** llama.cpp's converter applies
per-architecture transforms that are not quantization — the Llama-family Q/K row permute
reconciling GGML's `(2i, 2i+1)` RoPE pairing with HF's `(i, i+d/2)` pairing, Gemma's
`norm.weight + 1`, fused-QKV reshapes, MoE expert stacking — and all of those must be
inverted on the way back. `transformers` owns those inverses
(`modeling_gguf_pytorch_utils.py::_reverse_permute_weights` and friends) and this tool
relies on them. Architectures transformers does not recognize are rejected rather than
silently mis-loaded, so the supported set is:

```python
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING; print(sorted(GGUF_CONFIG_MAPPING))
```

## It never compresses

The IR always holds full-precision weights — exactly the values llama.cpp computes with.
There is deliberately no compression flag: a re-quantized IR looks identical to a faithful
one from the outside, and conflating them makes the output useless for the accuracy work
this conversion is actually good for. Run NNCF `compress_weights` on the output as a
separate, explicit step if you want a compressed model, knowing that it re-derives scales
from the dequantized values and discards the GGUF's original grid.

Two guards enforce this:
- `load_in_8bit=False, quantization_config=None` on export, because optimum-intel otherwise
  applies int8 weight compression by itself for models above ~1B params.
- `assert_uncompressed()` reads the finished IR back and fails if any weight `Constant` has
  a low-precision element type (`u4/i4/u8/i8/nf4/f8*/f4e2m1/u2/i2`), so a silent compression
  regression in a dependency cannot slip through.

## The size cost

Dequantizing discards all compression, and that is not a small effect:

| GGUF | file | OV IR | CPU tok/s |
|---|---|---|---|
| Qwen3-8B Q3_K/Q4_K/Q5_K/Q6_K mix | 5.02 GB | 16.38 GB | 6.32 |
| Qwen3-8B IQ2_S/IQ3_XXS/IQ3_S mix | 3.37 GB | 16.40 GB | 6.20 |

A 3.37 GB model and a 5.02 GB model produce IRs within 20 MB of each other, at the same
speed. So this is the right method for **accuracy studies** — the IR reproduces the GGUF's
weights exactly, isolating weight-quantization error from runtime kernel effects — and the
wrong method for deployment. (llama.cpp also quantizes activations to Q8_0 per 32-block in
most kernels, so an fp16 IR scores slightly *better* than llama.cpp on the same file.)

## Commands

**`convert`**

| flag | effect |
|---|---|
| `-o/--out` | output dir (default `gguf2ov-out/<name>`) |
| `--dtype` | `float16` (default), `bfloat16`, `float32` |
| `--keep-intermediate` | keep the torch checkpoint (as large as the IR) |
| `--no-verify` | skip the post-export checks |
| `--device` | OV device for the checks |

Writes `<out>/ov/` plus `gguf2ov.json` recording provenance, sizes, timings and parity.

## Verification is part of converting

Every conversion checks itself, so you find out immediately whether the IR is faithful:

```
[verify] no compressed weight constants in the IR
[parity] 37 tokens: top-1 agreement 100.00%, cosine 0.99999845, max|dlogit| 0.02438
[smoke]  ov greedy: 'The capital of France is Paris. ...'
```

Reference logits are captured while the torch model is still loaded, so parity costs one extra
forward pass rather than a second load. A low score is reported loudly but does **not** fail
the conversion — the threshold is a judgement call, and the IR is written either way. Only the
compression check is fatal, since that is the tool's contract.

Because the IR is uncompressed, top-1 agreement below 99% means a real weight or graph problem
— a missing architecture transform — not precision drift. Measured on Qwen3-8B Q3_K: cosine
0.999998, 100% top-1, max |Δlogit| 0.036.

**Greedy decoded strings are not a valid parity test.** At 100% top-1 agreement a single
near-tie can still flip and cascade over a rollout: an early version of this tool reported a
"mismatch" purely because torch enumerated "Italy, Germany" and OV said "Germany, Italy".

## Gotchas handled for you

1. **optimum-intel silently int8-compresses** models above ~1B params on export. Covered by
   the two guards above.
2. **`save_pretrained` on an OV model does not copy tokenizer files.**
   `AutoTokenizer.from_pretrained(ov_dir)` then returns an empty-vocab tokenizer *with no
   exception*; prompts tokenize to `[]` and generation dies with `IndexError: index -1 is
   out of bounds for dimension 1 with size 0` deep inside transformers' sampling loop.
   `convert` copies the tokenizer and the smoke test asserts a non-empty tokenization.
3. **Split GGUFs** (`-00001-of-00003.gguf`) cannot be loaded by transformers at all. Merge
   them first with `llama-gguf-split --merge` and pass the merged file.

Validated with transformers 5.10.2, gguf 0.19.0, optimum-intel 2.0.0, openvino 2026.2.0.
