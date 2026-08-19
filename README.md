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
