# gguf2ov

Convert a GGUF checkpoint into uncompressed OpenVINO IR.

## Install

```bash
pip install git+https://github.com/anzr299/gguf2ov.git
```

## Use

```bash
# local file: -- one argument
gguf2ov convert ./Qwen3-4B-UD-IQ3_XXS.gguf -o out/iq3_xxs

# Hub: repo id + exact filename
gguf2ov convert unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-UD-IQ3_XXS.gguf -o out/iq3_xxs
```

`python -m gguf2ov ...` is equivalent to the `gguf2ov` command.

## Commands

**`convert`**

| flag | effect |
|---|---|
| `-o/--out` | output dir (default `gguf2ov-out/<name>`) |
| `--dtype` | `float16` (default), `bfloat16`, `float32` |
| `--keep-intermediate` | keep the torch checkpoint (as large as the IR) |
| `--no-verify` | skip the post-export checks |
| `--device` | OV device |

Writes `<out>/ov/` plus `gguf2ov.json` recording provenance, sizes, timings and parity.

| invocation | meaning |
|---|---|
| `convert ./model.gguf` | in local file the second argument is not needed, and passing it is an error |
| `convert owner/repo model.gguf` | Hub repo id + exact filename |


## Supported Model Architectures

Supported model architectures can be found here https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/integrations/ggml.py

