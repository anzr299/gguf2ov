"""GGUF -> dequantized torch checkpoint -> OpenVINO IR.

This tool never compresses. It dequantizes the GGUF and exports the weights as-is, so the
IR holds exactly the values llama.cpp would compute with. If you want a compressed OV
model, run NNCF compress_weights on the output yourself -- that is a separate decision with
its own accuracy cost, and folding it in here would make a faithful conversion
indistinguishable from a re-quantized one.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from . import __version__

SMOKE_PROMPT = "The capital of France is"

# Longer and more varied than the smoke prompt, to give the logit comparison something to
# disagree about if a weight transform is wrong.
VERIFY_TEXT = (
    "The capital of France is Paris. Machine learning models are often quantized to reduce "
    "their memory footprint, trading a small amount of accuracy for speed and a smaller file "
    "on disk."
)

# Below this top-1 logit agreement, an uncompressed IR is considered broken rather than drifty.
TOP1_THRESHOLD = 0.99

# Element types that can only appear in an IR whose weights were compressed.
COMPRESSED_TYPES = {"u4", "i4", "u8", "i8", "nf4", "f8e4m3", "f8e5m2", "f4e2m1", "u2", "i2"}


def _gb(nbytes: int | float | None) -> str:
    return "unknown" if not nbytes else f"{nbytes / 1e9:.2f} GB"


def dirsize(path: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


class CompressionDetected(RuntimeError):
    """Raised when an exported IR turns out to hold compressed weight constants."""


class Source(NamedTuple):
    """Exactly the pair transformers wants: from_pretrained(model_id, gguf_file=...).

    transformers resolves local-vs-Hub for gguf_file itself, downloading and caching as
    needed, so there is nothing to discover or fetch here.
    """

    model_id: str        # local directory or Hub repo id
    gguf_file: str       # exact filename within it
    size: int | None     # bytes, known only for local files

    def describe(self) -> str:
        joined = Path(self.model_id) / self.gguf_file
        return str(joined) if joined.is_file() else f"{self.model_id} :: {self.gguf_file}"


def resolve(source: str, gguf_file: str | None = None) -> Source:
    """Either `source` alone is the path to a local .gguf, or `source` is a Hub repo id and
    `gguf_file` names the file inside it."""
    path = Path(source).expanduser()
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False

    if is_file:
        # A local file needs no second argument; accepting one too would silently become
        # from_pretrained("./model.gguf", gguf_file="model.gguf") -> a bogus joined path.
        if gguf_file is not None:
            raise ValueError(
                f"{source!r} is already a .gguf file, so drop the second argument:\n"
                f"  gguf2ov convert {source}"
            )
        return Source(str(path.parent), path.name, path.stat().st_size)

    if gguf_file is None:
        raise FileNotFoundError(
            f"{source!r} is not a readable file. Either give the path to a local .gguf, or "
            f"give a Hub repo id followed by the exact filename:\n"
            f"  gguf2ov convert ./Qwen3-8B-Q4_K_M.gguf\n"
            f"  gguf2ov convert unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf"
        )
    return Source(source, gguf_file, None)


@dataclass
class ConvertResult:
    source: str
    arch: str
    dtype: str
    gguf_bytes: int | None
    torch_bytes: int
    ov_bytes: int
    dequantize_s: float
    export_s: float
    torch_dir: str | None
    ov_dir: str | None
    smoke_output: str | None = None
    parity: dict | None = None


def _cached_size(src: Source) -> int | None:
    """Size of the GGUF transformers just downloaded. Pure local cache lookup, no network."""
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(src.model_id, src.gguf_file)
        return Path(hit).stat().st_size if isinstance(hit, str) else None
    except Exception:  # noqa: BLE001 - a missing size is cosmetic
        return None


class DequantOutput(NamedTuple):
    seconds: float
    arch: str
    greedy: str | None
    verify_ids: dict | None      # tokenizer output the OV side must be fed
    ref_logits: object | None    # torch logits for verify_ids, kept for the parity check


def dequantize(src: Source, out_dir: Path, dtype_str: str = "float16",
               verify: bool = True) -> DequantOutput:
    """Dequantize a GGUF into a plain torch checkpoint at `out_dir`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dequant] {src.gguf_file} ({_gb(src.size)}) -> {dtype_str}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        src.model_id, gguf_file=src.gguf_file, dtype=dtype
    )
    tok = AutoTokenizer.from_pretrained(src.model_id, gguf_file=src.gguf_file)
    elapsed = time.time() - t0
    arch = getattr(model.config, "model_type", "?")

    nparam = sum(p.numel() for p in model.parameters())
    print(f"[dequant] {elapsed:.1f}s, arch {arch}, {nparam / 1e9:.2f} B params "
          f"({_gb(nparam * dtype.itemsize)} dense)")

    greedy, verify_ids, ref_logits = None, None, None
    if verify:
        model.eval()
        ids = tok(SMOKE_PROMPT, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=24, do_sample=False)
        greedy = tok.decode(gen[0], skip_special_tokens=True)
        print(f"[dequant] torch greedy: {greedy!r}")

        verify_ids = dict(tok(VERIFY_TEXT, return_tensors="pt"))
        with torch.no_grad():
            ref_logits = model(**verify_ids).logits.float()

    print(f"[dequant] saving -> {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    del model
    return DequantOutput(elapsed, arch, greedy, verify_ids, ref_logits)


def assert_uncompressed(ov_dir: Path) -> None:
    """Fail if the exported IR contains compressed weight constants.
    optimum intel compresses to int8 in cases where model is very big. 
    Need to ensure it is in full precision
    """
    import openvino as ov

    model = ov.Core().read_model(str(Path(ov_dir) / "openvino_model.xml"))
    offenders: dict[str, int] = {}
    for node in model.get_ops():
        if node.get_type_name() != "Constant":
            continue
        et = node.get_element_type().get_type_name()
        if et in COMPRESSED_TYPES:
            offenders[et] = offenders.get(et, 0) + 1

    if offenders:
        raise CompressionDetected(
            f"exported IR at {ov_dir} holds compressed constants {offenders}. gguf2ov must "
            "produce uncompressed weights; the export path quantized behind our back "
            "(check that load_in_8bit=False is still honoured by the installed "
            "optimum-intel)."
        )
    print("[verify] no compressed weight constants in the IR")


def export_ov(torch_dir: Path, ov_dir: Path) -> float:
    """Export a torch checkpoint to OV IR in full precision."""
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    ov_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] {torch_dir} -> {ov_dir}")
    t0 = time.time()
    # load_in_8bit=False is required: optimum-intel otherwise applies int8 weight
    # compression on its own for models above ~1B params, so the export would quietly come
    # back quantized. quantization_config=None makes the intent explicit.
    model = OVModelForCausalLM.from_pretrained(
        str(torch_dir), export=True, load_in_8bit=False, quantization_config=None,
    )
    model.save_pretrained(ov_dir)
    # save_pretrained does not copy tokenizer files. Without this,
    # AutoTokenizer.from_pretrained(ov_dir) returns an EMPTY-vocab tokenizer with no
    # exception, every prompt tokenizes to [], and generate() fails deep inside
    # transformers with "index -1 is out of bounds for dimension 1 with size 0".
    AutoTokenizer.from_pretrained(str(torch_dir)).save_pretrained(ov_dir)
    elapsed = time.time() - t0
    print(f"[export] {elapsed:.1f}s, IR = {_gb(dirsize(ov_dir))}")
    assert_uncompressed(ov_dir)
    return elapsed


def verify_parity(ov_dir: Path, deq: DequantOutput, device: str = "CPU") -> dict | None:
    """Compare the IR's logits against the dequantized torch model's, on the same tokens.

    A low score is reported loudly but does not fail the conversion, since the
    threshold is a judgement call. Note that greedy decoded text is NOT a valid check here --
    at 100% top-1 agreement a single near-tie can still flip and cascade over a rollout.
    """
    import torch
    from optimum.intel import OVModelForCausalLM

    if deq.verify_ids is None or deq.ref_logits is None:
        return None

    model = OVModelForCausalLM.from_pretrained(str(ov_dir), device=device)
    got = model(**deq.verify_ids).logits.float()
    del model

    ref = deq.ref_logits
    diff = (got - ref).abs()
    top1 = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(0, 1), ref.flatten(0, 1), dim=-1).mean().item()

    print(f"[parity] {ref.shape[1]} tokens: top-1 agreement {top1 * 100:.2f}%, "
          f"cosine {cos:.8f}, max|dlogit| {diff.max():.5f}")
    if top1 < TOP1_THRESHOLD:
        print(f"[parity] WARNING: below {TOP1_THRESHOLD * 100:.0f}%. The IR is uncompressed, so "
              "this is unlikely to be precision drift -- suspect a missing architecture "
              "transform (RoPE permute, norm offset, QKV split). The IR was still written.")
    return {
        "tokens": int(ref.shape[1]),
        "top1_agreement": round(top1, 6),
        "cosine": round(cos, 8),
        "max_abs_logit_diff": round(diff.max().item(), 5),
        "mean_abs_logit_diff": round(diff.mean().item(), 6),
    }


def smoke_test_ov(ov_dir: Path, device: str = "CPU", max_new_tokens: int = 24) -> str:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    print(f"[smoke] compiling IR on {device}")
    t0 = time.time()
    model = OVModelForCausalLM.from_pretrained(str(ov_dir), device=device)
    tok = AutoTokenizer.from_pretrained(str(ov_dir))
    print(f"[smoke] compiled in {time.time() - t0:.1f}s")

    ids = tok(SMOKE_PROMPT, return_tensors="pt")
    if ids["input_ids"].shape[1] == 0:
        raise RuntimeError(
            f"tokenizer in {ov_dir} produced 0 tokens -- its vocab files are missing or "
            "empty, so the IR directory is not self-sufficient"
        )
    t0 = time.time()
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - t0
    text = tok.decode(gen[0], skip_special_tokens=True)
    ntok = gen.shape[1] - ids["input_ids"].shape[1]
    print(f"[smoke] ov greedy: {text!r}")
    print(f"[smoke] {ntok} tokens in {dt:.1f}s = {ntok / dt:.2f} tok/s on {device}")
    return text


def convert(src: Source, out_dir: Path, *, dtype: str = "float16",
            keep_intermediate: bool = False, verify: bool = True,
            device: str = "CPU") -> ConvertResult:
    out_dir = Path(out_dir)
    torch_dir = out_dir / "torch"
    ov_dir = out_dir / "ov"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[source] {src.describe()}")
    deq = dequantize(src, torch_dir, dtype, verify=verify)
    torch_bytes = dirsize(torch_dir)
    # For a Hub source the size is unknown until transformers has cached the file.
    gguf_bytes = src.size if src.size else _cached_size(src)

    export_s = export_ov(torch_dir, ov_dir)
    ov_bytes = dirsize(ov_dir)

    parity, ov_out = None, None
    if verify:
        parity = verify_parity(ov_dir, deq, device)
        ov_out = smoke_test_ov(ov_dir, device)

    if not keep_intermediate:
        print(f"[cleanup] removing intermediate torch checkpoint ({_gb(torch_bytes)})")
        shutil.rmtree(torch_dir)

    result = ConvertResult(
        source=src.describe(), arch=deq.arch, dtype=dtype, gguf_bytes=gguf_bytes,
        torch_bytes=torch_bytes, ov_bytes=ov_bytes, dequantize_s=round(deq.seconds, 1),
        export_s=round(export_s, 1),
        torch_dir=str(torch_dir) if keep_intermediate else None,
        ov_dir=str(ov_dir),
        smoke_output=ov_out or deq.greedy,
        parity=parity,
    )
    manifest = out_dir / "gguf2ov.json"
    manifest.write_text(json.dumps(
        {"gguf2ov_version": __version__, "compressed": False, **asdict(result)},
        indent=2) + "\n")
    print(f"[done] manifest -> {manifest}")
    return result


def format_summary(r: ConvertResult) -> str:
    def ratio(n: int) -> str:
        return f"   ({n / r.gguf_bytes:.2f}x gguf)" if r.gguf_bytes else ""

    lines = ["", "=" * 64,
             f"source        {r.source}",
             f"architecture  {r.arch}",
             f"gguf          {_gb(r.gguf_bytes)}"]
    if r.torch_bytes:
        lines.append(f"torch {r.dtype:<8s}{_gb(r.torch_bytes)}{ratio(r.torch_bytes)}")
    if r.ov_bytes:
        lines.append(f"ov ir {r.dtype:<8s}{_gb(r.ov_bytes)}{ratio(r.ov_bytes)}"
                     f"   uncompressed")
    if r.parity:
        lines.append(f"parity        top-1 {r.parity['top1_agreement'] * 100:.2f}%, "
                     f"cosine {r.parity['cosine']:.6f} vs the dequantized torch model")
    lines.append(f"timing        dequantize {r.dequantize_s:.1f}s, export {r.export_s:.1f}s")
    lines.append(f"output        {r.ov_dir}")
    lines.append("=" * 64)
    return "\n".join(lines)
