from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .errors import Gguf2ovError

EPILOG = """\
specifying the GGUF:
  gguf2ov convert ./Qwen3-8B-Q4_K_M.gguf                        local file, one argument
  gguf2ov convert unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf    Hub repo id + exact filename

Name the file exactly; nothing is guessed. transformers resolves local-vs-Hub and downloads
if needed.

Every conversion verifies itself: the IR is checked for compressed weight constants, its
logits are compared against the dequantized torch model, and a greedy generation is run.
Pass --no-verify to skip the last two. Results are recorded in <out>/gguf2ov.json.

The exported IR is uncompressed: it holds exactly the weights llama.cpp would compute with.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gguf2ov",
        description="Convert a GGUF checkpoint into uncompressed OpenVINO IR.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"gguf2ov {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    cv = sub.add_parser("convert", help="GGUF -> torch -> OpenVINO IR",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    cv.add_argument("source", help="path to a local .gguf, or a Hub repo id")
    cv.add_argument("gguf_file", nargs="?",
                    help="exact filename inside the Hub repo; omit when source is a local file")
    cv.add_argument("-o", "--out", help="output directory (default gguf2ov-out/<name>)")
    cv.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="dequantization dtype (default float16)")
    cv.add_argument("--keep-intermediate", action="store_true",
                    help="keep the dequantized torch checkpoint (as large as the IR)")
    cv.add_argument("--no-verify", action="store_true",
                    help="skip the post-export checks: logit parity against the dequantized "
                         "torch model, plus a greedy generation")
    cv.add_argument("--device", default="CPU", help="OV device for the checks")
    return p


def cmd_convert(args) -> int:
    from . import convert as conv

    src = conv.resolve(args.source, args.gguf_file)
    out = Path(args.out) if args.out else Path("gguf2ov-out") / Path(src.gguf_file).stem
    result = conv.convert(src, out, dtype=args.dtype,
                          keep_intermediate=args.keep_intermediate,
                          verify=not args.no_verify, device=args.device)
    print(conv.format_summary(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"convert": cmd_convert}[args.command](args)
    # Only our own errors are reduced to a message. Anything else -- a ValueError from
    # transformers, a RuntimeError from openvino -- is a bug or a broken environment, and is
    # left to propagate so the traceback survives.
    except Gguf2ovError as e:
        print(f"gguf2ov: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
