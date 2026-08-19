"""The errors gguf2ov raises on purpose.

The CLI prints these as a single line and hides the traceback, and lets everything else
escape with its stack intact. That split is the point: an error we defined is a message
for the user, while a ValueError from torch, transformers or openvino is a bug or a broken
environment, and the stack is the only useful thing about it. Catching broad builtins
instead conflates the two -- that is how a packaging.version.InvalidVersion raised three
frames deep inside transformers surfaced as a bare "gguf2ov: Invalid version: 'N/A'" with
nothing to go on.

This module stays import-light on purpose: the CLI needs these names at except-clause
time, and must not drag in torch to get them.
"""

from __future__ import annotations


class Gguf2ovError(Exception):
    """Base class for every error gguf2ov raises deliberately."""


class SourceError(Gguf2ovError):
    """The arguments do not name a readable GGUF checkpoint."""


class UnsupportedArchitecture(Gguf2ovError):
    """The installed transformers cannot dequantize this GGUF's architecture."""


class VerificationError(Gguf2ovError):
    """A post-export check found the exported IR unusable."""


class CompressionDetected(VerificationError):
    """An exported IR turned out to hold compressed weight constants."""
