"""Train a composed geo2wf experiment."""

from __future__ import annotations


def main() -> None:
    # The root module remains the compatibility entry point and now understands
    # Hydra overrides. Keeping one runtime preserves run-directory/DDP behavior.
    from geo2wf.training import _entrypoint

    _entrypoint()
