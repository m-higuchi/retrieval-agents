"""Utilities for Ollama."""


def run_ollama() -> int:
    """Run Ollama.

    Returns:
        int: PID
    """
    import subprocess

    process = subprocess.Popen(
        ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return process.pid
