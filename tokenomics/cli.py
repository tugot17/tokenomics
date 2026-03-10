"""Thin CLI dispatcher for tokenomics subcommands."""

import sys


SUBCOMMANDS = {
    "completion": ("tokenomics.completion_benchmark", "main"),
    "embedding": ("tokenomics.embedding_benchmark", "main"),
    "plot-completion": ("tokenomics.plot_completion_benchmark", "main"),
    "plot-embedding": ("tokenomics.plot_embedding_benchmark", "main"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: tokenomics <subcommand> [args ...]\n")
        print("subcommands:")
        for name in SUBCOMMANDS:
            print(f"  {name}")
        print("\nRun 'tokenomics <subcommand> --help' for subcommand help.")
        sys.exit(0)

    name = sys.argv[1]
    if name not in SUBCOMMANDS:
        print(f"error: unknown subcommand '{name}'")
        print(f"available: {', '.join(SUBCOMMANDS)}")
        sys.exit(1)

    # Strip the subcommand from argv so argparse in each module sees the right args
    sys.argv = [f"tokenomics {name}"] + sys.argv[2:]

    module_path, func_name = SUBCOMMANDS[name]
    from importlib import import_module
    mod = import_module(module_path)
    getattr(mod, func_name)()
