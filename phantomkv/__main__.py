import sys

from .cli import main

# Guard required on macOS: vLLM's EngineCore is started with `spawn`, which
# re-imports this module as __mp_main__; unguarded, it would re-run the CLI.
if __name__ == "__main__":
    sys.exit(main())
