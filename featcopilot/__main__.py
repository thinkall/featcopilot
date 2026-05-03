"""Enable ``python -m featcopilot`` to dispatch to the CLI."""

from featcopilot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
