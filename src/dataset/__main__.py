"""Allow ``python -m dataset`` to run the dataset CLI."""

from .cli import main

raise SystemExit(main())
