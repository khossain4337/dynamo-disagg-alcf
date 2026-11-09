# `vllm bench serve` with a 48 h request timeout instead of the hardcoded 6 h (serve.py:870).
# Usage: python bench_serve_long_timeout.py bench serve <args>
import sys

import aiohttp

_init = aiohttp.ClientSession.__init__


def _init_long(self, *args, **kwargs):
    # Only the request session passes a timeout; the /tokenize and /v1/models sessions keep the default.
    if "timeout" in kwargs:
        kwargs["timeout"] = aiohttp.ClientTimeout(total=172800)
    _init(self, *args, **kwargs)


aiohttp.ClientSession.__init__ = _init_long

from vllm.entrypoints.cli.main import main  # noqa: E402

sys.argv[0] = "vllm"
sys.exit(main())
