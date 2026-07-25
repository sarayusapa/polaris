# Auto-imported at interpreter startup (site). Backfill sys.get_int_max_str_digits, which
# torch 2.13's _dynamo/polyfills/sys.py requires but Isaac Sim Kit's Python context lacks.
import sys
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda n: None
