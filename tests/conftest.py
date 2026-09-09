"""Pytest session setup.

The API now defaults to SCREENING_BACKEND=matlab (PS SIH26038 requires the
MATLAB pipeline). The existing contract tests in test_api.py were written for
the reference PyTorch pipeline and run without a local MATLAB install, so pin
the backend to "python" for the test session. Set SCREENING_BACKEND explicitly
in the environment to override (e.g. to run the same contract tests against the
MATLAB backend on a machine that has it).
"""

from __future__ import annotations

import os

os.environ.setdefault("SCREENING_BACKEND", "python")
