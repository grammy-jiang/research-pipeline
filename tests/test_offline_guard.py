"""Opt-in pytest plugin: block external network access before DNS or HTTP.

Run with PYTHONPATH=tests:src pytest -p test_offline_guard tests/unit.
Use independent temporary source-budget state for the test session.
"""

import os
import socket
import tempfile
from urllib.parse import urlsplit

import requests

_LOCAL = {"localhost", "127.0.0.1", "::1"}
_ORIGINAL_DNS = socket.getaddrinfo
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_SEND = requests.adapters.HTTPAdapter.send
_STATE = None
_PREVIOUS_BUDGET = None


class ExternalNetworkBlocked(BaseException):
    """Fail a test even when provider code catches Exception."""


def _dns(host, *args, **kwargs):
    if host not in _LOCAL and host is not None:
        raise ExternalNetworkBlocked("External DNS/network is disabled in unit tests")
    return _ORIGINAL_DNS(host, *args, **kwargs)


def _connect(sock, address):
    if isinstance(address, tuple) and address[0] not in _LOCAL:
        raise ExternalNetworkBlocked("External socket is disabled in unit tests")
    return _ORIGINAL_CONNECT(sock, address)


def _send(adapter, request, *args, **kwargs):
    if urlsplit(request.url).hostname not in _LOCAL:
        raise ExternalNetworkBlocked("External HTTP is disabled in unit tests")
    return _ORIGINAL_SEND(adapter, request, *args, **kwargs)


def pytest_sessionstart(session):
    global _STATE, _PREVIOUS_BUDGET
    _STATE = tempfile.TemporaryDirectory(prefix="research-pipeline-test-budgets-")
    _PREVIOUS_BUDGET = os.environ.get("RESEARCH_PIPELINE_REQUEST_STATE_DIR")
    os.environ["RESEARCH_PIPELINE_REQUEST_STATE_DIR"] = _STATE.name
    socket.getaddrinfo = _dns
    socket.socket.connect = _connect
    requests.adapters.HTTPAdapter.send = _send


def pytest_sessionfinish(session, exitstatus):
    socket.getaddrinfo = _ORIGINAL_DNS
    socket.socket.connect = _ORIGINAL_CONNECT
    requests.adapters.HTTPAdapter.send = _ORIGINAL_SEND
    if _PREVIOUS_BUDGET is None:
        os.environ.pop("RESEARCH_PIPELINE_REQUEST_STATE_DIR", None)
    else:
        os.environ["RESEARCH_PIPELINE_REQUEST_STATE_DIR"] = _PREVIOUS_BUDGET
    if _STATE:
        _STATE.cleanup()
