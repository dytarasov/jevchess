#!/bin/sh
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q chess certifi; }
exec .venv/bin/python server.py
