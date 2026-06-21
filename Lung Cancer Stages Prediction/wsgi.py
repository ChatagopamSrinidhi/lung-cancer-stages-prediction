#!/usr/bin/env python
"""WSGI entry point for production deployment using Waitress."""

from app import app

if __name__ == '__main__':
    # Run with Waitress (pure Python, no external dependencies)
    from waitress import serve
    print("Starting Lung Cancer Detection Server on http://0.0.0.0:8000")
    serve(app, host='0.0.0.0', port=8000, threads=4)
