"""Local web control room for MaiForge training and inference."""

from .server import create_app, main

__all__ = ["create_app", "main"]
