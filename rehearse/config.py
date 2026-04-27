"""Environment-backed settings.

Loads secrets and runtime configuration from environment variables (or a .env
file in dev). Nothing in the rest of the package reads os.environ directly.
"""
