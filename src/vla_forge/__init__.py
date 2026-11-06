"""vla-forge: type what a robot should do; agents train a vision-language-action policy for it."""


def main() -> None:
    from .server import main as serve

    serve()
