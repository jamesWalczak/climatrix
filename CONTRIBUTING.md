# Contributing Guidelines

Thank you for contributing! Please follow these rules:

## 🍴 Forks

Fork the repository first.

## 🖊️ Commits

- All commits **must be signed off** using `git commit -s`
- Contributors certify compliance with [Developer Certificate of Origin](https://developercertificate.org/)

## 🌿 Branch Naming

- Use one of the following formats:
  - `f-<ticket_id>` for features (e.g., `f-1234`)
  - `b-<ticket_id>` for bugfixes (e.g., `b-5678`)

## 📦 Pull Requests

- Make sure your PR is linked to an issue or ticket.
- Add a clear description of what it does.
- Don't forget about unit tests.

## 🔧 Pre-commit hooks

This repo uses [pre-commit](https://pre-commit.com/) for code formatting:

- `black` for code style,
- `isort` for import sorting,
- `pre-commit-hook` for checking against `breakpoint()`, private keys, and mixed endline characters,
- `pyupgrade` for upgrading syntax to newer versions of the language,
- `flake8` for checking against PEP8 compliance,
- `mdformat` for Markdown files formatting

To set it up:

```bash
pip install pre-commit
pre-commit install
```
