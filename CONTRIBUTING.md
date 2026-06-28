# Contributing to ZeroTrust CyberShield XAI
Thanks for your interest in improving ZeroTrust CyberShield XAI! This project is open source
and welcomes contributions of all sizes — bug reports, documentation fixes,
new detectors, dataset adapters, or test improvements.
## Ways to contribute
- **Report a bug** — open an issue describing what you expected and what happened.
- **Suggest a feature** — open an issue describing the use case (see the Roadmap in the README).
- **Improve docs** — typos, clearer examples, and tutorials are all valuable.
- **Write code** — pick an open issue or propose something new, then open a pull request.
## Development setup
```bash
git clone https://github.com/rahman1114/zerotrust-cybershield-xai-_.git
cd zerotrust-cybershield-xai-_
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
## Before you open a pull request
1. **Format & lint**
```bash
   ruff check .
```
2. **Run the tests** — all must pass, and new behaviour should come with tests.
```bash
   pytest
```
3. **Keep changes focused** — one logical change per pull request is easiest to review.
4. **Write a clear description** — explain the *why*, not just the *what*.
## Pull request workflow
1. Fork the repository and create a branch: `git checkout -b feature/my-change`.
2. Make your changes and commit with a descriptive message.
3. Push to your fork and open a pull request against `main`.
4. A maintainer will review; please respond to feedback and keep the branch up to date.
## Code style
- Target Python 3.10+.
- Keep modules small and single-purpose.
- Public functions and classes should have docstrings.
- Prefer clear names over comments, but comment any non-obvious security logic.
## Reporting security issues
If you discover a security vulnerability, please **do not** open a public issue.
Instead, email the maintainer listed in the repository so it can be addressed
responsibly.
## Code of Conduct
By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
