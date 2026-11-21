# Contribute

Agent guide (hard limits + agentic rules): [AGENTS.md](https://github.com/sertit/sertit-utils/blob/main/AGENTS.md).
Technical reference (modules, build & test, CI/CD, pitfalls): [ARCHITECTURE.md](https://github.com/sertit/sertit-utils/blob/main/ARCHITECTURE.md).
AI-assisted contribution policy: [AI_POLICY.md](https://github.com/sertit/sertit-utils/blob/main/AI_POLICY.md).

## Quick start

1. **Find or open an issue** with a consensus on the solution (see [Report issues](#report-issues)).
2. **Set up a dev environment** ([Submit patches](#submit-patches)).
3. **Make a small, focused change**, scoped to the issue.
4. **Run the checks**: `pre-commit run --all-files` must pass, plus the relevant tests if you have the test data (otherwise CI validates for you).
5. **Open a PR against `main`** with the [pull request template](https://github.com/sertit/sertit-utils/blob/main/.github/pull_request_template.md), a `CHANGES.md` entry, and a commit message per [the conventions below](#commit-message-conventions).

## Report issues

Issue tracker: https://github.com/sertit/sertit-utils/issues

Check that a similar issue does not already exist, then report:

- What you expected to happen, with a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) if possible.
- What actually happened, with the full traceback if there was an exception.
- Your Python and Sertit-Utils versions.
- For issues involving a specific file or product, its name and type. Never attach licensed data.

## Submit patches

```bash
conda env create -f environment.yml
conda activate sertit
pip install -e .
pre-commit install
```

or with plain pip: `pip install -r requirements.txt` then `pip install -e .`.

We use `pre-commit` (`ruff` check + format, JSON/YAML/XML and large-file checks), and we run the exact same hooks in CI, so install it:

```bash
pre-commit run --all-files
```

The hooks must pass before you submit a PR. If a hook fails, fix the issue; do not skip or bypass the hook.
Please be sure that your code is running on Python 3.11+.

### Copyright header

Every Python file starts with the Apache 2.0 header:

```python
# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

The year is the **current release year**, changed project-wide at release time only (a `DOC:` commit).
Do not update it in unrelated files; new files use the year as it appears in the other headers.

### Running the test suite

The full test suite reads data from SERTIT infrastructure (S3 or the local DB3, not redistributed), so most contributors cannot run it locally; CI takes care of that.

Two suites run anywhere:

```bash
python -m pytest ci/minimal --noconftest          # base-only imports
python -m pytest ci/test_optional_imports.py     # the import-failure rules
```

If you do have the test data, run the module suites (`CI_SERTIT_USE_S3=0` to read it from disk):

```bash
python -m pytest ci/test_<module>.py
```

## Commit message conventions

Commit prefixes in use (the EOReader ones, plus `DEPR:` for deprecations):
`FIX:` bug fix, `ENH:` enhancement, `OPTIM:` performance, `CI:` CI changes, `DEPS:` dependencies, `DOC:` documentation, `LINT:` linting/formatting, `TST:` tests, `DEPR:` deprecation.

Message structure:

- A subject line starting with the prefix, under 80 characters.
- One blank line, then optionally a body.
- Reference the related GitHub issue in the subject or body: `[#242](https://github.com/sertit/sertit-utils/issues/242)`.

Example:

```text
FIX: Handle chained extensions finishing by .gz in `path.get_ext` [#XX](https://github.com/sertit/sertit-utils/issues/XX)
```

## Pull requests

- Branch from `main`, open the PR against `main`, follow the [pull request template](https://github.com/sertit/sertit-utils/blob/main/.github/pull_request_template.md).
- Cover the change with tests when possible, and note it in `CHANGES.md` with the right prefix.
- Keep PRs small and focused.
- New features and behavior changes need a linked issue with a consensus on the solution.
- New functions respect the optional-dependency contract: modules stay importable with a base-only install, optional needs guarded with `@requires`
  ([ARCHITECTURE.md § Key patterns](https://github.com/sertit/sertit-utils/blob/main/ARCHITECTURE.md#key-patterns-for-contributors)).

## AI-Assisted Contributions

**The preferred path is human-written code**, with AI in a supporting role (finding bugs, suggesting tests, reviewing, improving documentation).
**AI-generated code is accepted when it is carefully reviewed**: you must be able to explain and defend every line you submit.

- **You are the author and fully accountable**, regardless of the tools used.
- **Be transparent**: write the PR description yourself and disclose material AI assistance in it.
- **No AI authorship or advertising**, and **agents must not act on your behalf** in project spaces without your explicit instruction.
- **Follow the pull request template**, including its *AI-assisted review* section.

Do not use AI to speak for you in issues and reviews (translation or grammar editing is fine);
human-to-human communication is essential for an open source community to thrive.

The full policy is the [EOReader AI policy](https://github.com/sertit/eoreader/blob/main/AI_POLICY.md),
applied here by [AI_POLICY.md](https://github.com/sertit/sertit-utils/blob/main/AI_POLICY.md).

## Release Sertit-Utils

Releases are made by tagging a commit on `main`, as in EOReader:

* Update `README.md` and `CHANGES.md`.
* Update the version in `sertit/__meta__.py` (`__version__`).
* Push local `main`, then create a GitHub release with a `v*` tag: this triggers the PyPI publish workflow (`.github/workflows/publish.yml`), and conda-forge packaging follows automatically.
