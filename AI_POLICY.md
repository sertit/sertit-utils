# AI Policy

> [!NOTE]
> In this policy, "AI" means generative AI tools (LLMs and coding agents) that write or modify code and produce human-like text.

**Sertit-Utils applies the [EOReader AI policy](https://github.com/sertit/eoreader/blob/main/AI_POLICY.md) as its reference:**
responsibility, disclosure, communication, authorship, licensing, large contributions,
the recommended workflow, and the handling of non-compliant contributions.

The only adaptations, where the EOReader wording refers to EOReader specifics:

- **Code quality invariants**: optional-dependency contract (every module imports with a base-only install, optional needs guarded with `@requires` in `sertit/utils.py`), lazy raster and dask functions, minimal diffs (see [ARCHITECTURE.md](https://github.com/sertit/sertit-utils/blob/main/ARCHITECTURE.md)).
- **Typical AI failure modes**: scope creep and unrelated refactors, hallucinated functions or keyword arguments, over-verbose code and comments, breaking the base install with a module-level hard import, missing edge-case tests, fixes that address the symptom rather than the cause.
- **Validation**: `pre-commit run --all-files`, and when test data is available `python -m pytest ci/test_<relevant>.py`.
  Otherwise the PR declares CI to be the validation path (the test data lives on SERTIT infrastructure, in S3 or the local DB3).
- **Agents**: they must not act in Sertit-Utils's project spaces (GitHub `sertit/sertit-utils`, SERTIT GitLab) **without human approval**, per the [hard limits in AGENTS.md](https://github.com/sertit/sertit-utils/blob/main/AGENTS.md).

Any difference not listed here follows the EOReader policy, so the two are reviewed together.

_Last reviewed: November 2026._
