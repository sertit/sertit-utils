### Your checklist for this pull request
Please review the [guidelines for contributing](../CONTRIBUTING.md) and the
[AI-assisted contribution policy](../AI_POLICY.md) to this repository.

- [ ] Check the commit's or even all commits' message styles matches our requested structure.
- [ ] Check your code additions will fail neither code linting checks nor unit test.
- [ ] Check `CHANGES.md` contains an entry with the right prefix (`FIX:`, `ENH:`, `OPTIM:`, `CI:`, `DEPS:`, `DOC:`, `LINT:`, `TST:`, `DEPR:`) — *or mark n/a for mechanical changes and say why*.
- [ ] The change is scoped to the issue it resolves (no unrelated changes) — *or explain in the description if no issue applies*.

### AI-assisted review

If AI assistance was material for this PR (and only then), please complete
this section. See [AI_POLICY.md](../AI_POLICY.md) for the policy (the EOReader one,
with the Sertit-Utils adaptations) and the recommended workflow.

- [ ] I personally reviewed all AI-generated code, tests, and documentation, and I understand them.
- [ ] I ran the project checks (`pre-commit run --all-files`, targeted `pytest`) and report the results in the description below.
- [ ] This PR does not attribute authorship to any AI tool or service.
- [ ] I disclosed AI assistance in the PR description (if applicable).

**Summary (a few sentences, in your own words — this is sufficient evidence of
review):**

- What AI generated, and what you accepted/rejected and why.
- What you manually verified (e.g., the import rules in `ci/expected_imports.py`).
- Which checks you ran (e.g., `pre-commit run --all-files`,
  `pytest ci/test_rasters.py`) — or state that CI is the validation
  path and why (e.g., no local test data).

> 

For mechanical PRs (dependency bump, pre-commit autoupdate, typo), write
"n/a" in this section.
### Description
Please describe your pull request in your own words.

### Further comments
If this is a relatively large or complex change, kick off the discussion by explaining why you chose the solution you did and what alternatives you considered, etc...

Thank you!
