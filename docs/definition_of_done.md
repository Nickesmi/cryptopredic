# Definition of Done

> Every GitHub issue in this repository is considered **complete** only when
> all items in this checklist are satisfied. No exceptions.

---

## Checklist

### Code Quality

- [ ] **Compiles** — `python -c "import src"` runs without error
- [ ] **No placeholder code** — no `TODO`, `FIXME`, `HACK`, `pass` (in non-abstract methods), or `raise NotImplementedError` in concrete implementations
- [ ] **No hardcoded secrets** — no API keys, passwords, or tokens in source files
- [ ] **No magic numbers** — all constants live in `src/config/settings.py`

### Tests

- [ ] **Unit tests written** — every new public function/class has at least one test
- [ ] **Tests pass** — `pytest tests/ -q` exits 0
- [ ] **No network in tests** — all tests use synthetic data or mocks; no live API calls
- [ ] **Existing tests untouched** — no tests removed or weakened

### Static Analysis

- [ ] **Lint passes** — `flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503` exits 0
- [ ] **Type annotations present** — all new functions have full type signatures

### Documentation

- [ ] **Docstrings updated** — every modified or new function/class has a complete docstring
- [ ] **`docs/` updated** — architecture, modeling, or API docs updated if behaviour changed
- [ ] **`.env.example` updated** — any new environment variables are documented with defaults
- [ ] **README updated** — Quick Start or module list updated if the public surface changed

### Architecture

- [ ] **Clean Architecture respected** — no inner layer imports from outer layer
- [ ] **Dependency Injection** — no concrete third-party classes instantiated inside business logic
- [ ] **SOLID principles** — no violations (SRP, OCP, LSP, ISP, DIP)
- [ ] **No duplicated code** — DRY; shared logic extracted into utilities or base classes

### Performance & Safety

- [ ] **No O(n²) over large datasets** — or justified with a comment if unavoidable
- [ ] **Backward compatibility** — existing public interfaces unchanged, or old behaviour versioned
- [ ] **Errors handled** — all exception paths logged and propagated cleanly

### Commits

- [ ] **Atomic commits** — each commit is a single logical change
- [ ] **Commit messages** follow Conventional Commits format (`feat/fix/test/docs:`)
- [ ] **Closing comment** on the GitHub issue summarises what was built

---

## Verification Commands

Run these locally before marking an issue done:

```bash
# 1. Import check
python -c "import src; print('Import OK')"

# 2. Unit tests
python -m pytest tests/ -q --tb=short

# 3. Lint
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# 4. Search for TODOs
grep -r "TODO\|FIXME\|HACK" src/ && echo "Found TODOs — fix before closing issue"

# 5. Search for hardcoded secrets (basic check)
grep -rn "api_key\s*=\s*['\"]" src/ || echo "No hardcoded keys found"
```

All commands must exit cleanly before the issue is closed.
