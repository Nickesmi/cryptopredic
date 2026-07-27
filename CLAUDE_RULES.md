# AI Coding Constitution

> These rules govern every code generation action taken by an AI assistant
> (Claude, Gemini, Copilot, or any other model) working in this repository.
> They exist to prevent regressions, maintain quality, and ensure the codebase
> stays production-ready across many automated commits.
>
> **Before writing a single line of code, the AI must read and apply every rule
> in this document.**

---

## Pre-Coding Checklist (Mandatory Design Review)

Before writing any code for an issue, answer these questions in a plan:

1. **Which modules will change?** List every file that will be modified or created.
2. **Will this violate SOLID?** Check each principle explicitly.
3. **Does this duplicate existing code?** Search the codebase first.
4. **Can this be implemented as an extension?** Prefer adding over modifying.
5. **Are tests required?** (Answer is always: yes.)
6. **Are docs affected?** Update `docs/` and docstrings if yes.
7. **Does this touch the public API?** If yes, maintain backward compatibility.

Only after answering these questions should code generation begin.

---

## Rule 1 — Scope Control

- **Never modify files unrelated to the current issue.**
- If a fix requires changing an unrelated file, create a separate issue for it.
- Only touch the minimum number of files needed to implement the feature.
- Do not "clean up" unrelated code while implementing a feature.

---

## Rule 2 — No Placeholder Implementations

Every function, class, and module written must be **production-ready**:

```python
# ✅ Correct — real implementation
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    ...

# ❌ Wrong — placeholder
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    # TODO: implement RSI
    return df
```

No `pass`, no `TODO`, no `raise NotImplementedError` in non-abstract methods,
no `return None` where a value is expected.

---

## Rule 3 — Never Remove Tests

- Tests must never be deleted unless the feature they test is explicitly removed.
- If a test fails after a change, **fix the code**, not the test (unless the
  test itself was wrong about expected behaviour).
- Every new function must have at least one new test.

---

## Rule 4 — Use Typed Python

All Python code must use type annotations on every function signature:

```python
# ✅ Correct
def train(
    self,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]: ...

# ❌ Wrong — untyped
def train(self, X, y, feature_names=None): ...
```

Use `from __future__ import annotations` at the top of every file.
Target `mypy --strict` compliance.

---

## Rule 5 — Prefer Composition Over Inheritance

Use dependency injection and protocol-based composition rather than deep
inheritance hierarchies. Only use inheritance for abstract base classes
(ABCs) that define contracts.

```python
# ✅ Correct — composition
class ForecastPipeline:
    def __init__(self, repo: PriceRepository, model: ForecastModel): ...

# ❌ Avoid — inheritance for implementation reuse
class BitcoinForecaster(XGBoostForecaster):  # fragile
    ...
```

---

## Rule 6 — Always Write Production-Ready Code

Write code as if it will handle production traffic immediately:

- Handle all error cases explicitly (no silent failures)
- Log meaningful messages at appropriate levels (DEBUG/INFO/WARNING/ERROR)
- Validate all inputs at system boundaries
- Never expose raw exception stack traces to end users
- Use `dataclasses` or `pydantic` for data transfer objects

---

## Rule 7 — No Hardcoded Secrets or Magic Numbers

```python
# ✅ Correct
from src.config.settings import XGB_N_ESTIMATORS
model = XGBRegressor(n_estimators=XGB_N_ESTIMATORS)

# ❌ Wrong — magic number
model = XGBRegressor(n_estimators=500)

# ❌ Never — hardcoded secret
API_KEY = "sk-abc123..."
```

All secrets belong in `.env` (git-ignored). All constants belong in
`src/config/settings.py`.

---

## Rule 8 — Atomic Commits

Each commit must represent exactly **one logical change**. Commit messages
must follow Conventional Commits format:

```
feat(scope): short imperative description

- Bullet 1: what changed and why
- Bullet 2: design decision made
- Bullet 3: edge cases handled

Closes #<issue-number>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `perf`

Never mix feature code, tests, and docs in a single commit.

---

## Rule 9 — Always Update Documentation

When changing behaviour, update in this order:

1. **Docstrings** on every modified function/class
2. **`docs/`** markdown files if the architecture or API changes
3. **`README.md`** if the quick-start or module list changes
4. **`.env.example`** if new environment variables are added

Undocumented features do not exist.

---

## Rule 10 — Use Async Where Appropriate

- FastAPI route handlers that do I/O must be `async def`
- HTTP calls inside route handlers must use `httpx` (async) not `requests`
- CPU-bound tasks (model training, feature computation) stay synchronous
  but must be offloaded to a thread pool via `asyncio.run_in_executor`
- Streamlit pages are synchronous (Streamlit's own execution model)

---

## Rule 11 — Explain Design Decisions

Every non-trivial implementation choice must be explained in either:
- A comment in the code (`# Reason: ...`)
- The commit message body
- A docstring `Note:` section

"I chose X over Y because Z" is always valuable context.

---

## Rule 12 — Definition of Done

An issue is **not complete** until every item below is checked:

| Check | Requirement |
|-------|-------------|
| ✅ Compiles | `python -c "import src"` succeeds |
| ✅ Tests pass | `pytest tests/ -q` exits 0 |
| ✅ Lint passes | `flake8 src/ tests/ --max-line-length=88` exits 0 |
| ✅ Types checked | `mypy src/ --ignore-missing-imports` exits 0 (target) |
| ✅ Docs updated | Docstrings, `docs/`, README updated where relevant |
| ✅ No TODOs | `grep -r "TODO\|FIXME\|HACK" src/` returns nothing |
| ✅ No secrets | No API keys, passwords, or tokens in source files |
| ✅ Performance | No O(n²) loops over large datasets without justification |
| ✅ Backward compat | Existing public interfaces unchanged (or versioned) |
| ✅ Atomic commits | Each commit is a single logical change with a clear message |

---

## Rule 13 — Issue Workflow

For every GitHub issue:

1. **Read** the issue, README, architecture docs, and coding standards first.
2. **Plan** — answer the Pre-Coding Checklist above.
3. **Implement** — layer by layer (config → data → features → model → API → tests → docs).
4. **Verify** — run tests and lint before committing.
5. **Commit** — one atomic commit per layer/component.
6. **Close** — add a closing comment summarising what was built.
7. **Suggest** — recommend the next issue to implement.

Never skip steps.
