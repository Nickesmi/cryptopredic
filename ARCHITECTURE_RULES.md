# Architecture Rules

> These rules are **non-negotiable** and apply to every line of code committed
> to this repository. Any pull request that violates them must be revised before
> merging. They exist to prevent architectural drift across hundreds of issues
> and multiple contributors (human or AI).

---

## 1. Clean Architecture — Always

The codebase is organised into four concentric layers. **Dependencies must
always point inward.** An inner layer may never import from an outer layer.

```
┌──────────────────────────────────────────────┐
│  Infrastructure (outermost)                  │
│  API routes, Streamlit pages, DB adapters,   │
│  HTTP clients, file I/O, schedulers          │
├──────────────────────────────────────────────┤
│  Interface Adapters                          │
│  Repository implementations, serialisers,   │
│  DTO transformers, model wrappers            │
├──────────────────────────────────────────────┤
│  Use Cases / Application Logic               │
│  Feature pipelines, forecast orchestrators,  │
│  ranking engines, risk calculators           │
├──────────────────────────────────────────────┤
│  Entities / Domain (innermost)               │
│  Abstract interfaces, value objects,         │
│  domain models, pure business rules          │
└──────────────────────────────────────────────┘
```

**Violations to never commit:**
- An entity importing from infrastructure
- A use case calling a third-party API directly
- Business logic inside a FastAPI route handler or Streamlit widget
- A model class knowing about HTTP or databases

---

## 2. All External Services Must Be Adapters

Every third-party dependency (CoinGecko, Twitter/X API, Redis, S3, etc.)
must be hidden behind an abstract interface defined in the **Entities** layer.

```python
# ✅ Correct — entity defines the contract
class PriceRepository(ABC):
    @abstractmethod
    def fetch(self, symbol: str, days: int) -> pd.DataFrame: ...

# ✅ Correct — infrastructure provides the implementation
class CoinGeckoPriceRepository(PriceRepository): ...

# ❌ Wrong — use case calls the API directly
class FeaturePipeline:
    def build(self):
        requests.get("https://api.coingecko.com/...")  # NEVER
```

---

## 3. Dependency Injection Everywhere

Concrete implementations must be injected; never instantiated inside business
logic. This makes every component independently testable without network access.

```python
# ✅ Correct
def run_pipeline(symbol, horizon, repository: PriceRepository = None):
    repo = repository or CoinGeckoPriceRepository()

# ❌ Wrong — hardcoded dependency
def run_pipeline(symbol, horizon):
    repo = CoinGeckoPriceRepository()  # not injectable → untestable
```

---

## 4. Every Model Implements a Common Interface

All forecast models must implement the `TimeSeriesForecaster` contract so
they are interchangeable. No caller should depend on a concrete model class.

```python
class ForecastModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> dict: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel": ...
```

---

## 5. Every Indicator Is a Plugin

Technical indicators, sentiment features, and on-chain features must be
**pure, stateless functions** with the signature:

```python
def add_<indicator_name>(df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...
```

They are registered in `FeaturePipeline._steps` — never called directly by
model code. Adding a new indicator requires **no modification** of existing
code (Open/Closed Principle).

---

## 6. No Business Logic in React Components

Frontend components are responsible for **rendering and user interaction only**.
All data transformation, scoring, and filtering happens in the API layer.

```jsx
// ✅ Correct — component renders what the API returns
const ForecastCard = ({ predicted_price, mape }) => ...

// ❌ Wrong — business logic in component
const ForecastCard = ({ prices }) => {
  const predicted = prices.reduce(...) * riskMultiplier  // NEVER
}
```

---

## 7. No Duplicated Code

Before writing new code, search for existing implementations. If a pattern
appears more than once, extract it into a shared utility, base class, or
mixin. The **DRY** (Don't Repeat Yourself) principle is mandatory.

---

## 8. Forecasting Is Separate from Trading Decisions

The system pipeline has strict layer separation:

```
Market Data
      │
      ▼
Feature Engineering          ← src/features/
      │
      ▼
Forecast Models              ← src/models/
      │
      ▼
Confidence Estimator         ← src/evaluation/
      │
      ▼
Risk Manager                 ← src/ranking/
      │
      ▼
LLM Reasoning Agent          ← src/agents/  (future)
      │
      ▼
Frontend Visualisation       ← frontend/
```

A forecast model **never** makes a buy/sell decision.
A risk manager **never** fetches raw market data.
Each layer only speaks to its immediate neighbour.

---

## 9. Every New Feature Must Have Tests

No code is "done" until unit tests exist for it. Tests must:
- Use synthetic/mock data — **no network calls in tests**
- Cover the happy path, edge cases, and error conditions
- Be deterministic and fast (< 5 s total for the test suite)

---

## 10. Configuration Over Hardcoding

All tuneable values (thresholds, URLs, model hyperparameters, file paths)
must live in `src/config/settings.py` and be overridable via environment
variables. **No magic numbers or hardcoded strings** in business logic.

---

## Enforcement

These rules are checked during code review. CI (`ci.yml`) enforces:
- `flake8` for style (max-line-length = 88)
- `pytest` — all tests must pass
- `mypy` — strict type checking (to be added in a future issue)

Violations block merging.
