# Relational Feature Engineering

A practical example for **customer / orders** style data where aggregation features matter.

## Prediction-time history features

For event history and forecasting, use the explicit point-in-time engine rather
than aggregating a whole validation/test table:

```python
import pandas as pd
from featcopilot.engines import PointInTimeRelationalEngine

queries = pd.DataFrame({
    "customer_id": [1, 1],
    "prediction_at": pd.to_datetime(["2025-01-03", "2025-01-06"], utc=True),
})
events = pd.DataFrame({
    "customer_id": [1, 1, 1],
    "available_at": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-09"], utc=True),
    "amount": [10.0, 20.0, 999.0],
})

engine = PointInTimeRelationalEngine(
    aggregations=[{
        "name": "past_30d_mean",
        "table": "events",
        "entity_column": "customer_id",
        "available_at": "available_at",
        "value_column": "amount",
        "aggregation": "mean",
        "window": "30D",
    }],
    entity_column="customer_id",
    prediction_time_column="prediction_at",
)
features = engine.fit_transform(queries, related_tables={"events": events})
# past_30d_mean is [10.0, 15.0]; equal-time and future records are excluded.
```

Availability must reflect **when the value became known**, not merely when
an event occurred. Every query uses a strictly earlier cutoff. The trailing
window's lower boundary is inclusive. Naive datetimes are interpreted as UTC;
convert numeric timestamps with explicit units before calling the engine.

Fit copies the supplied history, so changing the caller's table cannot alter
inference. Transform uses that frozen snapshot and never learns from the query
batch. Refit explicitly to supply newly available observations. Pickled engines
contain derived history and should be handled as potentially sensitive artifacts.

Supported aggregates are `count`, `sum`, `mean`, `min`, `max`, population `std`,
and last non-missing `last`. Counts without a value column count rows; with a
value column they count finite values. Unknown entities have zero counts/sums
and missing other aggregates. Original query columns are retained; exclude
identifiers and raw timestamps from downstream models where appropriate.
For `last`, equal availability timestamps retain the supplied history's row
order; preaggregate or order tied records explicitly when that distinction matters.

Use `FeatureEngineerTransformer(engine="point_in_time", ...)` for sklearn
pipeline wrapping. This provides explicit aggregation semantics, not automatic
multi-hop synthesis or a demonstrated relational SOTA result.

## When to use this

Use the relational engine when:
- you have a primary table and one or more related tables
- useful signal lives in counts / sums / means / maxima over related entities
- you want a lighter-weight path than a full Featuretools setup

## Example

```python
import pandas as pd

from featcopilot.engines.relational import RelationalEngine

orders = pd.DataFrame(
    {
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [1, 1, 2, 2, 3],
        "amount": [100, 200, 150, 300, 50],
        "category": ["A", "B", "A", "A", "B"],
    }
)

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "age": [25, 35, 45],
        "income": [50000, 70000, 60000],
    }
)

engine = RelationalEngine(
    aggregation_functions=["mean", "sum", "count", "max", "min"],
    verbose=True,
)
engine.add_relationship("orders", "customers", "customer_id")

features = engine.fit_transform(
    orders,
    related_tables={"customers": customers},
)

print(features.columns.tolist())
```

## Typical generated features

- `customers_age_mean`
- `customers_income_mean`
- `amount_by_category_mean`
- `amount_by_category_count`

## Guardrails

The legacy `RelationalEngine` example above is for batch aggregation.
Its self-aggregations are not a point-in-time or frozen-statistics guarantee.
Use `PointInTimeRelationalEngine` for temporal histories, or fitted statistical
rules when group mappings must be learned once on an IID training set.

`RelationalEngine` now validates configured relationship keys:
- missing child key in the primary table -> raises early
- missing parent key in a related table -> raises early
- duplicate relationship definitions are ignored

That is boring, but it is the kind of boring that saves time.

## When Featuretools is still the better choice

Use Featuretools when you need:
- deeper DFS-style synthesis
- richer primitive libraries
- more complex multi-table workflows

Use FeatCopilot relational mode when you want:
- a smaller API surface
- straightforward aggregation features
- sklearn-friendly workflows
