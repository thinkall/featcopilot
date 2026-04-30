# Relational Feature Engineering

A practical example for **customer / orders** style data where aggregation features matter.

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
