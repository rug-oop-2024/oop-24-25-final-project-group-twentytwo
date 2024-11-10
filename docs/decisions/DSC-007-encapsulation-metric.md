```
# DSC-004: Encapsulation and Private Variables in `Metric` Class and Implementations
# Date: 2024-11-10
# Decision: Make `task_type` a class-level attribute and prevent direct modification of the `task_type` in Metric subclasses.
# Status: Accepted
# Motivation: Improve encapsulation and ensure that the metric’s task type is consistent and not directly modified by external code.
# Reason:
# - `task_type` represents the task for which the metric is used (either "Regression" or "Classification"). It should be immutable after class definition to prevent misclassification.
# - Allowing `task_type` to be directly modified outside the class could result in incorrect behavior when computing the metric.
# - Using a class-level constant for `task_type` ensures that it reflects the task for the specific metric and that it cannot be changed.
# Limitations:
# - The `task_type` cannot be dynamically adjusted at runtime, but this is not a concern as the task type is intrinsic to the metric's definition (e.g., Mean Squared Error is always for regression).
# Alternatives:
# - Keep `task_type` as an instance-level attribute and allow its modification via setter methods or direct assignment. This would introduce unnecessary complexity.
```