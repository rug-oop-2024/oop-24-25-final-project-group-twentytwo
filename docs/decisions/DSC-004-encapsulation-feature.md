```
# DSC-004: Encapsulation and Private Variables in `Feature` Class
# Date: 2024-11-10
# Decision: Make `is_target`, and `_values` private within the `Feature` class.
# Status: Accepted
# Motivation: Improve encapsulation and prevent direct external modification of important attributes.
# Reason:
- `_is_target` and `_values` should be treated as implementation details and accessed through methods to maintain control over the integrity and consistency of the `Feature` class.
- Direct access to `_is_target` and '_values' could result in unintended modifications.
- Using properties and setter methods ensures that these values are validated and modified in a controlled way.
# Limitations:
- The Python convention for private variables (e.g., `_data`) is not enforced by the language, adherence is important.
- The new encapsulation requires the use of getter and setter methods for accessing and modifying `_is_target` and `_values`, which may add some complexity.
# Alternatives:
- Keep '_is_target' and  '_value` public, which would make them easier to access but could cause leakage in the consistency and safety of the class.
```