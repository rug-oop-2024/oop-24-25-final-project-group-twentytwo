```
# DSC-003: Encapsulation and Private Variables in `Artifact` Class
# Date: 2024-11-10
# Decision: Make `data`, `metadata`, and `tags` private within the `Artifact` class. 
# Status: Accepted
# Motivation: Improve encapsulation and prevent direct external modification of important attributes.
# Reason:
- `data`, `metadata`, and `tags` should be treated as implementation details and accessed through methods to maintain control over the integrity and consistency of the `Artifact` class.
- Direct access to `data` could result in unintended modifications.
- Using properties and setter methods ensures that these values are validated and modified in a controlled way.
# Limitations:
- The Python convention for private variables (e.g., `_data`) is not enforced by the language, adherence is important.
- The new encapsulation requires the use of getter and setter methods for accessing and modifying `data`, `metadata`, and `tags`, which may add some overhead.
# Alternatives:
- Keep `data`, `metadata`, and `tags` public, which would make them easier to access but could cause leakage in the consistency and safety of the class.
```