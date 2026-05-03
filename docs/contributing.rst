Contributing
============

We welcome contributions to TF2JAX! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Add tests for your changes
6. Run the test suite
7. Submit a pull request

Development Setup
-----------------

Set up a development environment:

.. code-block:: bash

   git clone https://github.com/your-username/tf2jax.git
   cd tf2jax
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements_tests.txt
   pip install -e .

Running Tests
-------------

Run the full test suite:

.. code-block:: bash

   bash test.sh

Run specific tests:

.. code-block:: bash

   pytest tf2jax/_src/tf2jax_test.py -v

Run with coverage:

.. code-block:: bash

   pytest --cov=tf2jax --cov-report=html

Code Style
----------

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:

.. code-block:: bash

   pre-commit run --all-files

Adding New Operations
---------------------

To add support for a new TensorFlow operation:

1. Add the operation implementation in `tf2jax/_src/ops.py`
2. Add comprehensive tests in the appropriate test file
3. Update the documentation if needed
4. Ensure the operation works with JAX transformations

Example operation implementation:

.. code-block:: python

   def _my_operation(proto, inputs):
       """Convert MyOperation to JAX."""
       # Implementation here
       return jax.lax.my_operation(*inputs)

   # Register the operation
   _TF_OP_IMPLS["MyOperation"] = _my_operation

Pull Request Guidelines
-----------------------

- Keep pull requests focused and small
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style

Issue Reporting
---------------

When reporting issues, please include:

- TF2JAX version
- Python version
- TensorFlow version
- JAX version
- Minimal reproducible example
- Expected vs actual behavior

License
-------

By contributing to TF2JAX, you agree that your contributions will be licensed under the Apache License 2.0.
