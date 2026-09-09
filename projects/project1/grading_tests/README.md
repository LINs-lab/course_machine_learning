# Project 1 public tests

Keep `conftest.py` and `test_project1_public.py` in this directory. Install the
test dependencies in your Python environment:

```bash
python -m pip install numpy pytest
```

From the `project1` directory, run:

```bash
python -m pytest grading_tests --submission-dir /path/to/your/submission
```

The submission directory must contain `implementations.py` and `README.md`.
The six required functions must have docstrings and return `(w, loss)`, where
`w` is a one-dimensional NumPy array and `loss` is a scalar. Finish any remaining
TODOs before running the tests. The public tests are basic checks; passing them
does not guarantee full marks.

`environment.yml` is the existing TA grading environment. Its additional tools
and packages do not expand the libraries permitted in student implementations.
