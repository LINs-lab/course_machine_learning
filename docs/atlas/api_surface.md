# Public API Surface

Generated: 2026-09-09T15:43:16Z
Commit: a3763a2

## labs/lab02/template/test_utils.py
- function `test`

## lectures/week02/interactive/app/public/model.py
- class `NumericalError`
- function `_as_design_matrix`  [private]
- function `_as_matrix_vector_weights`  [private]
- function `_as_matrix_vector`  [private]
- function `design_matrix`
- function `predict`
- function `residuals`
- function `objective`
- function `loss_rho_psi`
- function `gradient_mse`
- function `hessian_half_mse`
- function `stable_step_limit`
- function `gd_path`
- function `batch_gradient_mse`
- function `enumerate_batch_gradients`
- function `batch_gradient_summary`
- function `_validate_batch_size`  [private]
- function `least_squares_summary`
- function `projection_summary`
- function `null_space_family`
- class `LeastSquaresSummary`
- class `ProjectionSummary`
- class `NullSpaceFamily`
  - method `weights_for`
- class `BatchGradientSummary`

## projects/project1/grading_tests/conftest.py
- function `pytest_addoption`
- function `github_repo_path`
- function `student_implementations`

## projects/project1/grading_tests/test_project1_public.py
- function `initial_w`
- function `y`
- function `tx`
- function `test_file_exists`
- function `test_function_exists`
- function `test_function_has_docstring`
- function `test_no_todo_left`
- function `test_mean_squared_error_gd_0_step`
- function `test_mean_squared_error_gd`
- function `test_mean_squared_error_sgd`
- function `test_least_squares`
- function `test_ridge_regression_lambda0`
- function `test_ridge_regression_lambda1`
- function `test_logistic_regression_0_step`
- function `test_logistic_regression`
- function `test_reg_logistic_regression`
- function `test_reg_logistic_regression_0_step`

## projects/project1/helpers.py
- function `load_csv_data`
- function `create_csv_submission`
