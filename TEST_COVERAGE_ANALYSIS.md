# Test Coverage Analysis

## Current State

**Test coverage: 0%** - The project contains no test files, no test framework configuration, and no automated testing infrastructure.

The codebase consists of 9 Python scripts (~3,900 lines) covering a data pipeline from API ingestion through ML model training and evaluation. There are 20+ defined functions and significant inline logic, none of which is tested.

---

## Inventory of Testable Functions

### Script 2 - JSON to CSV Conversion (`2_convert_json_to_csv_v7.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `is_nested(json_obj)` | 37-46 | Checks if a JSON object contains nested structures | **High** - pure function, no I/O |
| `json_to_csv_limit(limit)` | 79-114 | Flattens nested JSON with sizes into CSV | Medium - requires file fixtures |

### Script 3 - Normalisation (`3_Normalisation_v6.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `process_csv(input_file, output_file, columns_to_include)` | 29-34 | Filters CSV to selected columns | Medium - file I/O wrapper |
| `combine_columns(input_file, output_file)` | 90-113 | Fills NaN in primary column from secondary column | **High** - core data logic |
| `remove_rows_with_missing_data(csv_file_path, threshold)` | 181-202 | Drops rows exceeding a missing-value threshold | **High** - critical data quality step |
| `impute_missing_values_median(csv_file_path)` | 221-251 | Replaces NaN with column medians | **High** - affects all downstream analysis |
| `convert_angles_to_radians(csv_file_path, angle_columns, new_file_name)` | 272-310 | Converts degree columns to radians | **High** - mathematical correctness matters |
| `encode_subcategory(csv_file_path, output_file_path)` | 518-565 | Maps subcategory strings to integer labels | **High** - label correctness is critical |

### Script 4 - Sampling (`4_sampling_v1.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `sample_rows(input_file, output_file, num_samples)` | 10-21 | Random samples N rows from CSV | Medium - randomness needs seed control |
| `align_csv_columns(source_csv_path, target_csv_path, output_csv_path)` | 96-114 | Reorders target CSV columns to match source | **High** - column alignment is fragile |

### Script 5 - Random Forest (`5_random_forest_v2.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `run_optimized_random_forest(csv_file_path, target_column)` | 24-86 | Trains RF classifier, evaluates, saves model | Medium - requires data fixtures |
| `predict_bike_type(features)` | 226-231 | Loads pickled model and predicts | Medium - requires saved model |
| `load_data_and_predict(csv_file_path, target_column)` | 234-258 | End-to-end prediction pipeline | Low - integration test |
| `analyze_errors(df, target_column)` | 382-410 | Computes misclassification statistics | **High** - pure DataFrame logic |

### Script 7 - Data Exploration (`7_Data_Exploration_v1.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `count_rows_with_missing_values(file_path)` | 152-168 | Counts rows with any NaN | **High** - pure computation |
| `calculate_missing_value_percentages(file_path)` | 187-200 | Column-level missing % of total | **High** - pure computation |
| `calculate_missing_pair_percentage(file_path, col1, col2)` | 249-262 | % of rows missing both columns | **High** - pure computation |
| `get_unique_values(file_path, column_name)` | 289-296 | Unique non-null values in a column | **High** - simple query |

### Script 8 - Visualisation (`8_visualisation_v1.py`)

| Function | Lines | Description | Testability |
|----------|-------|-------------|-------------|
| `calculate_and_visualize_statistics(file_path)` | 19-63 | Computes descriptive stats per subcategory | Medium - mixed computation and plotting |

### Inline Logic (not in functions)

| Script | Description | Testability |
|--------|-------------|-------------|
| `3_Normalisation_v6.py:336-420` | Ratio calculations (SRR, AI, CS/BBD, etc.) and composite indices | **High** - mathematical formulas |
| `3_Normalisation_v6.py:432-448` | Z-score standardization | Medium |
| `6_k_means_clustering_alg_v3.py:16-113` | K-Means clustering pipeline | Medium |
| `9_ratios_v3.py:27-44` | Ratio calculations (SRR, STRR, CSR, AI, etc.) | **High** - mathematical formulas |

---

## Recommended Test Priorities

### Priority 1: Data Transformation Correctness (Critical)

These functions transform the raw data that feeds all downstream ML models. Errors here silently corrupt every result.

**1. `is_nested()` - JSON structure detection**
- Test with flat dict, nested dict, list of dicts, list of primitives, empty structures
- This is a pure function and the easiest starting point

**2. `combine_columns()` - Column merging logic**
- Test that primary column values are preserved when both columns have values
- Test that secondary column fills in when primary is NaN
- Test that rows where both are NaN remain NaN
- Test with missing column names (the warning path)

**3. `remove_rows_with_missing_data()` - Row filtering**
- Test that rows at exactly the threshold boundary are handled correctly (30% threshold)
- Test with a row that has 0% missing (should be kept)
- Test with a row that has 100% missing (should be removed)
- Test that the non-missing data is preserved unchanged

**4. `impute_missing_values_median()` - Median imputation**
- Test that the `subcategory` column is excluded from imputation
- Test that missing numeric values are replaced with the correct column median
- Test that non-missing values are not altered
- Test with columns where all values are NaN (edge case)

**5. `encode_subcategory()` - Label encoding**
- Test that each subcategory string maps to the correct integer (gravel=0, race=1, etc.)
- Test that unknown subcategories produce NaN with a warning
- Test that the `subcategory` column is absent (error path)

### Priority 2: Mathematical Correctness (High)

Ratio and angle calculations must be numerically correct. An error in a formula could invalidate thesis results.

**6. `convert_angles_to_radians()` - Angle conversion**
- Test known values: 0 degrees = 0 radians, 90 degrees = pi/2, 180 degrees = pi
- Test that original degree columns are dropped
- Test with a column name not in the DataFrame (warning path)

**7. Ratio calculations (inline in `3_Normalisation_v6.py` and `9_ratios_v3.py`)**
- Extract ratio formulas into testable functions, then verify:
  - `SRR = stack / reach` with known values
  - `AI = (reach * tan(radians(headTubeAngle))) / wheelbase`
  - `CS/BBD = chainstay / bottomBracketDrop`
  - `STRR = headTubeAngle / trail`
  - Division-by-zero cases produce NaN, not exceptions
  - Infinity values are replaced with NaN

**8. Composite indices (Stability_Index, Handling_Index, Comfort_Index)**
- Verify weighted sums produce expected values with known inputs
- Confirm weights sum to 1.0

### Priority 3: Pipeline Integration (Medium)

These tests verify that the full data pipeline produces consistent results end-to-end.

**9. `align_csv_columns()` - Column reordering**
- Test that output columns match source column order exactly
- Test that missing columns in target get filled with NaN
- Test that extra columns in target are dropped

**10. `sample_rows()` - Data sampling**
- Test that the output has exactly `num_samples` rows
- Test reproducibility with the same random seed
- Test with `num_samples` larger than the DataFrame (should raise or handle gracefully)

**11. Random Forest pipeline**
- Test that `run_optimized_random_forest()` produces a valid pickle file
- Test that `predict_bike_type()` returns predictions with the correct shape
- Test that predictions are integers within the valid label range (0-8)

### Priority 4: Data Quality Analysis (Lower)

**12. Missing value analysis functions**
- `count_rows_with_missing_values()` - verify count and percentage against a known DataFrame
- `calculate_missing_value_percentages()` - verify per-column percentages
- `calculate_missing_pair_percentage()` - verify with known patterns of missingness

---

## Structural Recommendations

### 1. Add a `requirements.txt`
The project has no dependency file. At minimum:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
requests
tqdm
pytest
```

### 2. Add pytest configuration
Create a `pytest.ini` or add `[tool.pytest.ini_options]` to a `pyproject.toml`.

### 3. Extract inline logic into functions
Several critical calculations (ratios, standardization, clustering) are written as top-level script code. Extracting them into functions would make them testable without running the entire pipeline. Specific targets:
- Ratio calculations in `3_Normalisation_v6.py` lines 336-420
- Ratio calculations in `9_ratios_v3.py` lines 27-44
- K-Means clustering in `6_k_means_clustering_alg_v3.py` lines 16-113

### 4. Reduce file I/O coupling
Most functions take a file path, read the CSV internally, process it, and write to another file. Refactoring these to accept and return DataFrames would:
- Make unit testing straightforward (pass in test DataFrames, assert on returned DataFrames)
- Eliminate the need for test fixture files on disk
- Allow functions to be composed in memory rather than through intermediate CSV files

### 5. Create test fixtures
Build small, representative test DataFrames (5-10 rows) that exercise:
- Normal cases with complete data
- Edge cases with missing values, boundary thresholds
- Invalid cases (wrong column names, empty DataFrames)

---

## Suggested Test File Structure

```
tests/
    conftest.py              # Shared fixtures (sample DataFrames, temp directories)
    test_json_conversion.py  # Tests for is_nested(), json_to_csv_limit()
    test_normalisation.py    # Tests for combine_columns(), remove_rows(), impute(), encode()
    test_ratios.py           # Tests for angle conversion, ratio formulas, composite indices
    test_sampling.py         # Tests for sample_rows(), align_csv_columns()
    test_random_forest.py    # Tests for model training, prediction, error analysis
    test_data_exploration.py # Tests for missing value analysis functions
```

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Incorrect ratio formulas | **High** - invalidates thesis conclusions | Medium - formulas were manually coded | Unit tests with hand-calculated expected values |
| Median imputation bugs | **High** - biases all ML results | Medium - `fillna` with `inplace=True` has known pandas pitfalls | Test that non-null values are unchanged after imputation |
| Subcategory encoding errors | **High** - model trains on wrong labels | Low - mapping is explicit | Test round-trip: encode then verify mapping is bijective |
| Column merge logic errors | **High** - data loss or corruption | Medium - priority logic is non-obvious | Test all 4 combinations of null/non-null in primary/secondary |
| Row threshold filtering | Medium - removes too many or too few rows | Low | Test boundary conditions at exactly 30% |
| Division by zero in ratios | Medium - produces inf/NaN silently | Medium - depends on data quality | Test that inf is replaced with NaN per the existing cleanup code |

---

## Summary

The most impactful improvement is testing the data transformation functions in Priority 1 and the ratio calculations in Priority 2. These directly determine the validity of the ML model inputs and, by extension, the thesis results. A minimal test suite covering these ~12 functions would provide substantial confidence in the pipeline's correctness with modest effort.
