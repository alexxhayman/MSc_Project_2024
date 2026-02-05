"""Tests for sampling and column alignment functions (4_sampling_v1.py)."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.import_helpers import import_functions_from_script

_script = os.path.join(os.path.dirname(__file__), '..', '4_sampling_v1.py')
_funcs = import_functions_from_script(_script, ['sample_rows', 'align_csv_columns'])

sample_rows = _funcs['sample_rows']
align_csv_columns = _funcs['align_csv_columns']


# ---------------------------------------------------------------------------
# Tests for sample_rows()
# ---------------------------------------------------------------------------

class TestSampleRows:
    """Tests for the random sampling function."""

    def test_correct_number_of_rows(self, tmp_dir):
        """Output should have exactly num_samples rows."""
        df = pd.DataFrame({'a': range(100), 'b': range(100)})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        sample_rows(input_path, output_path, num_samples=10)
        result = pd.read_csv(output_path)

        assert len(result) == 10

    def test_reproducible_with_same_seed(self, tmp_dir):
        """Running twice should produce the same sample (bug fix verification)."""
        df = pd.DataFrame({'a': range(100), 'b': range(100)})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output1 = os.path.join(tmp_dir, 'output1.csv')
        output2 = os.path.join(tmp_dir, 'output2.csv')
        df.to_csv(input_path, index=False)

        sample_rows(input_path, output1, num_samples=10)
        sample_rows(input_path, output2, num_samples=10)

        result1 = pd.read_csv(output1)
        result2 = pd.read_csv(output2)

        pd.testing.assert_frame_equal(result1, result2)

    def test_columns_preserved(self, tmp_dir):
        """All columns from the input should appear in the output."""
        df = pd.DataFrame({'x': range(50), 'y': range(50), 'z': range(50)})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        sample_rows(input_path, output_path, num_samples=5)
        result = pd.read_csv(output_path)

        assert list(result.columns) == ['x', 'y', 'z']


# ---------------------------------------------------------------------------
# Tests for align_csv_columns()
# ---------------------------------------------------------------------------

class TestAlignCsvColumns:
    """Tests for the column reordering function."""

    def test_columns_reordered_to_match_source(self, tmp_dir):
        """Target columns should appear in the same order as source."""
        source_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        target_df = pd.DataFrame({'c': [30], 'a': [10], 'b': [20]})

        source_path = os.path.join(tmp_dir, 'source.csv')
        target_path = os.path.join(tmp_dir, 'target.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)

        align_csv_columns(source_path, target_path, output_path)
        result = pd.read_csv(output_path)

        assert list(result.columns) == ['a', 'b', 'c']
        assert result.loc[0, 'a'] == 10
        assert result.loc[0, 'b'] == 20
        assert result.loc[0, 'c'] == 30

    def test_missing_columns_filled_with_nan(self, tmp_dir):
        """Columns in source but not in target should be filled with NaN."""
        source_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        target_df = pd.DataFrame({'a': [10]})

        source_path = os.path.join(tmp_dir, 'source.csv')
        target_path = os.path.join(tmp_dir, 'target.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)

        align_csv_columns(source_path, target_path, output_path)
        result = pd.read_csv(output_path)

        assert list(result.columns) == ['a', 'b', 'c']
        assert result.loc[0, 'a'] == 10
        assert pd.isna(result.loc[0, 'b'])
        assert pd.isna(result.loc[0, 'c'])

    def test_extra_target_columns_dropped(self, tmp_dir):
        """Columns in target but not in source should be excluded."""
        source_df = pd.DataFrame({'a': [1], 'b': [2]})
        target_df = pd.DataFrame({'a': [10], 'b': [20], 'extra': [99]})

        source_path = os.path.join(tmp_dir, 'source.csv')
        target_path = os.path.join(tmp_dir, 'target.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)

        align_csv_columns(source_path, target_path, output_path)
        result = pd.read_csv(output_path)

        assert list(result.columns) == ['a', 'b']
        assert 'extra' not in result.columns
