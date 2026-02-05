"""Tests for data normalisation functions (3_Normalisation_v6.py).

These cover the Priority 1 data transformation functions that feed
all downstream ML models. Bugs here silently corrupt every result.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.import_helpers import import_functions_from_script

_script = os.path.join(os.path.dirname(__file__), '..', '3_Normalisation_v6.py')
_funcs = import_functions_from_script(_script, [
    'combine_columns',
    'remove_rows_with_missing_data',
    'impute_missing_values_median',
    'convert_angles_to_radians',
    'encode_subcategory',
])

combine_columns = _funcs['combine_columns']
remove_rows_with_missing_data = _funcs['remove_rows_with_missing_data']
impute_missing_values_median = _funcs['impute_missing_values_median']
convert_angles_to_radians = _funcs['convert_angles_to_radians']
encode_subcategory = _funcs['encode_subcategory']


# ---------------------------------------------------------------------------
# Tests for combine_columns()
# ---------------------------------------------------------------------------

class TestCombineColumns:
    """Tests for the column-merging logic that fills primary from secondary."""

    def test_primary_preserved_when_both_present(self, combine_columns_df, tmp_dir):
        """When primary has a value, it should NOT be overwritten by secondary."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        combine_columns_df.to_csv(input_path, index=False)

        combine_columns(input_path, output_path)
        result = pd.read_csv(output_path)

        # Row 0: primary trail is 60.0, should stay 60.0 (not overwritten by 61.0)
        assert result.loc[0, 'geometry.source.trailMM'] == 60.0

    def test_secondary_fills_nan_in_primary(self, combine_columns_df, tmp_dir):
        """When primary is NaN but secondary has a value, primary should be filled."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        combine_columns_df.to_csv(input_path, index=False)

        combine_columns(input_path, output_path)
        result = pd.read_csv(output_path)

        # Row 1: primary trail was NaN, secondary was 59.0
        assert result.loc[1, 'geometry.source.trailMM'] == 59.0
        # Row 0: primary rake was NaN, secondary was 50.0
        assert result.loc[0, 'geometry.source.rakeMM'] == 50.0

    def test_both_nan_stays_nan(self, combine_columns_df, tmp_dir):
        """When both primary and secondary are NaN, result should remain NaN."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        combine_columns_df.to_csv(input_path, index=False)

        combine_columns(input_path, output_path)
        result = pd.read_csv(output_path)

        # Row 3: both trail columns are NaN
        assert pd.isna(result.loc[3, 'geometry.source.trailMM'])
        # Row 3: both rake columns are NaN
        assert pd.isna(result.loc[3, 'geometry.source.rakeMM'])

    def test_secondary_not_modified(self, combine_columns_df, tmp_dir):
        """Secondary column values should not be altered."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        combine_columns_df.to_csv(input_path, index=False)

        combine_columns(input_path, output_path)
        result = pd.read_csv(output_path)

        assert result.loc[0, 'geometry.computed.trailMM'] == 61.0
        assert result.loc[1, 'geometry.computed.trailMM'] == 59.0


# ---------------------------------------------------------------------------
# Tests for remove_rows_with_missing_data()
# ---------------------------------------------------------------------------

class TestRemoveRowsWithMissingData:
    """Tests for the 30% missing-value threshold row removal."""

    def test_rows_exceeding_threshold_are_removed(self, missing_data_df, tmp_dir):
        """Rows with > 30% NaN columns should be dropped."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        missing_data_df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = remove_rows_with_missing_data(input_path, threshold=0.30)
        finally:
            os.chdir(original_dir)

        # Rows 1 and 3 have 4/10 = 40% NaN -> removed
        # Rows 0, 2, 4 should remain
        assert len(result) == 3

    def test_complete_rows_preserved(self, missing_data_df, tmp_dir):
        """Rows with no missing values should be kept intact."""
        input_path = os.path.join(tmp_dir, 'input.csv')
        missing_data_df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = remove_rows_with_missing_data(input_path, threshold=0.30)
        finally:
            os.chdir(original_dir)

        # Row 0 has no NaN -- check values unchanged
        assert result.iloc[0]['col1'] == 1.0
        assert result.iloc[0]['col5'] == 5.0

    def test_row_at_boundary_is_kept(self, tmp_dir):
        """A row with exactly 30% NaN (3/10 columns) should be kept."""
        df = pd.DataFrame({
            f'col{i}': [np.nan if i <= 3 else float(i)]
            for i in range(1, 11)
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = remove_rows_with_missing_data(input_path, threshold=0.30)
        finally:
            os.chdir(original_dir)

        assert len(result) == 1

    def test_all_nan_row_removed(self, tmp_dir):
        """A row with 100% NaN should be removed at any threshold."""
        df = pd.DataFrame({
            'a': [np.nan], 'b': [np.nan], 'c': [np.nan],
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = remove_rows_with_missing_data(input_path, threshold=0.30)
        finally:
            os.chdir(original_dir)

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests for impute_missing_values_median()
# ---------------------------------------------------------------------------

class TestImputeMissingValuesMedian:
    """Tests for median imputation."""

    def test_nan_replaced_with_median(self, tmp_dir):
        """NaN values should be replaced with the column median."""
        df = pd.DataFrame({
            'subcategory': ['gravel', 'race', 'endurance'],
            'colA': [10.0, np.nan, 30.0],
            'colB': [100.0, 200.0, np.nan],
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = impute_missing_values_median(input_path)
        finally:
            os.chdir(original_dir)

        # colA median of [10, 30] = 20.0
        assert result.loc[1, 'colA'] == 20.0
        # colB median of [100, 200] = 150.0
        assert result.loc[2, 'colB'] == 150.0

    def test_existing_values_not_altered(self, tmp_dir):
        """Non-NaN values should remain unchanged after imputation."""
        df = pd.DataFrame({
            'subcategory': ['gravel', 'race'],
            'colA': [10.0, np.nan],
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = impute_missing_values_median(input_path)
        finally:
            os.chdir(original_dir)

        assert result.loc[0, 'colA'] == 10.0

    def test_subcategory_excluded_from_imputation(self, tmp_dir):
        """The subcategory column should not be imputed."""
        df = pd.DataFrame({
            'subcategory': ['gravel', np.nan, 'endurance'],
            'colA': [10.0, 20.0, 30.0],
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        df.to_csv(input_path, index=False)

        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = impute_missing_values_median(input_path)
        finally:
            os.chdir(original_dir)

        # Subcategory NaN should remain NaN (not imputed)
        assert pd.isna(result.loc[1, 'subcategory'])


# ---------------------------------------------------------------------------
# Tests for encode_subcategory()
# ---------------------------------------------------------------------------

class TestEncodeSubcategory:
    """Tests for subcategory string-to-integer label encoding."""

    EXPECTED_MAPPING = {
        'gravel': 0, 'race': 1, 'endurance': 2, 'aero': 3,
        'triathlon': 4, 'cyclocross': 5, 'touring': 6,
        'general-road': 7, 'track': 8,
    }

    def test_all_known_subcategories_encoded_correctly(self, tmp_dir):
        """Each known subcategory should map to its defined integer."""
        names = list(self.EXPECTED_MAPPING.keys())
        df = pd.DataFrame({
            'subcategory': names,
            'value': range(len(names)),
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        result = encode_subcategory(input_path, output_path)

        for i, name in enumerate(names):
            assert result.loc[i, 'subcategory'] == self.EXPECTED_MAPPING[name], \
                f"{name} should map to {self.EXPECTED_MAPPING[name]}"

    def test_unknown_subcategory_becomes_nan(self, tmp_dir):
        """Subcategories not in the mapping should become NaN."""
        df = pd.DataFrame({
            'subcategory': ['gravel', 'unknown_type'],
            'value': [1, 2],
        })
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        result = encode_subcategory(input_path, output_path)

        assert result.loc[0, 'subcategory'] == 0  # gravel
        assert pd.isna(result.loc[1, 'subcategory'])  # unknown

    def test_encoding_is_bijective(self):
        """Each subcategory maps to a unique integer (no collisions)."""
        codes = list(self.EXPECTED_MAPPING.values())
        assert len(codes) == len(set(codes)), "Encoding has duplicate codes"


# ---------------------------------------------------------------------------
# Tests for convert_angles_to_radians()
# ---------------------------------------------------------------------------

class TestConvertAnglesToRadians:
    """Tests for degree-to-radian conversion."""

    def test_known_angle_values(self, tmp_dir):
        """0 deg=0, 90 deg=pi/2, 180 deg=pi."""
        df = pd.DataFrame({'angle': [0.0, 90.0, 180.0]})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        result = convert_angles_to_radians(input_path, ['angle'], output_path)

        assert np.isclose(result.loc[0, 'angle_radians'], 0.0)
        assert np.isclose(result.loc[1, 'angle_radians'], np.pi / 2)
        assert np.isclose(result.loc[2, 'angle_radians'], np.pi)

    def test_original_column_dropped(self, tmp_dir):
        """The original degree column should be removed."""
        df = pd.DataFrame({'angle': [45.0], 'other': [100.0]})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        result = convert_angles_to_radians(input_path, ['angle'], output_path)

        assert 'angle' not in result.columns
        assert 'angle_radians' in result.columns
        assert 'other' in result.columns

    def test_missing_column_produces_warning(self, tmp_dir, capsys):
        """Specifying a non-existent column should print a warning."""
        df = pd.DataFrame({'other': [1.0]})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        convert_angles_to_radians(input_path, ['nonexistent'], output_path)
        captured = capsys.readouterr()

        assert 'Warning' in captured.out
        assert 'nonexistent' in captured.out

    def test_typical_head_tube_angle(self, tmp_dir):
        """A 72 deg head tube angle should convert to ~1.2566 radians."""
        df = pd.DataFrame({'headTubeAngle': [72.0]})
        input_path = os.path.join(tmp_dir, 'input.csv')
        output_path = os.path.join(tmp_dir, 'output.csv')
        df.to_csv(input_path, index=False)

        result = convert_angles_to_radians(input_path, ['headTubeAngle'], output_path)

        expected = np.radians(72.0)
        assert np.isclose(result.loc[0, 'headTubeAngle_radians'], expected)
