"""Shared test fixtures for the bike geometry analysis pipeline."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_geometry_df():
    """A small DataFrame mimicking the filtered bike geometry data.

    Contains 5 rows with realistic geometry values for road bikes.
    Includes NaN values to test imputation logic.
    """
    return pd.DataFrame({
        'subcategory': ['gravel', 'race', 'endurance', 'aero', 'triathlon'],
        'geometry.source.stackMM': [580.0, 540.0, 570.0, 530.0, np.nan],
        'geometry.source.reachMM': [390.0, 385.0, 380.0, 395.0, 400.0],
        'geometry.source.seatTubeLengthMM': [520.0, 510.0, 530.0, 500.0, 490.0],
        'geometry.source.topTubeLengthMM': [555.0, 545.0, 560.0, 540.0, 550.0],
        'geometry.source.headTubeAngle': [71.5, 73.0, 72.0, 73.5, 76.0],
        'geometry.source.seatTubeAngle': [73.0, 74.0, 73.5, 74.5, 78.0],
        'geometry.source.chainstayLengthMM': [425.0, 410.0, 420.0, 405.0, 400.0],
        'geometry.source.bottomBracketDropMM': [72.0, 70.0, np.nan, 68.0, 65.0],
        'geometry.source.wheelbaseMM': [1020.0, 990.0, 1010.0, 985.0, 975.0],
        'geometry.source.rakeMM': [50.0, 45.0, 47.0, 43.0, np.nan],
        'geometry.source.trailMM': [63.0, 58.0, 60.0, 56.0, 42.0],
        'geometry.computed.stackReachRatio': [1.487, 1.403, 1.500, 1.342, np.nan],
        'geometry.computed.bottomBracketHeightMM': [270.0, 272.0, np.nan, 274.0, 277.0],
        'geometry.source.frontCenterMM': [595.0, 580.0, 590.0, 580.0, 575.0],
    })


@pytest.fixture
def sample_geometry_with_radians_df(sample_geometry_df):
    """The geometry DataFrame after angle conversion to radians.

    Simulates the output of convert_angles_to_radians().
    """
    df = sample_geometry_df.copy()
    df['geometry.source.headTubeAngle_radians'] = np.radians(df['geometry.source.headTubeAngle'])
    df['geometry.source.seatTubeAngle_radians'] = np.radians(df['geometry.source.seatTubeAngle'])
    df.drop(columns=['geometry.source.headTubeAngle', 'geometry.source.seatTubeAngle'], inplace=True)
    return df


@pytest.fixture
def combine_columns_df():
    """DataFrame for testing the combine_columns logic.

    Has primary/secondary column pairs with various NaN patterns.
    """
    return pd.DataFrame({
        'geometry.source.trailMM': [60.0, np.nan, 55.0, np.nan],
        'geometry.computed.trailMM': [61.0, 59.0, np.nan, np.nan],
        'geometry.source.rakeMM': [np.nan, 45.0, 47.0, np.nan],
        'geometry.computed.rakeMM': [50.0, 46.0, np.nan, np.nan],
    })


@pytest.fixture
def missing_data_df():
    """DataFrame for testing row-removal threshold logic.

    10 columns total. Threshold of 30% means rows with >3 NaN columns
    should be removed.
    """
    return pd.DataFrame({
        'col1': [1.0, np.nan, 1.0, np.nan, 1.0],
        'col2': [2.0, np.nan, 2.0, np.nan, 2.0],
        'col3': [3.0, np.nan, 3.0, np.nan, 3.0],
        'col4': [4.0, np.nan, 4.0, np.nan, np.nan],
        'col5': [5.0, 5.0, 5.0, 5.0, 5.0],
        'col6': [6.0, 6.0, 6.0, 6.0, 6.0],
        'col7': [7.0, 7.0, 7.0, 7.0, 7.0],
        'col8': [8.0, 8.0, 8.0, 8.0, 8.0],
        'col9': [9.0, 9.0, 9.0, 9.0, 9.0],
        'col10': [10.0, 10.0, 10.0, 10.0, 10.0],
    })
    # Row 0: 0 NaN (keep)
    # Row 1: 4 NaN = 40% (remove)
    # Row 2: 0 NaN (keep)
    # Row 3: 4 NaN = 40% (remove)
    # Row 4: 1 NaN = 10% (keep)
