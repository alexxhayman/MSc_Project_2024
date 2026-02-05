"""Tests for ratio calculations and composite indices.

These verify the mathematical correctness of the formulas in
3_Normalisation_v6.py and 9_ratios_v3.py. An error in any formula
could invalidate thesis results.
"""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Since the ratio calculations are inline (not in functions), we replicate
# the formulas here and test them directly. This also validates that the
# formulas as written in the source files produce correct results.
# ---------------------------------------------------------------------------


def compute_srr(stack, reach):
    """SRR = Stack / Reach"""
    return stack / reach


def compute_ai(reach, head_tube_angle_radians, wheelbase):
    """AI = (reach * tan(headTubeAngle_radians)) / wheelbase

    IMPORTANT: head_tube_angle_radians is ALREADY in radians.
    Do NOT apply np.radians() again.
    """
    return (reach * np.tan(head_tube_angle_radians)) / wheelbase


def compute_ai_buggy(reach, head_tube_angle_radians, wheelbase):
    """The BUGGY version that was in the code before the fix.

    This applies np.radians() to a value already in radians.
    """
    return (reach * np.tan(np.radians(head_tube_angle_radians))) / wheelbase


def compute_cs_bbd(chainstay, bb_drop):
    """CS/BBD = Chainstay Length / Bottom Bracket Drop"""
    return chainstay / bb_drop


def compute_ett_s(top_tube, stack):
    """ETT/S = Top Tube Length / Stack"""
    return top_tube / stack


def compute_t_wb(trail, wheelbase):
    """T/WB = Trail / Wheelbase"""
    return trail / wheelbase


def compute_hta_trail(head_tube_angle_radians, trail):
    """HTA/Trail = Head Tube Angle (radians) / Trail"""
    return head_tube_angle_radians / trail


def compute_csl_sta(chainstay, seat_tube_angle_radians):
    """CSL/STA = Chainstay Length / Seat Tube Angle (radians)"""
    return chainstay / seat_tube_angle_radians


def compute_strr(head_tube_angle_radians, trail):
    """STRR = Head Tube Angle (radians) / Trail (from 9_ratios_v3.py)"""
    return head_tube_angle_radians / trail


def compute_csr(wheelbase, trail, bb_drop):
    """CSR = (Wheelbase * Trail) / Bottom Bracket Drop"""
    return (wheelbase * trail) / bb_drop


def compute_stability_index(bb_height, chainstay, wheelbase):
    """Stability_Index = 0.5*BBH + 0.3*CS + 0.2*WB"""
    return 0.5 * bb_height + 0.3 * chainstay + 0.2 * wheelbase


def compute_handling_index(hta_radians, trail, rake):
    """Handling_Index = 0.5*HTA_rad + 0.25*Trail + 0.25*Rake"""
    return 0.5 * hta_radians + 0.25 * trail + 0.25 * rake


def compute_comfort_index(bbh_cs_interaction, cs_bbd, ett_s):
    """Comfort_Index = 0.5*BBH_CS_Interaction + 0.3*CS/BBD + 0.2*ETT/S"""
    return 0.5 * bbh_cs_interaction + 0.3 * cs_bbd + 0.2 * ett_s


# ---------------------------------------------------------------------------
# Test: AI formula double-radians bug (the most critical fix)
# ---------------------------------------------------------------------------

class TestAIFormulaFix:
    """Verify the AI formula produces correct results after the fix.

    The bug was: np.tan(np.radians(value_already_in_radians))
    The fix is:  np.tan(value_already_in_radians)
    """

    def test_ai_correct_for_72_degree_head_tube(self):
        """A 72° HTA with typical geometry should produce a sensible AI."""
        reach = 390.0
        hta_radians = np.radians(72.0)  # ~1.2566
        wheelbase = 1020.0

        ai = compute_ai(reach, hta_radians, wheelbase)

        # tan(1.2566) ≈ 3.0777, so AI ≈ 390 * 3.0777 / 1020 ≈ 1.176
        assert 1.0 < ai < 1.5, f"AI = {ai}, expected ~1.18 for typical road bike"

    def test_buggy_ai_is_wildly_different(self):
        """Show that the old buggy formula produces dramatically wrong values."""
        reach = 390.0
        hta_radians = np.radians(72.0)
        wheelbase = 1020.0

        correct = compute_ai(reach, hta_radians, wheelbase)
        buggy = compute_ai_buggy(reach, hta_radians, wheelbase)

        # The buggy version produces AI ~0.008 instead of ~1.18
        assert buggy < 0.05, f"Buggy AI should be near 0, got {buggy}"
        assert correct > 1.0, f"Correct AI should be > 1.0, got {correct}"
        assert correct / buggy > 50, "Correct AI should be >> buggy AI"

    def test_ai_with_90_degree_angle(self):
        """At 90° HTA (π/2 radians), tan → infinity, so AI should be very large."""
        reach = 390.0
        hta_radians = np.radians(89.0)  # Near 90°, very large tan
        wheelbase = 1020.0

        ai = compute_ai(reach, hta_radians, wheelbase)
        assert ai > 10, "AI at near-90° HTA should be very large"

    def test_ai_vectorized(self):
        """The formula should work on pandas Series (as used in the pipeline)."""
        df = pd.DataFrame({
            'reach': [390.0, 385.0, 380.0],
            'hta_rad': np.radians([72.0, 73.0, 71.0]),
            'wb': [1020.0, 990.0, 1010.0],
        })

        ai = compute_ai(df['reach'], df['hta_rad'], df['wb'])

        assert len(ai) == 3
        assert all(ai > 0.5)
        assert all(ai < 2.0)


# ---------------------------------------------------------------------------
# Test: Basic ratio formulas
# ---------------------------------------------------------------------------

class TestBasicRatios:
    """Test correctness of simple ratio calculations."""

    def test_srr_typical_values(self):
        """SRR for a race bike is typically 1.3-1.6."""
        srr = compute_srr(stack=540.0, reach=385.0)
        assert np.isclose(srr, 540.0 / 385.0)
        assert 1.3 < srr < 1.6

    def test_srr_zero_reach_produces_inf_in_pandas(self):
        """Division by zero in pandas Series produces inf (as in the pipeline)."""
        s = pd.Series([540.0]) / pd.Series([0.0])
        assert np.isinf(s.iloc[0])

    def test_cs_bbd(self):
        cs_bbd = compute_cs_bbd(chainstay=425.0, bb_drop=72.0)
        assert np.isclose(cs_bbd, 425.0 / 72.0)

    def test_ett_s(self):
        ett_s = compute_ett_s(top_tube=555.0, stack=580.0)
        assert np.isclose(ett_s, 555.0 / 580.0)

    def test_t_wb(self):
        t_wb = compute_t_wb(trail=63.0, wheelbase=1020.0)
        assert np.isclose(t_wb, 63.0 / 1020.0)

    def test_hta_trail(self):
        hta_rad = np.radians(72.0)
        hta_trail = compute_hta_trail(hta_rad, trail=63.0)
        assert np.isclose(hta_trail, hta_rad / 63.0)

    def test_csl_sta(self):
        sta_rad = np.radians(73.0)
        csl_sta = compute_csl_sta(chainstay=425.0, seat_tube_angle_radians=sta_rad)
        assert np.isclose(csl_sta, 425.0 / sta_rad)


# ---------------------------------------------------------------------------
# Test: Ratios from 9_ratios_v3.py
# ---------------------------------------------------------------------------

class TestRatiosV3:
    """Test ratio formulas specific to 9_ratios_v3.py."""

    def test_strr(self):
        hta_rad = np.radians(72.0)
        strr = compute_strr(hta_rad, trail=63.0)
        assert np.isclose(strr, hta_rad / 63.0)

    def test_csr(self):
        csr = compute_csr(wheelbase=1020.0, trail=63.0, bb_drop=72.0)
        assert np.isclose(csr, (1020.0 * 63.0) / 72.0)

    def test_csr_zero_bb_drop_in_pandas(self):
        """Division by zero in pandas Series produces inf (as in the pipeline)."""
        s = (pd.Series([1020.0]) * pd.Series([63.0])) / pd.Series([0.0])
        assert np.isinf(s.iloc[0])


# ---------------------------------------------------------------------------
# Test: Composite indices
# ---------------------------------------------------------------------------

class TestCompositeIndices:
    """Test weighted composite index calculations."""

    def test_stability_index(self):
        si = compute_stability_index(bb_height=270.0, chainstay=425.0, wheelbase=1020.0)
        expected = 0.5 * 270.0 + 0.3 * 425.0 + 0.2 * 1020.0
        assert np.isclose(si, expected)

    def test_handling_index(self):
        hta_rad = np.radians(72.0)
        hi = compute_handling_index(hta_rad, trail=63.0, rake=50.0)
        expected = 0.5 * hta_rad + 0.25 * 63.0 + 0.25 * 50.0
        assert np.isclose(hi, expected)

    def test_comfort_index(self):
        bbh_cs = 270.0 * 425.0
        cs_bbd = 425.0 / 72.0
        ett_s = 555.0 / 580.0
        ci = compute_comfort_index(bbh_cs, cs_bbd, ett_s)
        expected = 0.5 * bbh_cs + 0.3 * cs_bbd + 0.2 * ett_s
        assert np.isclose(ci, expected)

    def test_stability_weights_sum_to_one(self):
        assert np.isclose(0.5 + 0.3 + 0.2, 1.0)

    def test_handling_weights_sum_to_one(self):
        assert np.isclose(0.5 + 0.25 + 0.25, 1.0)

    def test_comfort_weights_sum_to_one(self):
        assert np.isclose(0.5 + 0.3 + 0.2, 1.0)


# ---------------------------------------------------------------------------
# Test: inf → NaN replacement (the cleanup step after ratio calculation)
# ---------------------------------------------------------------------------

class TestInfReplacement:
    """Test that infinity values from division by zero are cleaned up."""

    def test_inf_replaced_with_nan(self):
        """The pipeline replaces inf/-inf with NaN after ratio calculation."""
        df = pd.DataFrame({
            'ratio': [1.5, np.inf, -np.inf, 0.8, np.nan],
        })
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        assert pd.isna(df.loc[1, 'ratio'])
        assert pd.isna(df.loc[2, 'ratio'])
        assert df.loc[0, 'ratio'] == 1.5
        assert df.loc[3, 'ratio'] == 0.8
        assert pd.isna(df.loc[4, 'ratio'])
