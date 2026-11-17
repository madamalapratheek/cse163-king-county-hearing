"""
Pratheek Madamala
CSE 163 <SECTION>

Unit tests for rate calculations, ACS cleaning, MOE aggregation, ranking,
and cutoff classification using synthetic in-file test data. Can be run
with either `pytest test_analysis.py` or `python test_analysis.py`.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

import analysis as an
import data_prep as dp


def write_acs(
    path: Path, header: list[str], rows: list[list[str]]
) -> None:
    """
    Write a Census-style JSON array to path with header as the first row and
    rows as the remaining data rows.
    """
    with open(path, 'w') as out_file:
        json.dump([header] + rows, out_file)


def make_rates(lang: list[float], hear: list[float]) -> pd.DataFrame:
    """
    Build a small synthetic tract DataFrame with lang_rate and hear_rate
    columns for testing ranking and classification functions.
    """
    count = len(lang)
    return pd.DataFrame({
        'GEOID': [f'g{i}' for i in range(count)],
        'lang_rate': lang, 'hear_rate': hear,
        'lang_num': range(count), 'lang_den': [100] * count})


def test_compute_rate_by_hand() -> None:
    """Tests compute_rate on a basic hand-checked percentage (10 / 40)."""
    rate = dp.compute_rate(pd.Series([10.0]), pd.Series([40.0]))
    assert rate.iloc[0] == pytest.approx(25.0)


def test_zero_denominator_gives_nan_not_error() -> None:
    """Tests that compute_rate returns NaN for zero denominators."""
    rate = dp.compute_rate(pd.Series([0.0, 5.0]), pd.Series([0.0, 10.0]))
    assert np.isnan(rate.iloc[0])
    assert rate.iloc[1] == pytest.approx(50.0)


def test_language_rate_sums_twelve_fields() -> None:
    """Tests build_rates on all twelve B16004 language numerator fields."""
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'E': 48.0 for field in dp.LANG_DEN})
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    df = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(df, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    assert out['lang_num'].iloc[0] == 12
    assert out['lang_rate'].iloc[0] == pytest.approx(25.0)


def test_hearing_rate_hand_calculation() -> None:
    """Tests build_rates on hand-checked B18102 hearing counts (15 / 100)."""
    row: dict[str, object] = {'GEOID': '53033000100', 'NAME': 'a'}
    for field, value in zip(dp.HEAR_NUM, [4, 6, 2, 3]):
        row[field + 'E'] = float(value)
        row[field + 'M'] = 0.0
    for field, value in zip(dp.HEAR_DEN, [20, 30, 10, 40]):
        row[field + 'E'] = float(value)
        row[field + 'M'] = 0.0
    out = dp.build_rates(pd.DataFrame([row]), dp.HEAR_NUM, dp.HEAR_DEN,
                         'hear')
    assert out['hear_den'].iloc[0] == 100
    assert out['hear_rate'].iloc[0] == pytest.approx(15.0)


def test_missing_estimate_makes_rate_nan_not_zero() -> None:
    """Tests that one missing numerator component makes the rate NaN."""
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    row['B16004_050E'] = np.nan
    row['B16004_046E'] = 40.0
    df = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(df, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    assert np.isnan(out['lang_num'].iloc[0])
    assert np.isnan(out['lang_rate'].iloc[0])


def test_census_missing_codes_become_nan_and_zeros_stay() -> None:
    """Tests that load_acs_json masks Census sentinel codes but keeps 0."""
    header = ['NAME', 'B16004_046E', 'B16004_046M', 'state', 'county',
              'tract']
    rows = [['t1', '-666666666', '-555555555', '53', '033', '000100'],
            ['t2', '0', '5', '53', '033', '000200']]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'acs.json'
        write_acs(path, header, rows)
        df = dp.load_acs_json(path)
    assert np.isnan(df.loc[0, 'B16004_046E'])
    assert np.isnan(df.loc[0, 'B16004_046M'])
    assert df.loc[1, 'B16004_046E'] == 0


def test_geoid_keeps_leading_zeros_and_has_eleven_digits() -> None:
    """Tests that load_acs_json preserves leading zeros in 11-digit GEOID."""
    header = ['NAME', 'NAME', 'B16004_046E', 'state', 'county', 'tract']
    rows = [['a', 'a', '7', '53', '033', '000100']]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'acs.json'
        write_acs(path, header, rows)
        df = dp.load_acs_json(path)
    assert df['GEOID'].iloc[0] == '53033000100'
    assert len(df['GEOID'].iloc[0]) == 11


def test_duplicate_geoid_is_rejected() -> None:
    """Tests that check_unique raises ValueError on duplicate GEOIDs."""
    df = pd.DataFrame({'GEOID': ['1', '1']})
    with pytest.raises(ValueError):
        dp.check_unique(df, 'GEOID', 'test table')


def test_unmatched_geoids_are_reported() -> None:
    """Tests that join_one_to_one reports unmatched GEOIDs from both sides."""
    left = pd.DataFrame({'GEOID': ['a', 'b'], 'x': [1, 2]})
    right = pd.DataFrame({'GEOID': ['b', 'c'], 'y': [3, 4]})
    merged, unmatched = dp.join_one_to_one(left, right)
    assert sorted(unmatched['GEOID']) == ['a', 'c']
    assert len(merged) == 3


def test_proportion_moe_without_denominator_error() -> None:
    """Tests proportion_moe when denominator MOE is zero."""
    moe = dp.proportion_moe(
        pd.Series([50.0]), pd.Series([100.0]),
        pd.Series([10.0]), pd.Series([0.0]))
    assert moe.iloc[0] == pytest.approx(10.0)


def test_sum_moe_counts_only_largest_zero_estimate_moe() -> None:
    """Tests sum_moe applies the Census single-largest zero-estimate rule."""
    df = pd.DataFrame([{
        'A_E': 10.0, 'A_M': 3.0,
        'B_E': 0.0, 'B_M': 4.0,
        'C_E': 0.0, 'C_M': 2.0,
    }])
    moe = dp.sum_moe(df, ['A_', 'B_', 'C_'])
    assert moe.iloc[0] == pytest.approx(5.0)


def test_sum_moe_returns_nan_when_any_component_missing() -> None:
    """Tests that sum_moe returns NaN when any estimate or MOE is missing."""
    df = pd.DataFrame([{
        'A_E': 10.0, 'A_M': 3.0,
        'B_E': 5.0, 'B_M': np.nan,
    }])
    assert np.isnan(dp.sum_moe(df, ['A_', 'B_']).iloc[0])


def test_ranking_breaks_ties_by_geoid_and_skips_nan() -> None:
    """Tests rank_language tie-breaking by GEOID and NaN exclusion."""
    df = make_rates([10.0, 30.0, 30.0, np.nan], [1.0, 2.0, 3.0, 4.0])
    ranked = an.rank_language(df)
    assert list(ranked['GEOID']) == ['g1', 'g2', 'g0']
    assert list(ranked['rank_pct']) == [1, 2, 3]


def test_tied_values_at_cutoff_are_all_high() -> None:
    """Tests classify and count_ties_at_cutoff include tied cutoff values."""
    df = make_rates([1.0, 2.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 3.0, 3.0])
    labeled, lang_cut, _ = an.classify(df, 0.75)
    assert lang_cut == 3.0
    assert labeled['high_lang'].sum() == 3
    assert an.count_ties_at_cutoff(df, 0.75) == (3, 3)


def test_group_labels_cover_all_four_cases() -> None:
    """Tests classify assigns all four overlap group labels accurately."""
    df = make_rates([1.0, 1.0, 9.0, 9.0], [1.0, 9.0, 1.0, 9.0])
    labeled, _, _ = an.classify(df, 0.5)
    groups = dict(zip(labeled['GEOID'], labeled['group']))
    assert groups == {'g0': 'Neither', 'g1': 'Hearing only',
                      'g2': 'Language only', 'g3': 'High on both'}


def test_wider_cutoff_contains_narrow_group_and_reports_new_tracts() -> None:
    """Tests new_at_wider_cutoff returns tracts added at the wider cutoff."""
    vals = [float(i) for i in range(12)]
    df = make_rates(vals, vals)
    narrow, _, _ = an.classify(df, 0.75)
    wide, _, _ = an.classify(df, 2 / 3)
    new_ids = an.new_at_wider_cutoff(narrow, wide)
    narrow_both = set(narrow.loc[narrow['group'] == 'High on both', 'GEOID'])
    wide_both = set(wide.loc[wide['group'] == 'High on both', 'GEOID'])
    assert narrow_both <= wide_both
    assert len(new_ids) == len(wide_both) - len(narrow_both)


def test_group_counts_includes_empty_groups() -> None:
    """Tests group_counts includes zero-count groups in GROUPS order."""
    df = make_rates([1.0, 2.0], [1.0, 2.0])
    labeled, _, _ = an.classify(df, 0.5)
    counts = an.group_counts(labeled)
    assert list(counts.index) == an.GROUPS
    assert counts['Hearing only'] == 0


def main():
    test_compute_rate_by_hand()
    test_zero_denominator_gives_nan_not_error()
    test_language_rate_sums_twelve_fields()
    test_hearing_rate_hand_calculation()
    test_missing_estimate_makes_rate_nan_not_zero()
    test_census_missing_codes_become_nan_and_zeros_stay()
    test_geoid_keeps_leading_zeros_and_has_eleven_digits()
    test_duplicate_geoid_is_rejected()
    test_unmatched_geoids_are_reported()
    test_proportion_moe_without_denominator_error()
    test_sum_moe_counts_only_largest_zero_estimate_moe()
    test_sum_moe_returns_nan_when_any_component_missing()
    test_ranking_breaks_ties_by_geoid_and_skips_nan()
    test_tied_values_at_cutoff_are_all_high()
    test_group_labels_cover_all_four_cases()
    test_wider_cutoff_contains_narrow_group_and_reports_new_tracts()
    test_group_counts_includes_empty_groups()
    print('All tests passed.')


if __name__ == '__main__':
    main()
