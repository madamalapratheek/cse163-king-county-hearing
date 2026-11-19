"""
Name: Krishna Pratheek Madamala
Section: AC
CSE 163, Autumn 2026
Final Project, Part 2: Exploratory Data Analysis

Unit tests for rate calculations, ACS cleaning, MOE aggregation, ranking,
and cutoff classification using synthetic in-file test data. Run with:
`python test_analysis.py`.
"""

import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import analysis as an
import data_prep as dp


def write_acs(path: Path, header: list[str],
              rows: list[list[str]]) -> None:
    """
    Write a Census-style JSON array to path with header as the first row and
    rows as data rows.
    """
    with open(path, 'w') as out_file:
        json.dump([header] + rows, out_file)


def make_rates(lang: list[float], hear: list[float]) -> pd.DataFrame:
    """
    Build a small synthetic tract DataFrame with lang_rate and hear_rate
    columns for testing ranking and classification functions.
    """
    count = len(lang)
    data = pd.DataFrame({
        'GEOID': ['g' + str(i) for i in range(count)],
        'lang_rate': lang, 'hear_rate': hear,
        'lang_num': range(count), 'lang_den': [100] * count
    })

    return data


def test_compute_rate_by_hand() -> None:
    """
    Test compute_rate on a basic hand calculation of 10 out of 40 (25%).
    """
    rate = dp.compute_rate(pd.Series([10.0]), pd.Series([40.0]))

    assert math.isclose(rate.iloc[0], 25.0)


def test_zero_denominator_gives_nan_not_error() -> None:
    """
    Test that compute_rate returns NaN for zero denominators without error.
    """
    rate = dp.compute_rate(pd.Series([0.0, 5.0]), pd.Series([0.0, 10.0]))

    assert np.isnan(rate.iloc[0])
    assert math.isclose(rate.iloc[1], 50.0)


def test_language_rate_sums_twelve_fields() -> None:
    """
    Test build_rates on all twelve B16004 language numerator fields.
    """
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'E': 48.0 for field in dp.LANG_DEN})
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    data = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(data, dp.LANG_NUM, dp.LANG_DEN, 'lang')

    assert out['lang_num'].iloc[0] == 12
    assert math.isclose(out['lang_rate'].iloc[0], 25.0)


def test_hearing_rate_hand_calculation() -> None:
    """
    Test build_rates on hand-checked B18102 hearing counts (15 out of 100).
    """
    row: dict[str, object] = {'GEOID': '53033000100', 'NAME': 'a'}
    for field, value in zip(dp.HEAR_NUM, [4, 6, 2, 3]):
        row[field + 'E'] = float(value)
        row[field + 'M'] = 0.0
    for field, value in zip(dp.HEAR_DEN, [20, 30, 10, 40]):
        row[field + 'E'] = float(value)
        row[field + 'M'] = 0.0
    out = dp.build_rates(
        pd.DataFrame([row]), dp.HEAR_NUM, dp.HEAR_DEN, 'hear'
    )

    assert out['hear_den'].iloc[0] == 100
    assert math.isclose(out['hear_rate'].iloc[0], 15.0)


def test_missing_estimate_makes_rate_nan_not_zero() -> None:
    """
    Test that one missing numerator component makes the rate NaN instead of 0.
    """
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    row['B16004_050E'] = np.nan
    row['B16004_046E'] = 40.0
    data = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(data, dp.LANG_NUM, dp.LANG_DEN, 'lang')

    assert np.isnan(out['lang_num'].iloc[0])
    assert np.isnan(out['lang_rate'].iloc[0])


def test_census_missing_codes_become_nan_and_zeros_stay() -> None:
    """
    Test that load_acs_json masks Census sentinel codes but preserves zeros.
    """
    header = ['NAME', 'B16004_046E', 'B16004_046M', 'state', 'county',
              'tract']
    rows = [['t1', '-666666666', '-555555555', '53', '033', '000100'],
            ['t2', '0', '5', '53', '033', '000200']]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'acs.json'
        write_acs(path, header, rows)
        data = dp.load_acs_json(path)

    assert np.isnan(data.loc[0, 'B16004_046E'])
    assert np.isnan(data.loc[0, 'B16004_046M'])
    assert data.loc[1, 'B16004_046E'] == 0


def test_geoid_keeps_leading_zeros_and_has_eleven_digits() -> None:
    """
    Test that load_acs_json preserves leading zeros in 11-digit GEOID codes.
    """
    header = ['NAME', 'NAME', 'B16004_046E', 'state', 'county', 'tract']
    rows = [['a', 'a', '7', '53', '033', '000100']]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / 'acs.json'
        write_acs(path, header, rows)
        data = dp.load_acs_json(path)

    assert data['GEOID'].iloc[0] == '53033000100'
    assert len(data['GEOID'].iloc[0]) == 11


def test_duplicate_geoid_is_rejected() -> None:
    """
    Test that check_unique raises a ValueError on duplicate GEOIDs.
    """
    data = pd.DataFrame({'GEOID': ['1', '1']})
    raised = False
    try:
        dp.check_unique(data, 'GEOID', 'test table')
    except ValueError:
        raised = True

    assert raised


def test_unmatched_geoids_are_reported() -> None:
    """
    Test that join_one_to_one reports unmatched GEOIDs from both tables.
    """
    left = pd.DataFrame({'GEOID': ['a', 'b'], 'x': [1, 2]})
    right = pd.DataFrame({'GEOID': ['b', 'c'], 'y': [3, 4]})
    merged, unmatched = dp.join_one_to_one(left, right)

    assert sorted(unmatched['GEOID']) == ['a', 'c']
    assert len(merged) == 3


def test_proportion_moe_without_denominator_error() -> None:
    """
    Test proportion_moe when the denominator has zero margin of error.
    """
    moe = dp.proportion_moe(
        pd.Series([50.0]), pd.Series([100.0]),
        pd.Series([10.0]), pd.Series([0.0])
    )

    assert math.isclose(moe.iloc[0], 10.0)


def test_sum_moe_counts_only_largest_zero_estimate_moe() -> None:
    """
    Test that sum_moe includes only the single largest zero-estimate MOE.
    """
    data = pd.DataFrame([{
        'A_E': 10.0, 'A_M': 3.0,
        'B_E': 0.0, 'B_M': 4.0,
        'C_E': 0.0, 'C_M': 2.0,
    }])
    moe = dp.sum_moe(data, ['A_', 'B_', 'C_'])

    assert math.isclose(moe.iloc[0], 5.0)


def test_sum_moe_returns_nan_when_any_component_missing() -> None:
    """
    Test that sum_moe returns NaN when any component estimate or MOE is NaN.
    """
    data = pd.DataFrame([{
        'A_E': 10.0, 'A_M': 3.0,
        'B_E': 5.0, 'B_M': np.nan,
    }])

    assert np.isnan(dp.sum_moe(data, ['A_', 'B_']).iloc[0])


def test_ranking_breaks_ties_by_geoid_and_skips_nan() -> None:
    """
    Test rank_language tie-breaking by GEOID and exclusion of NaN rates.
    """
    data = make_rates([10.0, 30.0, 30.0, np.nan], [1.0, 2.0, 3.0, 4.0])
    ranked = an.rank_language(data)

    assert list(ranked['GEOID']) == ['g1', 'g2', 'g0']
    assert list(ranked['rank_pct']) == [1, 2, 3]


def test_tied_values_at_cutoff_are_all_high() -> None:
    """
    Test that classify and count_ties_at_cutoff include tied cutoff values.
    """
    data = make_rates([1.0, 2.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 3.0, 3.0])
    labeled, lang_cut, _ = an.classify(data, 0.75)

    assert lang_cut == 3.0
    assert labeled['high_lang'].sum() == 3
    assert an.count_ties_at_cutoff(data, 0.75) == (3, 3)


def test_group_labels_cover_all_four_cases() -> None:
    """
    Test that classify assigns all four overlap group labels accurately.
    """
    data = make_rates([1.0, 1.0, 9.0, 9.0], [1.0, 9.0, 1.0, 9.0])
    labeled, _, _ = an.classify(data, 0.5)
    groups = dict(zip(labeled['GEOID'], labeled['group']))

    assert groups == {'g0': 'Neither', 'g1': 'Hearing only',
                      'g2': 'Language only', 'g3': 'High on both'}


def test_wider_cutoff_contains_narrow_group_and_reports_new_tracts() -> None:
    """
    Test new_at_wider_cutoff returns tracts newly added at the wider cutoff.
    """
    vals = [float(i) for i in range(12)]
    data = make_rates(vals, vals)
    narrow, _, _ = an.classify(data, 0.75)
    wide, _, _ = an.classify(data, 2 / 3)
    new_ids = an.new_at_wider_cutoff(narrow, wide)
    narrow_both = set(narrow.loc[narrow['group'] == 'High on both', 'GEOID'])
    wide_both = set(wide.loc[wide['group'] == 'High on both', 'GEOID'])

    assert narrow_both <= wide_both
    assert len(new_ids) == len(wide_both) - len(narrow_both)


def test_group_counts_includes_empty_groups() -> None:
    """
    Test group_counts includes zero-count groups in GROUPS order.
    """
    data = make_rates([1.0, 2.0], [1.0, 2.0])
    labeled, _, _ = an.classify(data, 0.5)
    counts = an.group_counts(labeled)

    assert list(counts.index) == an.GROUPS
    assert counts['Hearing only'] == 0


def main() -> None:
    """
    Run all 17 unit test functions and print a confirmation message.
    """
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
