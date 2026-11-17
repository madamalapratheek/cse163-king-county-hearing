"""Tests for the rate calculations, cleaning, ranking and cutoffs.

Run with: pytest test_analysis.py
All test data is built inside this file.
"""

import json

import numpy as np
import pandas as pd
import pytest

import analysis as an
import data_prep as dp


def write_acs(path, header, rows):
    """Write a Census-style JSON file (first row holds column names)."""
    with open(path, 'w') as f:
        json.dump([header] + rows, f)


def test_compute_rate_by_hand():
    """10 of 40 is 25 percent."""
    rate = dp.compute_rate(pd.Series([10.0]), pd.Series([40.0]))
    assert rate.iloc[0] == pytest.approx(25.0)


def test_zero_denominator_gives_nan_not_error():
    rate = dp.compute_rate(pd.Series([0.0, 5.0]), pd.Series([0.0, 10.0]))
    assert np.isnan(rate.iloc[0])
    assert rate.iloc[1] == pytest.approx(50.0)


def test_language_rate_sums_twelve_fields():
    """Each of the twelve numerator fields is 1; the denominator is 48."""
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'E': 48.0 for field in dp.LANG_DEN})
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    df = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(df, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    assert out['lang_num'].iloc[0] == 12
    assert out['lang_rate'].iloc[0] == pytest.approx(25.0)


def test_hearing_rate_hand_calculation():
    """Counts 4+6+2+3 = 15 out of 20+30+10+40 = 100 gives 15 percent."""
    row = {'GEOID': '53033000100', 'NAME': 'a'}
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


def test_missing_estimate_makes_rate_nan_not_zero():
    """One missing piece of the numerator makes the whole rate missing."""
    row = {field + 'E': 1.0 for field in dp.LANG_NUM}
    row.update({field + 'M': 0.0 for field in dp.LANG_NUM + dp.LANG_DEN})
    row['B16004_050E'] = np.nan
    row['B16004_046E'] = 40.0
    df = pd.DataFrame([dict(row, GEOID='53033000100', NAME='a')])
    out = dp.build_rates(df, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    assert np.isnan(out['lang_num'].iloc[0])
    assert np.isnan(out['lang_rate'].iloc[0])


def test_census_missing_codes_become_nan_and_zeros_stay(tmp_path):
    header = ['NAME', 'B16004_046E', 'B16004_046M', 'state', 'county',
              'tract']
    rows = [['t1', '-666666666', '-555555555', '53', '033', '000100'],
            ['t2', '0', '5', '53', '033', '000200']]
    path = tmp_path / 'acs.json'
    write_acs(path, header, rows)
    df = dp.load_acs_json(path)
    assert np.isnan(df.loc[0, 'B16004_046E'])
    assert np.isnan(df.loc[0, 'B16004_046M'])
    assert df.loc[1, 'B16004_046E'] == 0


def test_geoid_keeps_leading_zeros_and_has_eleven_digits(tmp_path):
    header = ['NAME', 'NAME', 'B16004_046E', 'state', 'county', 'tract']
    rows = [['a', 'a', '7', '53', '033', '000100']]
    path = tmp_path / 'acs.json'
    write_acs(path, header, rows)
    df = dp.load_acs_json(path)
    assert df['GEOID'].iloc[0] == '53033000100'
    assert len(df['GEOID'].iloc[0]) == 11


def test_duplicate_geoid_is_rejected():
    df = pd.DataFrame({'GEOID': ['1', '1']})
    with pytest.raises(ValueError):
        dp.check_unique(df, 'GEOID', 'test table')


def test_unmatched_geoids_are_reported():
    left = pd.DataFrame({'GEOID': ['a', 'b'], 'x': [1, 2]})
    right = pd.DataFrame({'GEOID': ['b', 'c'], 'y': [3, 4]})
    merged, unmatched = dp.join_one_to_one(left, right, 'test')
    assert sorted(unmatched['GEOID']) == ['a', 'c']
    assert len(merged) == 3


def test_proportion_moe_without_denominator_error():
    """With no error in the denominator, MOE is num_moe / den * 100."""
    moe = dp.proportion_moe(
        pd.Series([50.0]), pd.Series([100.0]),
        pd.Series([10.0]), pd.Series([0.0]))
    assert moe.iloc[0] == pytest.approx(10.0)


def make_rates(lang, hear):
    n = len(lang)
    return pd.DataFrame({
        'GEOID': [f'g{i}' for i in range(n)],
        'lang_rate': lang, 'hear_rate': hear,
        'lang_num': range(n), 'lang_den': [100] * n})


def test_ranking_breaks_ties_by_geoid_and_skips_nan():
    df = make_rates([10.0, 30.0, 30.0, np.nan], [1, 2, 3, 4])
    ranked = an.rank_language(df)
    assert list(ranked['GEOID']) == ['g1', 'g2', 'g0']
    assert list(ranked['rank_pct']) == [1, 2, 3]


def test_tied_values_at_cutoff_are_all_high():
    """Rates 1, 2, 3, 3, 3 have a 0.75 quantile of 3, so three are high."""
    df = make_rates([1.0, 2.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 3.0, 3.0])
    labeled, lang_cut, _ = an.classify(df, 0.75)
    assert lang_cut == 3.0
    assert labeled['high_lang'].sum() == 3
    assert an.count_ties_at_cutoff(df, 0.75) == (3, 3)


def test_group_labels_cover_all_four_cases():
    df = make_rates([1.0, 1.0, 9.0, 9.0], [1.0, 9.0, 1.0, 9.0])
    labeled, _, _ = an.classify(df, 0.5)
    groups = dict(zip(labeled['GEOID'], labeled['group']))
    assert groups == {'g0': 'Neither', 'g1': 'Hearing only',
                      'g2': 'Language only', 'g3': 'High on both'}


def test_wider_cutoff_contains_narrow_group_and_reports_new_tracts():
    df = make_rates(list(range(12)), list(range(12)))
    narrow, _, _ = an.classify(df, 0.75)
    wide, _, _ = an.classify(df, 2 / 3)
    new_ids = an.new_at_wider_cutoff(narrow, wide)
    assert set(narrow.loc[narrow['group'] == 'High on both', 'GEOID']) <= \
        set(wide.loc[wide['group'] == 'High on both', 'GEOID'])
    assert len(new_ids) == (wide['group'] == 'High on both').sum() - \
        (narrow['group'] == 'High on both').sum()


def test_group_counts_includes_empty_groups():
    df = make_rates([1.0, 2.0], [1.0, 2.0])
    labeled, _, _ = an.classify(df, 0.5)
    counts = an.group_counts(labeled)
    assert list(counts.index) == an.GROUPS
    assert counts['Hearing only'] == 0
