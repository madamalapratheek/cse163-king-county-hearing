"""
Name: Krishna Pratheek Madamala
Section: AC
CSE 163, Autumn 2026
Final Project, Part 2: Exploratory Data Analysis

Ranks tracts by limited-English rate and count, classifies tracts into
high-barrier overlap groups at quantile cutoffs, and computes summary stats.
"""

import pandas as pd

TOP_QUARTILE = 0.75
TOP_TERCILE = 2 / 3
GROUPS = ['High on both', 'Language only', 'Hearing only', 'Neither']


def rank_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank tracts in df with a non-missing lang_rate from highest to lowest.

    Ties are broken by ascending GEOID. Returns a copy sorted by lang_rate
    with added integer columns rank_pct (by rate) and rank_count (by
    lang_num).
    """
    valid = df.dropna(subset=['lang_rate']).copy()
    by_pct = valid.sort_values(
        ['lang_rate', 'GEOID'], ascending=[False, True])
    by_pct['rank_pct'] = range(1, len(by_pct) + 1)
    counts = by_pct.sort_values(
        ['lang_num', 'GEOID'], ascending=[False, True])
    counts['rank_count'] = range(1, len(counts) + 1)
    return by_pct.merge(counts[['GEOID', 'rank_count']], on='GEOID')


def cutoff_values(df: pd.DataFrame, quantile: float) -> tuple[float, float]:
    """
    Return (language_threshold, hearing_threshold) for lang_rate and hear_rate
    in df at the given quantile.
    """
    return (float(df['lang_rate'].quantile(quantile)),
            float(df['hear_rate'].quantile(quantile)))


def classify(
    df: pd.DataFrame, quantile: float
) -> tuple[pd.DataFrame, float, float]:
    """
    Label each tract in df as high on both, one, or neither barrier at the
    given quantile cutoff.

    Tracts whose rate meets or exceeds the cutoff are marked high so ties at
    the cutoff are included. Returns (labeled_df, lang_cut, hear_cut).
    """
    lang_cut, hear_cut = cutoff_values(df, quantile)
    out = df.copy()
    out['high_lang'] = out['lang_rate'] >= lang_cut
    out['high_hear'] = out['hear_rate'] >= hear_cut
    out['group'] = 'Neither'
    out.loc[out['high_lang'] & ~out['high_hear'], 'group'] = 'Language only'
    out.loc[~out['high_lang'] & out['high_hear'], 'group'] = 'Hearing only'
    out.loc[out['high_lang'] & out['high_hear'], 'group'] = 'High on both'
    return out, lang_cut, hear_cut


def group_counts(labeled: pd.DataFrame) -> pd.Series:
    """
    Count tracts in labeled for each of the four barrier groups.

    Returns a Series indexed in GROUPS order with 0 for any absent group.
    """
    counts = labeled['group'].value_counts()
    return counts.reindex(GROUPS, fill_value=0)


def count_ties_at_cutoff(
    df: pd.DataFrame, quantile: float
) -> tuple[int, int]:
    """
    Return (lang_ties, hear_ties), the counts of tracts in df whose rate
    equals the cutoff exactly at the given quantile.
    """
    lang_cut, hear_cut = cutoff_values(df, quantile)
    return (int((df['lang_rate'] == lang_cut).sum()),
            int((df['hear_rate'] == hear_cut).sum()))


def new_at_wider_cutoff(
    narrow: pd.DataFrame, wide: pd.DataFrame
) -> list[str]:
    """
    Return a sorted list of GEOIDs labeled 'High on both' in wide but not in
    narrow.
    """
    narrow_ids = set(narrow.loc[narrow['group'] == 'High on both', 'GEOID'])
    wide_ids = set(wide.loc[wide['group'] == 'High on both', 'GEOID'])
    return sorted(wide_ids - narrow_ids)


def summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Return a seven-number summary (mean, std, min, 25%, 50%, 75%, max) for
    each column listed in columns from df.
    """
    stats = df[columns].describe().T
    return stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
