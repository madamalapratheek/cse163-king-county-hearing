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


def rank_language(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rank tracts in data with a non-missing lang_rate from highest to lowest.
    Break ties by ascending GEOID and add integer columns rank_pct and
    rank_count.
    """
    valid = data.dropna(subset=['lang_rate']).copy()
    by_pct = valid.sort_values(
        ['lang_rate', 'GEOID'], ascending=[False, True]
    )
    by_pct['rank_pct'] = range(1, len(by_pct) + 1)
    counts = by_pct.sort_values(
        ['lang_num', 'GEOID'], ascending=[False, True]
    )
    counts['rank_count'] = range(1, len(counts) + 1)
    ranked = by_pct.merge(counts[['GEOID', 'rank_count']], on='GEOID')

    return ranked


def cutoff_values(data: pd.DataFrame, quantile: float) -> tuple[float, float]:
    """
    Return the language and hearing rate threshold values in data at the
    given quantile.
    """
    lang_cut = float(data['lang_rate'].quantile(quantile))
    hear_cut = float(data['hear_rate'].quantile(quantile))

    return lang_cut, hear_cut


def classify(data: pd.DataFrame,
             quantile: float) -> tuple[pd.DataFrame, float, float]:
    """
    Label each tract in data as high on both, language only, hearing only,
    or neither at the given quantile cutoff. Include tied values at the cutoff.
    """
    lang_cut, hear_cut = cutoff_values(data, quantile)
    labeled = data.copy()
    labeled['high_lang'] = labeled['lang_rate'] >= lang_cut
    labeled['high_hear'] = labeled['hear_rate'] >= hear_cut
    labeled['group'] = 'Neither'
    labeled.loc[labeled['high_lang'] & ~labeled['high_hear'], 'group'] = (
        'Language only'
    )
    labeled.loc[~labeled['high_lang'] & labeled['high_hear'], 'group'] = (
        'Hearing only'
    )
    labeled.loc[labeled['high_lang'] & labeled['high_hear'], 'group'] = (
        'High on both'
    )

    return labeled, lang_cut, hear_cut


def group_counts(labeled: pd.DataFrame) -> pd.Series:
    """
    Return tract counts for each group in labeled, ordered by GROUPS with
    0 for any empty groups.
    """
    counts = labeled['group'].value_counts()

    return counts.reindex(GROUPS, fill_value=0)


def count_ties_at_cutoff(data: pd.DataFrame,
                         quantile: float) -> tuple[int, int]:
    """
    Return the number of tracts in data whose rate matches the cutoff
    threshold exactly at the given quantile.
    """
    lang_cut, hear_cut = cutoff_values(data, quantile)
    lang_ties = int((data['lang_rate'] == lang_cut).sum())
    hear_ties = int((data['hear_rate'] == hear_cut).sum())

    return lang_ties, hear_ties


def new_at_wider_cutoff(narrow: pd.DataFrame, wide: pd.DataFrame) -> list[str]:
    """
    Return a sorted list of GEOIDs labeled High on both at the wider cutoff
    that were not in the narrow cutoff.
    """
    narrow_ids = set(narrow.loc[narrow['group'] == 'High on both', 'GEOID'])
    wide_ids = set(wide.loc[wide['group'] == 'High on both', 'GEOID'])

    return sorted(wide_ids - narrow_ids)


def summary_stats(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Return a seven-number summary table (mean, std, min, 25%, 50%, 75%, max)
    for each column in columns from data.
    """
    stats = data[columns].describe().T

    return stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
