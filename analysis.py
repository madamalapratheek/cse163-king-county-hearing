"""Rank tracts, classify them at percentile cutoffs, and summarize."""

TOP_QUARTILE = 0.75
TOP_TERCILE = 2 / 3
GROUPS = ['High on both', 'Language only', 'Hearing only', 'Neither']


def rank_language(df):
    """Rank tracts with a valid language rate, highest rate first.

    Ties in the rate are broken by GEOID. Adds rank_pct (rank by rate)
    and rank_count (rank by estimated count, ties broken by GEOID).
    """
    valid = df.dropna(subset=['lang_rate']).copy()
    by_pct = valid.sort_values(
        ['lang_rate', 'GEOID'], ascending=[False, True])
    by_pct['rank_pct'] = range(1, len(by_pct) + 1)
    counts = by_pct.sort_values(
        ['lang_num', 'GEOID'], ascending=[False, True])
    counts['rank_count'] = range(1, len(counts) + 1)
    return by_pct.merge(counts[['GEOID', 'rank_count']], on='GEOID')


def cutoff_values(df, quantile):
    """Return (language threshold, hearing threshold) at a quantile."""
    return (df['lang_rate'].quantile(quantile),
            df['hear_rate'].quantile(quantile))


def classify(df, quantile):
    """Label each tract as high on both, one, or neither measure.

    A tract is high on a measure when its rate meets or exceeds the
    cutoff, so tied values at the cutoff are all included. Returns the
    labeled table and the two thresholds.
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


def group_counts(labeled):
    """Count tracts in each of the four groups, always in GROUPS order."""
    counts = labeled['group'].value_counts()
    return counts.reindex(GROUPS, fill_value=0)


def count_ties_at_cutoff(df, quantile):
    """Count tracts whose rate equals the cutoff exactly, per measure."""
    lang_cut, hear_cut = cutoff_values(df, quantile)
    return (int((df['lang_rate'] == lang_cut).sum()),
            int((df['hear_rate'] == hear_cut).sum()))


def new_at_wider_cutoff(narrow, wide):
    """GEOIDs high on both at the wider cutoff but not the narrow one."""
    narrow_ids = set(narrow.loc[narrow['group'] == 'High on both', 'GEOID'])
    wide_ids = set(wide.loc[wide['group'] == 'High on both', 'GEOID'])
    return sorted(wide_ids - narrow_ids)


def summary_stats(df, columns):
    """Mean, standard deviation, min, quartiles and max per column."""
    stats = df[columns].describe().T
    return stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
