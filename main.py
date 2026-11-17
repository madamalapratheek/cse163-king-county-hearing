"""Run the full EDA: load, join, rank, classify, plot, and save results.

Usage: python main.py
Reads data/raw/lang.json, data/raw/hearing.json and the extracted tract
shapefile in data/tl_2024_53_tract/. Writes tables, figures and a
summary.json file to results/.
"""

import json
import os

import data_prep as dp
import analysis as an
import plots

LANG_JSON = 'data/raw/lang.json'
HEAR_JSON = 'data/raw/hearing.json'
TRACT_SHP = 'data/tl_2024_53_tract/tl_2024_53_tract.shp'
OUT = 'results'
STAT_COLUMNS = ['lang_rate', 'hear_rate', 'lang_num', 'lang_den',
                'hear_den']


def load_and_join():
    """Load both ACS tables and the boundaries, then join them on GEOID.

    Returns (language_hearing, with_geometry, report) where report holds
    counts used in the write-up.
    """
    lang_raw = dp.load_acs_json(LANG_JSON)
    hear_raw = dp.load_acs_json(HEAR_JSON)
    tracts = dp.load_tracts(TRACT_SHP)
    for df, name in [(lang_raw, 'language'), (hear_raw, 'hearing')]:
        dp.check_unique(df, 'GEOID', name)
        assert (df['GEO_ID'].str[-11:] == df['GEOID']).all()
    lang = dp.build_rates(lang_raw, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    hear = dp.build_rates(hear_raw, dp.HEAR_NUM, dp.HEAR_DEN, 'hear')
    lh, miss1 = dp.join_one_to_one(
        lang, hear.drop(columns='NAME'), 'language + hearing')
    full, miss2 = dp.join_one_to_one(tracts, lh, 'with geometry')
    report = {
        'tracts_language': len(lang_raw),
        'tracts_hearing': len(hear_raw),
        'tracts_boundaries': len(tracts),
        'unmatched_language_hearing': len(miss1),
        'unmatched_geometry': len(miss2),
        'language_columns_raw': lang_raw.shape[1],
        'hearing_columns_raw': hear_raw.shape[1],
        'zero_or_missing_lang_den': int((lh['lang_den'].fillna(0) <= 0)
                                        .sum()),
        'zero_or_missing_hear_den': int((lh['hear_den'].fillna(0) <= 0)
                                        .sum()),
        'null_lang_rate': int(lh['lang_rate'].isna().sum()),
        'null_hear_rate': int(lh['hear_rate'].isna().sum()),
    }
    return lh, full, report


def describe_joins(lh, full, both):
    """Shapes and summary statistics for each joined dataset."""
    return {
        'language_hearing_shape': list(lh.shape),
        'with_geometry_shape': list(full.shape),
        'both_valid_rows': len(both),
        'stats_language_hearing': an.summary_stats(
            lh, STAT_COLUMNS).round(2).to_dict('index'),
        'stats_both_valid': an.summary_stats(
            both, STAT_COLUMNS).round(2).to_dict('index'),
        'correlation_pearson': round(
            both['lang_rate'].corr(both['hear_rate']), 3),
        'correlation_spearman': round(
            both['lang_rate'].corr(both['hear_rate'], method='spearman'),
            3),
    }


def language_results(lh):
    """Top-ten tracts by percentage and the rank comparison example."""
    ranked = an.rank_language(lh)
    cols = ['GEOID', 'NAME', 'lang_num', 'lang_den', 'lang_rate',
            'lang_moe', 'rank_pct', 'rank_count']
    top = ranked.head(10)[cols]
    top_by_count = ranked.sort_values('rank_count').head(10)[cols]
    example = top.sort_values('rank_count').iloc[-1]
    top.to_csv(f'{OUT}/top10_language_pct.csv', index=False)
    top_by_count.to_csv(f'{OUT}/top10_language_count.csv', index=False)
    overlap = len(set(top['GEOID']) & set(top_by_count['GEOID']))
    return ranked, top, {
        'n_ranked': len(ranked),
        'top10_pct': top.round(2).to_dict('records'),
        'top10_count': top_by_count.round(2).to_dict('records'),
        'top10_overlap_pct_and_count': overlap,
        'rank_example': {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in example.to_dict().items()},
        'top10_min_den': float(top['lang_den'].min()),
    }


def overlap_results(both, full):
    """Classify tracts at both cutoffs and attach geometry for maps."""
    results, labeled = {}, {}
    cutoffs = [('Top quartile', an.TOP_QUARTILE),
               ('Top tercile', an.TOP_TERCILE)]
    geo = full[['GEOID', 'geometry']]
    for name, quantile in cutoffs:
        table, lang_cut, hear_cut = an.classify(both, quantile)
        counts = an.group_counts(table)
        ties = an.count_ties_at_cutoff(both, quantile)
        results[name] = {
            'language_threshold': round(lang_cut, 3),
            'hearing_threshold': round(hear_cut, 3),
            'group_counts': counts.to_dict(),
            'tied_at_language_cutoff': ties[0],
            'tied_at_hearing_cutoff': ties[1],
            'high_both_share': round(counts['High on both']
                                     / len(both), 4),
        }
        labeled[name] = geo.merge(table, on='GEOID').pipe(
            lambda g: g.set_geometry('geometry'))
        table.to_csv(f'{OUT}/overlap_{name.lower().replace(" ", "_")}.csv',
                     index=False)
    quartile, tercile = labeled['Top quartile'], labeled['Top tercile']
    entering = an.new_at_wider_cutoff(quartile, tercile)
    results['entering_at_tercile'] = entering
    results['high_both_quartile_ids'] = sorted(
        quartile.loc[quartile['group'] == 'High on both', 'GEOID'])
    return results, labeled


def save_figure(fig, name, width=1000, height=700):
    """Save a figure as interactive HTML and, when possible, PNG."""
    fig.write_html(f'{OUT}/figures/{name}.html', include_plotlyjs='cdn')
    try:
        fig.write_image(f'{OUT}/figures/{name}.png', width=width,
                        height=height, scale=2)
    except Exception as err:  # PNG export needs kaleido and Chrome
        print(f'PNG export skipped for {name}: {err}')


def make_figures(full, ranked, both, labeled, results):
    """Create and save every figure."""
    lang_map = full.merge(ranked[['GEOID', 'rank_pct']], on='GEOID',
                          how='left').set_geometry('geometry')
    counts = {name: an.group_counts(labeled[name]) for name in labeled}
    save_figure(plots.language_rate_map(lang_map), 'fig1_language_map')
    save_figure(plots.scatter_rates(both), 'fig2_scatter', 900, 650)
    save_figure(plots.overlap_bar(counts), 'fig3_overlap_bar', 900, 600)
    for key, name in [('Top quartile', 'fig4a_overlap_map_quartile'),
                      ('Top tercile', 'fig4b_overlap_map_tercile')]:
        title = f"Overlap at the {key.lower()} cutoff"
        save_figure(plots.overlap_map(labeled[key], title), name)
    selectable = plots.overlap_map_selectable(labeled)
    selectable.write_html(f'{OUT}/figures/fig4_overlap_map_selectable.html',
                          include_plotlyjs='cdn')


def main():
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    lh, full, report = load_and_join()
    both = lh.dropna(subset=['lang_rate', 'hear_rate'])
    summary = {'report': report}
    summary.update(describe_joins(lh, full, both))
    ranked, top, lang_info = language_results(lh)
    summary['language'] = lang_info
    overlap, labeled = overlap_results(both, full)
    summary['overlap'] = overlap
    make_figures(full, ranked, both, labeled, overlap)
    with open(f'{OUT}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary['report'], indent=2))
    print('Done. Results are in the results/ folder.')


if __name__ == '__main__':
    main()
