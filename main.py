"""
Name: Krishna Pratheek Madamala
Section: AC
CSE 163, Autumn 2026
Final Project, Part 2: Exploratory Data Analysis

Runs the full King County older-adult language and hearing barriers EDA:
loads and joins ACS tables with tract boundaries, computes rankings and
overlap classifications, exports summary JSON/CSVs, and saves all figures.
"""

import json
import os
from typing import Any

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

import analysis as an
import data_prep as dp
import plots

LANG_JSON = 'data/raw/lang.json'
HEAR_JSON = 'data/raw/hearing.json'
TRACT_SHP = 'data/tl_2024_53_tract/tl_2024_53_tract.shp'
OUT = 'results'
STAT_COLUMNS = [
    'lang_rate', 'hear_rate', 'lang_num', 'lang_den',
    'hear_num', 'hear_den', 'lang_moe', 'hear_moe',
]


def load_and_join() -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """
    Load both ACS JSON tables and the King County tract shapefile, compute
    rates and MOEs, and join them one-to-one on GEOID.

    Returns (lang_hear, with_geometry, report) where report contains row,
    column, unmatched-key, and missing-value counts.
    """
    lang_raw = dp.load_acs_json(LANG_JSON)
    hear_raw = dp.load_acs_json(HEAR_JSON)
    tracts = dp.load_tracts(TRACT_SHP)
    for df, name in [(lang_raw, 'language'), (hear_raw, 'hearing')]:
        dp.check_unique(df, 'GEOID', name)
        assert (df['GEO_ID'].str[-11:] == df['GEOID']).all()
    lang = dp.build_rates(lang_raw, dp.LANG_NUM, dp.LANG_DEN, 'lang')
    hear = dp.build_rates(hear_raw, dp.HEAR_NUM, dp.HEAR_DEN, 'hear')
    lang_hear, miss_lh = dp.join_one_to_one(lang, hear.drop(columns='NAME'))
    full, miss_geo = dp.join_one_to_one(tracts, lang_hear)
    lang_cols = [f + s for f in dp.LANG_DEN + dp.LANG_NUM for s in ('E', 'M')]
    hear_cols = [f + s for f in dp.HEAR_DEN + dp.HEAR_NUM for s in ('E', 'M')]
    lang_missing = {col: int(lang_raw[col].isna().sum()) for col in lang_cols}
    hear_missing = {col: int(hear_raw[col].isna().sum()) for col in hear_cols}
    report: dict[str, Any] = {
        'tracts_language': len(lang_raw),
        'tracts_hearing': len(hear_raw),
        'tracts_boundaries': len(tracts),
        'unmatched_language_hearing': len(miss_lh),
        'unmatched_geometry': len(miss_geo),
        'language_columns_raw': lang_raw.shape[1],
        'hearing_columns_raw': hear_raw.shape[1],
        'language_selected_columns_count': len(lang_cols),
        'hearing_selected_columns_count': len(hear_cols),
        'language_selected_missing_total': sum(lang_missing.values()),
        'hearing_selected_missing_total': sum(hear_missing.values()),
        'language_selected_missing_by_column': lang_missing,
        'hearing_selected_missing_by_column': hear_missing,
        'zero_or_missing_lang_den': int(
            (lang_hear['lang_den'].fillna(0) <= 0).sum()),
        'zero_or_missing_hear_den': int(
            (lang_hear['hear_den'].fillna(0) <= 0).sum()),
        'null_lang_rate': int(lang_hear['lang_rate'].isna().sum()),
        'null_hear_rate': int(lang_hear['hear_rate'].isna().sum()),
    }
    return lang_hear, full, report


def describe_joins(
    lang_hear: pd.DataFrame, full: gpd.GeoDataFrame, both: pd.DataFrame
) -> dict[str, Any]:
    """
    Compute dataset shapes, seven-number summaries, and Pearson/Spearman
    correlations across the joined tables (lang_hear, full, both).
    """
    return {
        'language_hearing_shape': list(lang_hear.shape),
        'with_geometry_shape': list(full.shape),
        'both_valid_rows': len(both),
        'stats_language_hearing': an.summary_stats(
            lang_hear, STAT_COLUMNS).round(2).to_dict('index'),
        'stats_both_valid': an.summary_stats(
            both, STAT_COLUMNS).round(2).to_dict('index'),
        'correlation_pearson': round(
            both['lang_rate'].corr(both['hear_rate']), 3),
        'correlation_spearman': round(
            both['lang_rate'].rank().corr(both['hear_rate'].rank()), 3),
    }


def language_results(lang_hear: pd.DataFrame) -> dict[str, Any]:
    """
    Rank tracts in lang_hear by language-barrier percentage and count, save
    the top-ten CSV tables, and return a summary dict of ranking metrics.
    """
    ranked = an.rank_language(lang_hear)
    cols = ['GEOID', 'NAME', 'lang_num', 'lang_den', 'lang_rate',
            'lang_moe', 'rank_pct', 'rank_count']
    top = ranked.head(10)[cols]
    top_by_count = ranked.sort_values('rank_count').head(10)[cols]
    example = top.sort_values('rank_count').iloc[-1]
    top.to_csv(f'{OUT}/top10_language_pct.csv', index=False)
    top_by_count.to_csv(f'{OUT}/top10_language_count.csv', index=False)
    overlap = len(set(top['GEOID']) & set(top_by_count['GEOID']))
    return {
        'n_ranked': len(ranked),
        'top10_pct': top.round(2).to_dict('records'),
        'top10_count': top_by_count.round(2).to_dict('records'),
        'top10_overlap_pct_and_count': overlap,
        'rank_example': {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in example.to_dict().items()},
        'top10_min_den': float(top['lang_den'].min()),
    }


def overlap_results(
    both: pd.DataFrame, full: gpd.GeoDataFrame
) -> tuple[dict[str, Any], dict[str, gpd.GeoDataFrame]]:
    """
    Classify valid tracts in both at the top-quartile and top-tercile
    cutoffs, save overlap CSVs, and attach full tract geometries for maps.

    Returns (results_summary, labeled_geodataframes).
    """
    results: dict[str, Any] = {}
    labeled: dict[str, gpd.GeoDataFrame] = {}
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
            'high_both_share': round(
                counts['High on both'] / len(both), 4),
        }
        mapped = geo.merge(table, on='GEOID', how='left')
        mapped['group'] = mapped['group'].fillna(plots.NO_DATA_GROUP)
        labeled[name] = mapped.set_geometry('geometry')
        table.to_csv(f'{OUT}/overlap_{name.lower().replace(" ", "_")}.csv',
                     index=False)
    quartile, tercile = labeled['Top quartile'], labeled['Top tercile']
    entering = an.new_at_wider_cutoff(quartile, tercile)
    results['entering_at_tercile'] = entering
    results['high_both_quartile_ids'] = sorted(
        quartile.loc[quartile['group'] == 'High on both', 'GEOID'])
    return results, labeled


def save_figure(
    fig: go.Figure, name: str, width: int = 1000, height: int = 700
) -> None:
    """
    Save fig as interactive HTML and, when kaleido/Chrome is available, as a
    high-resolution PNG using the given width and height.
    """
    fig.write_html(f'{OUT}/figures/{name}.html', include_plotlyjs='cdn')
    try:
        fig.write_image(f'{OUT}/figures/{name}.png', width=width,
                        height=height, scale=2)
    except Exception as err:  # PNG export needs kaleido and Chrome
        print(f'PNG export skipped for {name}: {err}')


def make_figures(
    full: gpd.GeoDataFrame, both: pd.DataFrame,
    labeled: dict[str, gpd.GeoDataFrame]
) -> None:
    """
    Generate and save all maps, scatter plots, and grouped bar charts from
    full, both, and the cutoff-labeled GeoDataFrames.
    """
    counts = {name: an.group_counts(labeled[name]) for name in labeled}
    save_figure(plots.language_rate_map(full), 'fig1_language_map')
    save_figure(plots.scatter_rates(both), 'fig2_scatter', 900, 650)
    save_figure(plots.overlap_bar(counts), 'fig3_overlap_bar', 900, 600)
    for key, name in [('Top quartile', 'fig4a_overlap_map_quartile'),
                      ('Top tercile', 'fig4b_overlap_map_tercile')]:
        title = f'Overlap at the {key.lower()} cutoff'
        save_figure(plots.overlap_map(labeled[key], title), name)
    selectable = plots.overlap_map_selectable(labeled)
    selectable.write_html(f'{OUT}/figures/fig4_overlap_map_selectable.html',
                          include_plotlyjs='cdn')


def main():
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    lang_hear, full, report = load_and_join()
    both = lang_hear.dropna(subset=['lang_rate', 'hear_rate'])
    summary: dict[str, Any] = {'report': report}
    summary.update(describe_joins(lang_hear, full, both))
    summary['language'] = language_results(lang_hear)
    overlap, labeled = overlap_results(both, full)
    summary['overlap'] = overlap
    make_figures(full, both, labeled)
    with open(f'{OUT}/summary.json', 'w') as out_file:
        json.dump(summary, out_file, indent=2, default=float)
    print(json.dumps(summary['report'], indent=2))
    print('Done. Results are in the results/ folder.')


if __name__ == '__main__':
    main()
