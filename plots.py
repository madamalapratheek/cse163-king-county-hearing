"""Plotly figures for the language and hearing analysis."""

import json
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import GROUPS

# Categorical colors follow the first three slots of a validated palette;
# "Neither" is a neutral gray so the three highlighted groups stand out.
GROUP_COLORS = {
    'High on both': '#2a78d6',
    'Language only': '#eb6834',
    'Hearing only': '#1baf7a',
    'Neither': '#d9d9d6',
}
INK = '#0b0b0b'
LAYOUT = dict(
    font=dict(family='Arial, sans-serif', size=14, color=INK),
    paper_bgcolor='white',
    margin=dict(l=20, r=20, t=70, b=20),
)


def _geojson(gdf):
    """Return GeoJSON for a table indexed by GEOID."""
    return json.loads(gdf.set_index('GEOID').to_json())


def _style_map(fig, gdf, width=1000, height=700):
    """Center and zoom the tile-free map on the tracts in gdf."""
    west, south, east, north = gdf.total_bounds
    span = max((east - west) * height / width, north - south) * 1.1
    zoom = math.log2(360 * height / (512 * span / 1.0)) - 0.5
    fig.update_layout(
        map=dict(style='white-bg', zoom=zoom,
                 center=dict(lon=(west + east) / 2,
                             lat=(south + north) / 2)),
        **LAYOUT)
    return fig


def language_rate_map(gdf):
    """Choropleth of the language-barrier rate by tract."""
    valid = gdf.dropna(subset=['lang_rate'])
    text = [f'{g}: {r:.1f}% ({n:.0f} of {d:.0f})' for g, r, n, d in zip(
        valid['GEOID'], valid['lang_rate'], valid['lang_num'],
        valid['lang_den'])]
    fig = go.Figure(go.Choroplethmap(
        geojson=_geojson(valid), locations=valid['GEOID'],
        z=valid['lang_rate'], colorscale='Blues', text=text,
        hoverinfo='text', marker_line_width=0.3,
        marker_line_color='white',
        colorbar=dict(title='% of residents<br>age 65+')))
    fig.update_layout(
        title='Older adults with limited English by tract, King County')
    return _style_map(fig, gdf)


def scatter_rates(df):
    """Scatter of hearing rate against language rate, one dot per tract."""
    fig = px.scatter(
        df, x='lang_rate', y='hear_rate',
        labels={'lang_rate': 'Limited English, age 65+ (% of tract)',
                'hear_rate': 'Hearing difficulty, age 65+ (% of tract)'},
        hover_data={'GEOID': True, 'lang_rate': ':.1f',
                    'hear_rate': ':.1f'})
    fig.update_traces(marker=dict(
        size=8, color='#2a78d6', opacity=0.7,
        line=dict(width=1, color='white')))
    fig.update_layout(
        title='Language and hearing difficulty rates by tract',
        plot_bgcolor='white', **LAYOUT)
    fig.update_xaxes(gridcolor='#ececea', zeroline=False)
    fig.update_yaxes(gridcolor='#ececea', zeroline=False)
    return fig


def overlap_bar(counts):
    """Grouped bars: tracts in each group at the two cutoffs.

    counts is a dict mapping cutoff name to a Series of group counts.
    """
    rows = [
        {'Cutoff': cutoff, 'Group': group, 'Tracts': int(series[group])}
        for cutoff, series in counts.items() for group in GROUPS]
    fig = px.bar(
        pd.DataFrame(rows), x='Group', y='Tracts', color='Cutoff',
        barmode='group', text='Tracts',
        color_discrete_sequence=['#2a78d6', '#eb6834'])
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title='Tracts by group at the top-quartile and top-tercile cutoffs',
        plot_bgcolor='white', **LAYOUT)
    fig.update_yaxes(gridcolor='#ececea')
    return fig


def _group_traces(gdf, geojson, visible):
    """One choropleth trace per group, for a labeled tract table."""
    traces = []
    for group in GROUPS:
        part = gdf[gdf['group'] == group]
        color = GROUP_COLORS[group]
        traces.append(go.Choroplethmap(
            geojson=geojson, locations=part['GEOID'],
            z=[1] * len(part), name=f'{group} ({len(part)})',
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=True, visible=visible,
            marker_line_width=0.2, marker_line_color='white',
            text=part['GEOID'], hoverinfo='text+name'))
    return traces


def overlap_map(gdf, title):
    """Map of the four groups for one cutoff."""
    fig = go.Figure(_group_traces(gdf, _geojson(gdf), True))
    fig.update_layout(title=title, legend_title_text='Group')
    return _style_map(fig, gdf)


def overlap_map_selectable(labeled_by_cutoff):
    """Map with a dropdown to switch between the cutoffs."""
    names = list(labeled_by_cutoff)
    first = labeled_by_cutoff[names[0]]
    geojson = _geojson(first)
    fig = go.Figure()
    for i, name in enumerate(names):
        for trace in _group_traces(
                labeled_by_cutoff[name], geojson, i == 0):
            fig.add_trace(trace)
    per = len(GROUPS)
    buttons = []
    for i, name in enumerate(names):
        visible = [j // per == i for j in range(per * len(names))]
        buttons.append(dict(
            label=name, method='update',
            args=[{'visible': visible}, {'title': f'Overlap: {name}'}]))
    fig.update_layout(
        title=f'Overlap: {names[0]}', legend_title_text='Group',
        updatemenus=[dict(buttons=buttons, x=0.0, y=1.08)])
    return _style_map(fig, first)
