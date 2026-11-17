"""
Name: Krishna Pratheek Madamala
Section: AC
CSE 163, Autumn 2026
Final Project, Part 2: Exploratory Data Analysis

Creates interactive Plotly maps, scatter plots, and grouped bar charts for
the King County older-adult language and hearing barrier analysis.
"""

import json
import math
from typing import Any

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import GROUPS

# High-contrast colors for the three highlighted groups; neutral grays for
# "Neither" and tracts with no 65+ population data.
NO_DATA_GROUP = 'No 65+ data'
MAP_GROUPS = GROUPS + [NO_DATA_GROUP]
GROUP_COLORS = {
    'High on both': '#2a78d6',
    'Language only': '#eb6834',
    'Hearing only': '#1baf7a',
    'Neither': '#d9d9d6',
    NO_DATA_GROUP: '#8c8c8c',
}
INK = '#0b0b0b'
LAYOUT = dict(
    font=dict(family='Arial, sans-serif', size=14, color=INK),
    paper_bgcolor='white',
    margin=dict(l=20, r=20, t=70, b=20),
)


def _geojson(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Convert gdf indexed by GEOID into a GeoJSON FeatureCollection dict.

    Returns an empty FeatureCollection if gdf has no rows.
    """
    if gdf.empty:
        return {'type': 'FeatureCollection', 'features': []}
    return json.loads(gdf.set_index('GEOID').to_json())


def _style_map(
    fig: go.Figure, gdf: gpd.GeoDataFrame, width: int = 1000,
    height: int = 700
) -> go.Figure:
    """
    Center and zoom a tile-free map figure on the tract bounds in gdf using
    the target width and height in pixels. Returns fig.
    """
    west, south, east, north = gdf.total_bounds
    span = max((east - west) * height / width, north - south) * 1.1
    zoom = math.log2(360 * height / (512 * span)) - 0.5
    fig.update_layout(
        map=dict(style='white-bg', zoom=zoom,
                 center=dict(lon=(west + east) / 2,
                             lat=(south + north) / 2)),
        **LAYOUT)
    return fig


def language_rate_map(gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    Build a choropleth map of tract-level limited-English rates from gdf.

    Tracts with missing lang_rate are drawn in a separate neutral gray trace
    labeled 'No 65+ data'. Returns the Plotly Figure.
    """
    valid = gdf.dropna(subset=['lang_rate'])
    missing = gdf[gdf['lang_rate'].isna()]
    text = [
        f'{geoid}: {rate:.1f}% ({num:.0f} of {den:.0f})'
        for geoid, rate, num, den in zip(
            valid['GEOID'], valid['lang_rate'], valid['lang_num'],
            valid['lang_den'])
    ]
    traces = [go.Choroplethmap(
        geojson=_geojson(valid), locations=valid['GEOID'],
        z=valid['lang_rate'], colorscale='Blues', text=text,
        hoverinfo='text', marker_line_width=0.3,
        marker_line_color='white',
        colorbar=dict(title='% of residents<br>age 65+'))]
    if not missing.empty:
        no_data_color = GROUP_COLORS[NO_DATA_GROUP]
        traces.append(go.Choroplethmap(
            geojson=_geojson(missing), locations=missing['GEOID'],
            z=[1] * len(missing),
            name=f'{NO_DATA_GROUP} ({len(missing)})',
            colorscale=[[0, no_data_color], [1, no_data_color]],
            showscale=False, showlegend=True,
            marker_line_width=0.3, marker_line_color='white',
            text=[f'{geoid}: {NO_DATA_GROUP}' for geoid in missing['GEOID']],
            hoverinfo='text'))
    fig = go.Figure(traces)
    fig.update_layout(
        title='Older adults with limited English by tract, King County',
        legend=dict(x=0.01, y=0.02))
    return _style_map(fig, gdf)


def scatter_rates(df: pd.DataFrame) -> go.Figure:
    """
    Build a scatter plot of hear_rate against lang_rate with one point per
    tract in df. Returns the Plotly Figure.
    """
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


def overlap_bar(counts: dict[str, pd.Series]) -> go.Figure:
    """
    Build a grouped bar chart comparing tract counts across the four overlap
    groups for each cutoff in counts. Returns the Plotly Figure.
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


def _group_traces(
    gdf: gpd.GeoDataFrame, visible: bool
) -> list[go.Choroplethmap]:
    """
    Build one Choroplethmap trace per group in MAP_GROUPS from gdf with the
    given initial visibility. Each trace serializes only its own geometries.
    """
    traces = []
    for group in MAP_GROUPS:
        part = gdf[gdf['group'] == group]
        color = GROUP_COLORS[group]
        traces.append(go.Choroplethmap(
            geojson=_geojson(part), locations=part['GEOID'],
            z=[1] * len(part), name=f'{group} ({len(part)})',
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=True, visible=visible,
            marker_line_width=0.2, marker_line_color='white',
            text=part['GEOID'], hoverinfo='text+name'))
    return traces


def overlap_map(gdf: gpd.GeoDataFrame, title: str) -> go.Figure:
    """
    Build a choropleth map of the barrier overlap groups in gdf (including
    missing-data tracts) with the given title. Returns the Plotly Figure.
    """
    fig = go.Figure(_group_traces(gdf, True))
    fig.update_layout(title=title, legend_title_text='Group')
    return _style_map(fig, gdf)


def overlap_map_selectable(
    labeled_by_cutoff: dict[str, gpd.GeoDataFrame]
) -> go.Figure:
    """
    Build an interactive choropleth map with a dropdown menu switching
    between the labeled cutoff GeoDataFrames in labeled_by_cutoff.
    """
    names = list(labeled_by_cutoff)
    first = labeled_by_cutoff[names[0]]
    fig = go.Figure()
    for i, name in enumerate(names):
        for trace in _group_traces(labeled_by_cutoff[name], i == 0):
            fig.add_trace(trace)
    per_cutoff = len(MAP_GROUPS)
    total_traces = per_cutoff * len(names)
    buttons = []
    for i, name in enumerate(names):
        visible = [j // per_cutoff == i for j in range(total_traces)]
        buttons.append(dict(
            label=name, method='update',
            args=[{'visible': visible}, {'title': f'Overlap: {name}'}]))
    fig.update_layout(
        title=f'Overlap: {names[0]}', legend_title_text='Group',
        updatemenus=[dict(buttons=buttons, x=0.0, y=1.08)])
    return _style_map(fig, first)
