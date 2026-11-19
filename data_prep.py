"""
Name: Krishna Pratheek Madamala
Section: AC
CSE 163, Autumn 2026
Final Project, Part 2: Exploratory Data Analysis

Loads Census ACS tables and tract boundaries, cleans missing values, and
computes King County tract-level language and hearing barrier rates with
margins of error.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

KING_COUNTY_FIPS = '033'

# Census Bureau placeholder codes for unavailable estimates.
MISSING_CODES = {
    -666666666, -999999999, -888888888,
    -555555555, -333333333, -222222222,
}

# B16004: age by language spoken at home by ability to speak English,
# rows for residents 65+. "Less than very well" is the "well",
# "not well" and "not at all" rows within each of four language groups.
LANG_DEN = ['B16004_046']
LANG_NUM = [
    'B16004_050', 'B16004_051', 'B16004_052',
    'B16004_055', 'B16004_056', 'B16004_057',
    'B16004_060', 'B16004_061', 'B16004_062',
    'B16004_065', 'B16004_066', 'B16004_067',
]

# B18102: sex by age by hearing difficulty. Rows 015/018 are male 65-74
# and 75+, and 034/037 are female 65-74 and 75+. The row after each total
# (016, 019, 035, 038) counts people with a hearing difficulty.
HEAR_DEN = ['B18102_015', 'B18102_018', 'B18102_034', 'B18102_037']
HEAR_NUM = ['B18102_016', 'B18102_019', 'B18102_035', 'B18102_038']


def load_acs_json(dataset_filepath: str | Path) -> pd.DataFrame:
    """
    Read the Census API JSON file at dataset_filepath, clean missing values,
    and return it as a DataFrame with an 11-digit GEOID column.
    """
    with open(dataset_filepath) as f:
        rows = json.load(f)
    data = pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    data = data.loc[:, ~data.columns.duplicated()]
    value_cols = data.columns[data.columns.str.fullmatch(r'B\d{5}_\d{3}[EM]')]
    numbers = data[value_cols].apply(pd.to_numeric, errors='coerce')
    cleaned = numbers.mask(numbers.isin(MISSING_CODES))
    keep = data.drop(columns=value_cols)
    geoid = keep['state'] + keep['county'] + keep['tract']
    result = pd.concat([keep, cleaned], axis=1).assign(GEOID=geoid)

    return result


def check_unique(data: pd.DataFrame, key: str, name: str) -> None:
    """
    Raise a ValueError if the key column in data contains duplicate values.
    """
    if not data[key].is_unique:
        raise ValueError('Duplicate ' + key + ' values in ' + name)


def sum_fields(data: pd.DataFrame, fields: list[str],
               suffix: str) -> pd.Series:
    """
    Sum columns in data matching each field in fields with suffix. Return
    NaN if any component value is NaN.
    """
    cols = [field + suffix for field in fields]
    totals = data[cols].sum(axis=1, min_count=len(cols))

    return totals


def sum_moe(data: pd.DataFrame, fields: list[str]) -> pd.Series:
    """
    Calculate the aggregated margin of error across fields in data using
    the Census Bureau rule for zero estimates.
    """
    est = data[[field + 'E' for field in fields]].set_axis(fields, axis=1)
    moe = data[[field + 'M' for field in fields]].set_axis(fields, axis=1)
    has_missing = est.isna().any(axis=1) | moe.isna().any(axis=1)
    nonzero_sq = (moe.where(est != 0, 0.0) ** 2).sum(axis=1)
    max_zero = moe.where(est == 0, 0.0).max(axis=1)
    result = (nonzero_sq + max_zero ** 2) ** 0.5

    return result.where(~has_missing)


def compute_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """
    Return num divided by den as a percentage. Return NaN where den is
    zero or missing.
    """
    rate = num / den.where(den > 0) * 100

    return rate


def proportion_moe(num: pd.Series, den: pd.Series,
                   num_moe: pd.Series, den_moe: pd.Series) -> pd.Series:
    """
    Calculate the percentage point margin of error for a rate using the
    Census proportion formula with ratio fallback.
    """
    safe_den = den.where(den > 0)
    prop = num / safe_den
    proportion = num_moe ** 2 - prop ** 2 * den_moe ** 2
    ratio = num_moe ** 2 + prop ** 2 * den_moe ** 2
    inside = proportion.where(proportion >= 0, ratio)
    result = inside ** 0.5 / safe_den * 100

    return result


def build_rates(data: pd.DataFrame, num_fields: list[str],
                den_fields: list[str], prefix: str) -> pd.DataFrame:
    """
    Compute numerator, denominator, rate, and rate margin of error for
    the given table fields and prefix.
    """
    num = sum_fields(data, num_fields, 'E')
    den = sum_fields(data, den_fields, 'E')
    out = pd.DataFrame({'GEOID': data['GEOID'], 'NAME': data['NAME']})
    out[prefix + '_num'] = num
    out[prefix + '_den'] = den
    out[prefix + '_rate'] = compute_rate(num, den)
    out[prefix + '_moe'] = proportion_moe(
        num, den, sum_moe(data, num_fields), sum_moe(data, den_fields)
    )

    return out


def load_tracts(shapefile_filepath: str | Path) -> gpd.GeoDataFrame:
    """
    Read Washington tract boundaries, filter to King County, and return
    GEOID and geometry in WGS84 coordinates.
    """
    tracts = gpd.read_file(shapefile_filepath)
    tracts = tracts[tracts['COUNTYFP'] == KING_COUNTY_FIPS]
    check_unique(tracts, 'GEOID', 'tract boundaries')
    selected = tracts[['GEOID', 'geometry']].to_crs(epsg=4326)

    return selected


def join_one_to_one(left: pd.DataFrame,
                    right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge left and right one-to-one on GEOID and return the merged table
    and any unmatched rows.
    """
    merged = left.merge(
        right, on='GEOID', how='outer', validate='one_to_one',
        indicator=True
    )
    unmatched = merged.loc[merged['_merge'] != 'both', ['GEOID', '_merge']]
    result = merged.drop(columns='_merge')

    return result, unmatched
