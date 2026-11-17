"""
Pratheek Madamala
CSE 163 <SECTION>

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


def load_acs_json(path: str | Path) -> pd.DataFrame:
    """
    Read a Census API JSON file at path into a cleaned DataFrame.

    Preserves geographic codes as strings, converts estimate (E) and margin
    of error (M) columns to numbers, replaces Census missing-value codes with
    NaN, drops duplicate columns, and adds an 11-digit GEOID column.
    """
    with open(path) as acs_file:
        rows = json.load(acs_file)
    df = pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    df = df.loc[:, ~df.columns.duplicated()]
    value_cols = df.columns[df.columns.str.fullmatch(r'B\d{5}_\d{3}[EM]')]
    numbers = df[value_cols].apply(pd.to_numeric, errors='coerce')
    cleaned = numbers.mask(numbers.isin(MISSING_CODES))
    keep = df.drop(columns=value_cols)
    geoid = keep['state'] + keep['county'] + keep['tract']
    return pd.concat([keep, cleaned], axis=1).assign(GEOID=geoid)


def check_unique(df: pd.DataFrame, key: str, name: str) -> None:
    """
    Raise ValueError if column key in df contains duplicate values.

    Uses name in the error message to identify the table being checked.
    """
    if not df[key].is_unique:
        raise ValueError(f'Duplicate {key} values in {name}')


def sum_fields(
    df: pd.DataFrame, fields: list[str], suffix: str
) -> pd.Series:
    """
    Sum columns in df named <field><suffix> for each field in fields.

    Returns a Series of row sums, with NaN for any row where at least one
    component column is NaN.
    """
    cols = [field + suffix for field in fields]
    return df[cols].sum(axis=1, min_count=len(cols))


def sum_moe(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """
    Compute the aggregated margin of error across fields in df.

    Follows the Census Bureau zero-estimate rule: squares and sums MOEs for
    nonzero estimates and includes only the single largest MOE among any
    zero-estimate components. Returns NaN for rows with any missing estimate
    or MOE.
    """
    est = df[[field + 'E' for field in fields]].set_axis(fields, axis=1)
    moe = df[[field + 'M' for field in fields]].set_axis(fields, axis=1)
    has_missing = est.isna().any(axis=1) | moe.isna().any(axis=1)
    nonzero_sq = (moe.where(est != 0, 0.0) ** 2).sum(axis=1)
    max_zero = moe.where(est == 0, 0.0).max(axis=1)
    result = (nonzero_sq + max_zero ** 2) ** 0.5
    return result.where(~has_missing)


def compute_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """
    Return num / den * 100 as a percentage Series.

    Returns NaN wherever den is zero or missing.
    """
    return num / den.where(den > 0) * 100


def proportion_moe(
    num: pd.Series, den: pd.Series, num_moe: pd.Series, den_moe: pd.Series
) -> pd.Series:
    """
    Compute the rate margin of error in percentage points from numerator and
    denominator estimates (num, den) and their MOEs (num_moe, den_moe).

    Uses the Census Bureau proportion formula and falls back to the ratio
    formula whenever the proportion radical would be negative. Returns NaN
    where den is zero or missing.
    """
    safe_den = den.where(den > 0)
    prop = num / safe_den
    proportion = num_moe ** 2 - prop ** 2 * den_moe ** 2
    ratio = num_moe ** 2 + prop ** 2 * den_moe ** 2
    inside = proportion.where(proportion >= 0, ratio)
    return inside ** 0.5 / safe_den * 100


def build_rates(
    df: pd.DataFrame, num_fields: list[str], den_fields: list[str],
    prefix: str
) -> pd.DataFrame:
    """
    Compute numerator, denominator, rate, and rate MOE columns from df using
    num_fields and den_fields.

    Returns a DataFrame with GEOID, NAME, and <prefix>_num, <prefix>_den,
    <prefix>_rate, and <prefix>_moe.
    """
    num = sum_fields(df, num_fields, 'E')
    den = sum_fields(df, den_fields, 'E')
    out = pd.DataFrame({'GEOID': df['GEOID'], 'NAME': df['NAME']})
    out[prefix + '_num'] = num
    out[prefix + '_den'] = den
    out[prefix + '_rate'] = compute_rate(num, den)
    out[prefix + '_moe'] = proportion_moe(
        num, den, sum_moe(df, num_fields), sum_moe(df, den_fields))
    return out


def load_tracts(path: str | Path) -> gpd.GeoDataFrame:
    """
    Read the Washington tract shapefile at path, filter to King County,
    verify unique GEOIDs, and return GEOID and geometry in WGS84 (EPSG:4326).
    """
    tracts = gpd.read_file(path)
    tracts = tracts[tracts['COUNTYFP'] == KING_COUNTY_FIPS]
    check_unique(tracts, 'GEOID', 'tract boundaries')
    return tracts[['GEOID', 'geometry']].to_crs(epsg=4326)


def join_one_to_one(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outer-merge left and right one-to-one on GEOID.

    Returns (merged, unmatched) where unmatched contains GEOIDs present in
    only one input along with the merge indicator column.
    """
    merged = left.merge(
        right, on='GEOID', how='outer', validate='one_to_one',
        indicator=True)
    unmatched = merged.loc[merged['_merge'] != 'both', ['GEOID', '_merge']]
    return merged.drop(columns='_merge'), unmatched
