"""Load the ACS tables and tract boundaries and compute tract-level rates.

Rates follow the project proposal:
  language rate = residents 65+ who speak English less than "very well"
                  divided by all residents 65+ (table B16004)
  hearing rate  = residents 65+ with a hearing difficulty divided by
                  civilian noninstitutionalized residents 65+ (B18102)
"""

import json

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


def load_acs_json(path):
    """Read a Census API JSON file into a numeric DataFrame.

    The first row of the file holds the column names. Geographic codes are
    kept as strings so leading zeros are preserved. Estimates (E) and
    margins of error (M) become numbers, and Census missing-value codes
    become NaN. The response repeats the NAME column, so duplicates are
    dropped. Returns the table with a new 11-digit GEOID column.
    """
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    df = df.loc[:, ~df.columns.duplicated()]
    value_cols = df.columns[df.columns.str.fullmatch(r'B\d{5}_\d{3}[EM]')]
    numbers = df[value_cols].apply(pd.to_numeric, errors='coerce')
    cleaned = numbers.mask(numbers.isin(MISSING_CODES))
    keep = df.drop(columns=value_cols)
    geoid = keep['state'] + keep['county'] + keep['tract']
    return pd.concat([keep, cleaned], axis=1).assign(GEOID=geoid)


def check_unique(df, key, name):
    """Raise ValueError if the key column has duplicate values."""
    if not df[key].is_unique:
        raise ValueError(f'Duplicate {key} values in {name}')


def sum_fields(df, fields, suffix):
    """Sum estimate (E) or margin (M) columns; NaN if any piece is NaN."""
    cols = [field + suffix for field in fields]
    return df[cols].sum(axis=1, min_count=len(cols))


def sum_moe(df, fields):
    """Margin of error of a sum: square root of the summed squares."""
    cols = [field + 'M' for field in fields]
    squares = (df[cols] ** 2).sum(axis=1, min_count=len(cols))
    return squares ** 0.5


def compute_rate(num, den):
    """Return num / den * 100, or NaN where den is zero or missing."""
    return num / den.where(den > 0) * 100


def proportion_moe(num, den, num_moe, den_moe):
    """Margin of error (percentage points) for a rate, using the
    Census Bureau's proportion formula, or its ratio formula where the
    proportion formula would take a square root of a negative number."""
    safe_den = den.where(den > 0)
    p = num / safe_den
    proportion = num_moe ** 2 - p ** 2 * den_moe ** 2
    ratio = num_moe ** 2 + p ** 2 * den_moe ** 2
    inside = proportion.where(proportion >= 0, ratio)
    return inside ** 0.5 / safe_den * 100


def build_rates(df, num_fields, den_fields, prefix):
    """Compute numerator, denominator, rate and rate MOE for a table.

    Output columns are GEOID, NAME and <prefix>_num, <prefix>_den,
    <prefix>_rate and <prefix>_moe.
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


def load_tracts(path):
    """Read the Washington tract shapefile, keep King County, use WGS84."""
    tracts = gpd.read_file(path)
    tracts = tracts[tracts['COUNTYFP'] == KING_COUNTY_FIPS]
    check_unique(tracts, 'GEOID', 'tract boundaries')
    return tracts[['GEOID', 'geometry']].to_crs(epsg=4326)


def join_one_to_one(left, right, name):
    """Outer-merge two tables on GEOID and report unmatched rows.

    Returns (merged, unmatched) where unmatched lists the GEOIDs that
    were in only one of the inputs, with a column saying which side.
    """
    merged = left.merge(
        right, on='GEOID', how='outer', validate='one_to_one',
        indicator=True)
    unmatched = merged.loc[merged['_merge'] != 'both', ['GEOID', '_merge']]
    print(f'{name}: {len(unmatched)} unmatched GEOIDs')
    return merged.drop(columns='_merge'), unmatched
