# Language and Hearing Barriers Among Older Adults in King County

CSE 163 Final Project, Part 2 (Exploratory Data Analysis), Strand B:
Sound Generations. This code compares two tract-level rates for King County
residents age 65 or older: the share who speak English less than "very
well" (ACS table B16004) and the share with a hearing difficulty (ACS table
B18102), and maps where both are high.

## Setup

1. Install Python 3.10 or newer, then install the libraries:

   ```
   pip install -r requirements.txt
   ```

   Static PNG export of the figures needs Chrome or Chromium installed
   (used by kaleido). Without it, `main.py` still saves interactive HTML
   figures and prints a message for each PNG it skips.

2. Download the data. The Census Data API requires a free key from
   https://api.census.gov/data/key_signup.html. Save each response as JSON:

   - `data/raw/lang.json`:
     `https://api.census.gov/data/2024/acs/acs5?get=NAME,group(B16004)&for=tract:*&in=state:53%20county:033&key=YOUR_KEY`
   - `data/raw/hearing.json`: the same link with `group(B18102)`.
   - Tract boundaries: choose Washington at
     https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=Census+Tracts
     and extract the ZIP into `data/tl_2024_53_tract/`.

   The code does not need the key, only the saved files. Variable
   definitions: https://api.census.gov/data/2024/acs/acs5/groups/B16004.html
   and https://api.census.gov/data/2024/acs/acs5/groups/B18102.html

## Run

```
python main.py
```

Outputs go to `results/`: `summary.json` (every number used in the report),
CSV tables, and `figures/` (HTML and PNG). The file
`results/figures/fig4_overlap_map_selectable.html` has a dropdown to switch
between the top-quartile and top-tercile cutoffs.

## Tests and style

```
pytest test_analysis.py
flake8
```

## Files

- `data_prep.py`: loading, cleaning, rates, margins of error, joins
- `analysis.py`: ranking, cutoffs, groups, summary statistics
- `plots.py`: Plotly figures
- `main.py`: runs the whole analysis
- `test_analysis.py`: tests with built-in test data
