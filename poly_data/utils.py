import json
from poly_utils.google_utils import get_spreadsheet
import pandas as pd 
import os
from decimal import Decimal

def pretty_print(txt, dic):
    print("\n", txt, json.dumps(dic, indent=4))


def _normalize_token_id(val):
    """
    Ensure token/condition ids are stringified without scientific notation.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, str):
        return val.strip()
    try:
        # Use Decimal to avoid float scientific notation
        return format(Decimal(str(val)), 'f').split('.')[0]
    except Exception:
        return str(val)

def get_sheet_df(read_only=None):
    """
    Get sheet data with optional read-only mode
    
    Args:
        read_only (bool): If None, auto-detects based on credentials availability
    """
    all = 'All Markets'
    sel = 'Selected Markets'

    # Auto-detect read-only mode if not specified
    if read_only is None:
        creds_file = 'credentials.json' if os.path.exists('credentials.json') else '../credentials.json'
        read_only = not os.path.exists(creds_file)
        if read_only:
            print("No credentials found, using read-only mode")

    try:
        spreadsheet = get_spreadsheet(read_only=read_only)
    except FileNotFoundError:
        print("No credentials found, falling back to read-only mode")
        spreadsheet = get_spreadsheet(read_only=True)

    wk = spreadsheet.worksheet(sel)
    df = pd.DataFrame(wk.get_all_records())
    df = df[df['question'] != ""].reset_index(drop=True)

    wk2 = spreadsheet.worksheet(all)
    df2 = pd.DataFrame(wk2.get_all_records())
    df2 = df2[df2['question'] != ""].reset_index(drop=True)

    result = df.merge(df2, on='question', how='inner')

    # Handle duplicate columns from merge (e.g., token1_x / token1_y)
    suffixes = ['_x', '_y']
    dup_bases = set()
    for col in result.columns:
        for suf in suffixes:
            if col.endswith(suf):
                dup_bases.add(col[: -len(suf)])

    for base in dup_bases:
        left = base + '_x'
        right = base + '_y'
        if left in result.columns or right in result.columns:
            result[base] = result[left] if left in result.columns else result[right]
            if left in result.columns and right in result.columns:
                result[base] = result[left].where(result[left].notna() & (result[left] != ""), result[right])
            result.drop(columns=[c for c in [left, right] if c in result.columns], inplace=True, errors='ignore')

    # Normalize IDs to plain strings (avoid scientific notation from Sheets)
    for col in ["token1", "token2", "condition_id"]:
        if col in result.columns:
            result[col] = result[col].apply(_normalize_token_id)

    wk_p = spreadsheet.worksheet('Hyperparameters')
    records = wk_p.get_all_records()
    hyperparams, current_type = {}, None

    for r in records:
        # Update current_type only when we have a non-empty type value
        # Handle both string and NaN values from pandas
        type_value = r['type']
        if type_value and str(type_value).strip() and str(type_value) != 'nan':
            current_type = str(type_value).strip()
        
        # Skip rows where we don't have a current_type set
        if current_type:
            # Convert numeric values to appropriate types
            value = r['value']
            try:
                # Try to convert to float if it's numeric
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                elif isinstance(value, (int, float)):
                    value = float(value)
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails
            
            hyperparams.setdefault(current_type, {})[r['param']] = value

    return result, hyperparams
