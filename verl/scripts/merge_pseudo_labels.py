import pandas as pd
import json
import argparse
import os

def normalize_super_canonical(text):
    if not isinstance(text, str):
        return ""
    import re
    # 1. Lowercase
    text = text.lower()
    # 2. Remove all backslashes and LaTeX commands (recursive-ish)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\\', '', text)
    # 3. Remove all non-alphanumeric characters
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def normalize_canonical(text):
    if not isinstance(text, str):
        return ""
    import re
    text = text.strip().lower()
    text = re.sub(r'\s+', '', text)
    return text

def extract_content(instruction):
    # Verl's parquet often stores instruction as a list of dicts (ChatML)
    if isinstance(instruction, (list, pd.Series, pd.Index)) or str(type(instruction)).find('array') != -1:
        for msg in instruction:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
    elif isinstance(instruction, str):
        return instruction
    return ""

def main():
    parser = argparse.ArgumentParser(description="Robustly merge pseudo-labels into a verl training parquet.")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to existing training parquet")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to pseudo-label JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged parquet")
    parser.add_argument("--by_order", action="store_true", help="Match by row order (only if counts match)")
    args = parser.parse_args()

    if not os.path.exists(args.parquet_path):
        print(f"Error: Parquet file {args.parquet_path} not found.")
        return
    if not os.path.exists(args.jsonl_path):
        print(f"Error: JSONL file {args.jsonl_path} not found.")
        return

    print(f"Loading Parquet: {args.parquet_path}")
    df_parquet = pd.read_parquet(args.parquet_path)
    
    # Robust column identification
    target_col = None
    for col in ['instruction', 'prompt']:
        if col in df_parquet.columns:
            target_col = col
            break
            
    if not target_col:
        print(f"Error: Could not find 'instruction' or 'prompt' column in Parquet.")
        print(f"Available columns: {df_parquet.columns.tolist()}")
        return

    print(f"Loading JSONL: {args.jsonl_path}")
    pseudo_data = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                pseudo_data.append(item)
            except Exception as e:
                print(f"Skip invalid line: {e}")
    df_jsonl = pd.DataFrame(pseudo_data)

    if args.by_order:
        print("Matching by row order...")
        if len(df_parquet) != len(df_jsonl):
            print(f"Error: Row counts do not match! Parquet: {len(df_parquet)}, JSONL: {len(df_jsonl)}")
            return
        
        # Sanity check: how many rows actually match by text when aligned by order?
        print("Performing sanity check on row order...")
        df_parquet['match_key_raw'] = df_parquet[target_col].apply(extract_content)
        df_parquet['match_key'] = df_parquet['match_key_raw'].apply(normalize_super_canonical)
        
        jsonl_keys_norm = [normalize_super_canonical(item.get("instruction", item.get("problem", ""))) for item in pseudo_data]
        
        order_matches = 0
        mismatch_indices = []
        for idx in range(len(df_parquet)):
            if df_parquet['match_key'].iloc[idx] == jsonl_keys_norm[idx]:
                order_matches += 1
            else:
                mismatch_indices.append(idx)
        
        print(f"Sanity Check: {order_matches}/{len(df_parquet)} rows match by text at the same index.")
        print(f"Confidence Score: {100 * order_matches / len(df_parquet):.2f}%")
        
        if order_matches < len(df_parquet):
            print(f"!!! CRITICAL WARNING: ORDER MISMATCH DETECTED !!!")
            print(f"Confidence Score {100 * order_matches / len(df_parquet):.2f}% is too low.")
            print(f"DO NOT USE THIS MERGED FILE FOR TRAINING. THE LABELS ARE SCRAMBLED.")
            if mismatch_indices:
                first_mismatch = mismatch_indices[0]
                print(f"First mismatch at index {first_mismatch}:")
                print(f"  Parquet: {df_parquet['match_key_raw'].iloc[first_mismatch][:100]}...")
                print(f"  JSONL  : {pseudo_data[first_mismatch].get('instruction', '')[:100]}...")

        df_merged = df_parquet.copy()
        df_merged['voted_answer'] = df_jsonl['voted_answer'].values
        matched_count = len(df_merged)
    else:
        print(f"Extracting match keys using SUPER CANONICAL mode (column: '{target_col}')...")
        df_parquet['match_key_raw'] = df_parquet[target_col].apply(extract_content)
        df_parquet['match_key'] = df_parquet['match_key_raw'].apply(normalize_super_canonical)
        
        # Load JSONL keys
        jsonl_keys_raw = []
        jsonl_keys_norm = []
        for item in pseudo_data:
            raw = item.get("instruction", item.get("problem", ""))
            jsonl_keys_raw.append(raw)
            jsonl_keys_norm.append(normalize_super_canonical(raw))
        
        df_jsonl['match_key'] = jsonl_keys_norm
        df_jsonl['match_key_raw_jsonl'] = jsonl_keys_raw

        print(f"Merging on match_key... (Parquet rows: {len(df_parquet)}, JSONL rows: {len(df_jsonl)})")
        df_merged = pd.merge(df_parquet, df_jsonl[['match_key', 'voted_answer']], on='match_key', how='left')
        matched_count = df_merged['voted_answer'].notna().sum()

    # Statistics
    print(f"Merge Summary:")
    print(f" - Total Parquet Rows  : {len(df_parquet)}")
    print(f" - Matched Rows         : {matched_count}")
    print(f" - Match Rate           : {100 * matched_count / len(df_parquet):.2f}%")

    if not args.by_order and matched_count < len(df_parquet):
        print("\nDEBUG: First few unmatched samples:")
        unmatched = df_merged[df_merged['voted_answer'].isna()].head(2)
        for _, row in unmatched.iterrows():
            print(f"--- Parquet Key (Raw): {row['match_key_raw'][:100]}...")
            # Try to find a 'similar' key in JSONL for comparison
            print(f"--- Normalized Key   : {row['match_key'][:100]}...")
        
        print("\nIf you are sure the order is identical, use --by_order to skip text matching.")

    # Clean up and save
    cols_to_drop = ['match_key', 'match_key_raw'] if 'match_key' in df_merged.columns else []
    df_merged = df_merged.drop(columns=cols_to_drop)
    df_merged.to_parquet(args.output_path)
    print(f"Successfully saved merged parquet to: {args.output_path}")

if __name__ == "__main__":
    main()
