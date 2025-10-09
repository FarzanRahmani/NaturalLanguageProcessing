import pandas as pd
import json
import argparse
import os
import sys

def normalize_iteration(val):
    try:
        if pd.isna(val):
            return None
        
        if isinstance(val, str):
            val = val.strip()
            if val == '':
                return None
            valf = float(val.replace(',', ''))
        else:
            valf = float(val)
        return int(valf)
    except Exception:
        return None

def convert_excel_to_json(xlsx_path: str, json_path: str, sheet_name: str | None = None):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"فایل پیدا نشد: {xlsx_path}")

    try:
        if sheet_name:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"خطا در خواندن فایل اکسل: {e}")

    expected = {
        'شماره': 'iteration',
        'متن': 'text',
        'سوال': 'question',
        'جواب': 'answer'
    }

    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"ستون‌های زیر پیدا نشدند در فایل اکسل: {missing}. نام دقیق ستون‌ها باید '{list(expected.keys())}' باشند.")

    df = df.rename(columns=expected)

    df['iteration'] = df['iteration'].apply(normalize_iteration)
    df = df.dropna(subset=['iteration'])
    df['iteration'] = df['iteration'].astype(int)

    df = df[['iteration', 'text', 'question', 'answer']]

    records = df.to_dict(orient='records')

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise RuntimeError(f"خطا در نوشتن فایل JSON: {e}")

    print(f"تبدیل انجام شد. فایل JSON در: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="تبدیل فایل .xlsx به .json با نگاشت مشخص شده.")
    parser.add_argument("input", help="مسیر فایل ورودی .xlsx")
    parser.add_argument("output", help="مسیر فایل خروجی .json")
    parser.add_argument("--sheet", "-s", help="نام شیت اکسل (اگر نخواهید از اولین شیت استفاده شود)", default=None)

    args = parser.parse_args()

    try:
        convert_excel_to_json(args.input, args.output, args.sheet)
    except Exception as e:
        print(f"خطا: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()