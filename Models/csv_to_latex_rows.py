import pandas as pd
import sys

# In case this helps anyone...
# Usage:
#   python csv_to_latex_rows.py path/to/file.csv
#
#prints LaTeX rows *WITHOUT* rounding.

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python csv_to_latex_rows.py path/to/file.csv")

    path = sys.argv[1]
    df = pd.read_csv(path)

    for c in df.columns:

        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else repr(float(x)))
        else:
            df[c] = df[c].astype(str)


    def esc(s: str) -> str:
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("&", "\\&")
                 .replace("%", "\\%")
                 .replace("_", "\\_")
                 .replace("#", "\\#")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("^", "\\textasciicircum{}")
                 .replace("~", "\\textasciitilde{}"))

    for c in df.columns:
        if not all(df[c].str.match(r"^-?\d+(\.\d+)?([eE]-?\d+)?$") | (df[c] == "")):
            df[c] = df[c].map(esc)

    for _, row in df.iterrows():
        print(" & ".join(row.values) + r" \\")  

if __name__ == "__main__":
    main()
