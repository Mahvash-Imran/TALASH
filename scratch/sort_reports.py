import pandas as pd
from pathlib import Path

# Paths
output_dir = Path("data/analysis")
profiles_path = output_dir / "educational_profiles.csv"
degrees_path = output_dir / "edu_degrees.csv"
gaps_path = output_dir / "edu_gaps.csv"
excel_path = output_dir / "educational_profiles.xlsx"
report_path = output_dir / "edu_report.txt"

# 1. Load profiles
if not profiles_path.exists():
    print("Error: educational_profiles.csv does not exist.")
    exit(1)

df_profiles = pd.read_csv(profiles_path)

# 2. Sort profiles
strength_ranks = {"Strong": 1, "Moderate": 2, "Weak": 3}
df_profiles["rank"] = df_profiles["educational_strength"].map(lambda x: strength_ranks.get(x, 99))
df_profiles = df_profiles.sort_values(by=["rank", "candidate_id"]).drop(columns=["rank"])

# Save sorted CSV
df_profiles.to_csv(profiles_path, index=False)
print("Saved sorted educational_profiles.csv")

# 3. Save Excel
df_degrees = pd.read_csv(degrees_path) if degrees_path.exists() else None
df_gaps = pd.read_csv(gaps_path) if gaps_path.exists() else None

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_profiles.to_excel(writer, sheet_name="Profiles", index=False)
    if df_degrees is not None:
        df_degrees.to_excel(writer, sheet_name="Degrees", index=False)
    if df_gaps is not None:
        df_gaps.to_excel(writer, sheet_name="Gaps", index=False)
print("Saved sorted Excel workbook")

# 4. Generate edu_report.txt
lines = [
    "=" * 70,
    "  TALASH Module 2 - Educational Profile Report",
    "=" * 70,
    f"  Candidates analysed : {len(df_profiles)}",
    "",
]

strength_counts = df_profiles["educational_strength"].value_counts().to_dict()
lines.append("  Strength Distribution:")
for label in ["Strong", "Moderate", "Weak"]:
    if label in strength_counts:
        lines.append(f"    {label:<22}: {strength_counts[label]}")

lines += ["", "-" * 70, "  Per-Candidate Summaries", "-" * 70]
for _, row in df_profiles.iterrows():
    lines.append(f"\n  [{row['educational_strength']}] {row['candidate_id']}")
    lines.append(f"  Highest Degree : {row['highest_degree']}")
    lines.append(f"  Progression    : {'Consistent' if row['progression_consistent'] else 'Inconsistent'}")
    lines.append(f"  Perf. Trend    : {row['performance_trend']}")
    lines.append(f"  Gaps (total/significant/unexplained): "
                 f"{row['total_gaps']}/{row['significant_gaps']}/{row['unexplained_gaps']}")
    if pd.notna(row['summary']):
        lines.append(f"  Summary: {row['summary']}")

lines.append("\n" + "=" * 70)

report_path.write_text("\n".join(lines), encoding="utf-8")
print("Saved sorted edu_report.txt")
