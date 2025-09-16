import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Directories
ROOT = Path("corpus_metric")
IDF_DIR = ROOT / "idf"
DEP_DIR = ROOT / "parse_depth"
PMI_DIR = ROOT / "pmi"

# Order & display names
order = ["Papers", "Shakespeare", "Victorian", "NewPoems", "War&Peace",
         "RedChamber", "Dickens", "StKing", "HarryPotter"]

books = {
    "shakes": "Shakespeare",
    "victorian": "Victorian",
    "dikens": "Dickens",
    "warpeace": "War&Peace",
    "stone_Y": "RedChamber",
    "stephen": "StKing",
    "potter": "HarryPotter",
    "modern": "NewPoems",
    "articles": "Papers",
}

# TWiKer values you provided
twiker_vals = {
    "Papers": 0.0022295082453638315,
    "Shakespeare": 0.004539864603430033,
    "Victorian": 0.0035589272156357765,
    "NewPoems": 0.004812487866729498,
    "War&Peace": 0.005361916963011026,
    "RedChamber": 0.005125000141561031,
    "Dickens": 0.006139091681689024,
    "StKing": 0.005679713096469641,
    "HarryPotter": 0.006766146514564753,
}

# Keys to look for
key_idf = "Avg External IDF (token-weighted)"
key_dep = "Mean dependency depth"
key_pmi = "Mean NPMI"

def extract_value(path: Path, key: str):
    if not path.exists():
        return None
    pat = re.compile(rf"^{re.escape(key)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line.strip())
        if m:
            return float(m.group(1))
    return None

def main():
    rows = []
    for key in books:
        name = books[key]
        idf_val = extract_value(IDF_DIR / f"{key}.txt", key_idf)
        dep_val = extract_value(DEP_DIR / f"{key}.txt", key_dep)
        pmi_val = extract_value(PMI_DIR / f"{key}_mean_npmi.txt", key_pmi)
        twiker_val = twiker_vals.get(name, None)
        rows.append((name, idf_val, dep_val, pmi_val, twiker_val))

    # Reorder rows to match 'order'
    rows = [r for n in order for r in rows if r[0] == n]

    # Print markdown table
    print("| Name         | External IDF (token-weighted) | Mean Dependency Depth | Mean NPMI   | TWiKer Deviation (ours) |")
    print("|--------------|-------------------------------|-------------------------|-------------|---------------------------|")
    for name, idf_val, dep_val, pmi_val, twiker_val in rows:
        print(f"| {name:12} | {idf_val:.4f}                        | {dep_val:.3f}                  | {pmi_val:.6f}    | {twiker_val:.6f}       |")

    # Prepare data for plots
    names = [r[0] for r in rows]
    idf_vals = [r[1] for r in rows]
    dep_vals = [r[2] for r in rows]
    pmi_vals = [r[3] for r in rows]
    twi_vals = [r[4] for r in rows]

    metrics = [
        ("External IDF (token-weighted)", idf_vals),
        ("Mean Dependency Depth", dep_vals),
        ("Mean NPMI", pmi_vals),
        ("TWiKer Deviation (ours)", twi_vals),
    ]

    # Plot horizontal bar charts
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    y_pos = np.arange(len(names))

    for ax, (title, vals) in zip(axes, metrics):
        ax.barh(y_pos, vals, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
