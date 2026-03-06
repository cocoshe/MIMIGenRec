import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except ImportError:
    print("Please install: pip install matplotlib")
    exit(1)


def load_time_distribution(reviews_path: str):
    """Stream reviews and aggregate counts by year and by year-month."""
    by_year = defaultdict(int)
    by_year_month = defaultdict(int)
    with open(reviews_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("unixReviewTime")
                if t is None:
                    continue
                ts = int(t)
                dt = datetime.utcfromtimestamp(ts)
                by_year[dt.year] += 1
                by_year_month[(dt.year, dt.month)] += 1
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return dict(by_year), dict(by_year_month)


def plot_distribution(by_year_month: dict, out_path: str, category: str = ""):
    """Line chart: review count by month (time series)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    keys = sorted(by_year_month.keys())
    xs = [y + (m - 1) / 12.0 for y, m in keys]
    counts = [by_year_month[k] for k in keys]
    ax.fill_between(xs, counts, alpha=0.3)
    ax.plot(xs, counts, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Time (year-month)")
    ax.set_ylabel("Number of reviews")
    title = "Review distribution by month"
    if category:
        title = f"{category} — " + title
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # X-axis: show year-month labels, tick every 6 months for denser scale
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    def fmt_year_month(x):
        if x < 0:
            return ""
        y, frac = int(x), x - int(x)
        m = (int(round(frac * 12)) % 12) + 1
        return f"{y}-{m:02d}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_year_month(x)))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Review time distribution: stat + visualize + save")
    parser.add_argument("--reviews", type=str, required=True, help="Path to review JSONL file")
    args = parser.parse_args()

    reviews_path = Path(args.reviews)
    if not reviews_path.is_absolute():
        reviews_path = (script_dir / reviews_path).resolve()
    else:
        reviews_path = reviews_path.resolve()
    if not reviews_path.exists():
        print(f"File not found: {reviews_path}")
        exit(1)

    print(f"Reading: {reviews_path}")
    by_year, by_year_month = load_time_distribution(str(reviews_path))
    total = sum(by_year.values())
    print(f"Total reviews: {total}")
    if total == 0:
        print("No data, exiting.")
        return

    # e.g. Toys_and_Games_5.json -> category_suffix "Toys_and_Games", category title "Toys and Games"
    stem = reviews_path.stem  # e.g. Toys_and_Games_5
    category_suffix = stem.replace("_5", "") if stem else ""
    category_title = category_suffix.replace("_", " ") if category_suffix else ""
    out_dir = reviews_path.parent / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"review_distribution_{category_suffix}.png" if category_suffix else "review_distribution.png"
    )
    plot_distribution(by_year_month, str(out_path), category=category_title)
    print("Done.")


if __name__ == "__main__":
    main()
