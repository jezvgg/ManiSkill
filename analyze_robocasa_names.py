import os
from collections import Counter

import matplotlib.pyplot as plt
import yaml


def analyze_base_names():
    layout_dir = os.path.expanduser(
        "~/.maniskill/data/scene_datasets/robocasa_dataset/assets/scenes/kitchen_layouts/"
    )
    layout_files = [f for f in os.listdir(layout_dir) if f.endswith(".yaml")]

    name_presence = Counter()
    total_layouts = len(layout_files)

    for l_file in layout_files:
        with open(os.path.join(layout_dir, l_file), "r") as f:
            data = yaml.safe_load(f)

        names_in_layout = set()

        def find_names(obj):
            if isinstance(obj, dict):
                if "name" in obj:
                    names_in_layout.add(obj["name"])
                for v in obj.values():
                    find_names(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_names(item)

        find_names(data)

        for name in names_in_layout:
            name_presence[name] += 1

    print(f"Base Name Presence across {total_layouts} layouts:")
    print("-" * 50)
    # Sort by frequency, then by name
    sorted_names = sorted(name_presence.items(), key=lambda x: (-x[1], x[0]))

    plot_names = []
    plot_freqs = []

    for name, count in sorted_names:
        freq = count / total_layouts
        print(f"{name:30} : {freq:6.1%} ({count}/{total_layouts})")
        # Filter for plotting to keep it readable (names appearing in >20% of layouts)
        if freq >= 0.20:
            plot_names.append(name)
            plot_freqs.append(freq)

    # Plotting
    plt.figure(figsize=(14, 8))
    bars = plt.bar(plot_names, plot_freqs, color="skyblue", edgecolor="navy")
    plt.xticks(rotation=90, fontsize=9)
    plt.ylabel("Presence Frequency (0.0 - 1.0)")
    plt.xlabel("Base Object Name")
    plt.title(
        f"RoboCasa Base Name Presence Frequency (Found in >20% of {total_layouts} layouts)"
    )

    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.0%}",
            va="bottom",
            ha="center",
            fontsize=8,
        )

    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("robocasa_names_histogram.png")
    print(
        f"\nHistogram for names with >=20% presence saved to 'robocasa_names_histogram.png'"
    )


if __name__ == "__main__":
    analyze_base_names()

    analyze_base_names()
