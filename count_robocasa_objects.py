import os
from collections import Counter

import matplotlib.pyplot as plt
import yaml


def count_objects_per_scene():
    layout_dir = os.path.expanduser(
        "~/.maniskill/data/scene_datasets/robocasa_dataset/assets/scenes/kitchen_layouts/"
    )

    # We want to know in how many files each object type/config appears
    object_presence_in_files = Counter()
    total_files = 0

    def get_labels_from_data(obj, labels_in_file):
        if isinstance(obj, dict):
            obj_type = obj.get("type")
            if obj_type:
                # Use only the primitive type name
                labels_in_file.add(obj_type)

            for v in obj.values():
                get_labels_from_data(v, labels_in_file)
        elif isinstance(obj, list):
            for item in obj:
                get_labels_from_data(item, labels_in_file)

    layout_files = [f for f in os.listdir(layout_dir) if f.endswith(".yaml")]
    total_files = len(layout_files)

    for filename in sorted(layout_files):
        with open(os.path.join(layout_dir, filename), "r") as f:
            try:
                data = yaml.safe_load(f)
                if not data:
                    continue

                labels_in_this_file = set()
                get_labels_from_data(data, labels_in_this_file)

                for label in labels_in_this_file:
                    object_presence_in_files[label] += 1

            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    return object_presence_in_files, total_files


if __name__ == "__main__":
    presence_counts, total_scenes = count_objects_per_scene()

    # Sort by frequency (how many scenes it appears in)
    sorted_presence = presence_counts.most_common()

    names = [item[0] for item in sorted_presence]
    # Frequency is (number of scenes with object) / (total scenes)
    frequencies = [item[1] / total_scenes for item in sorted_presence]

    print(f"Total scenes (layout files) analyzed: {total_scenes}")
    print("\nObject Presence Frequency (in what % of scenes they appear):")
    for name, count in sorted_presence:
        print(f"{name:40} : {count / total_scenes:6.1%} ({count}/{total_scenes})")

    # Plotting
    plt.figure(figsize=(16, 10))
    bars = plt.bar(names, frequencies)
    plt.xticks(rotation=90, fontsize=9)
    plt.xlabel("Object Type / Configuration")
    plt.ylabel("Presence Frequency (0.0 - 1.0)")
    plt.title(f"RoboCasa Object Presence Across {total_scenes} Kitchen Layouts")

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
            rotation=90,
        )

    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("robocasa_object_histogram.png")
    print(
        f"\nHistogram with {len(names)} categories saved to 'robocasa_object_histogram.png'"
    )
