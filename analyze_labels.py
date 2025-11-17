import sys
import os
import yaml
from collections import Counter


def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_label_files(base_dirs):
    files = []
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for fname in os.listdir(base):
            if fname.endswith('.txt'):
                files.append(os.path.join(base, fname))
    return files


def scan_labels(label_files, max_examples_per_class=5):
    counts = Counter()
    examples = {}
    for p in label_files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(parts[0])
                    counts[cls] += 1
                    examples.setdefault(cls, []).append(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    # reduce examples
    for k in list(examples.keys()):
        examples[k] = examples[k][:max_examples_per_class]
    return counts, examples


if __name__ == '__main__':
    # usage: python analyze_labels.py [path/to/data.yaml]
    if len(sys.argv) > 1:
        data_yaml = sys.argv[1]
    else:
        data_yaml = os.path.join(os.path.dirname(__file__), 'data.yaml')

    if not os.path.isfile(data_yaml):
        print(f"data.yaml not found at {data_yaml}")
        print("Pass path to data.yaml as first argument, e.g. python analyze_labels.py dataset_motorcycle/data.yaml")
        sys.exit(1)

    cfg = load_data_yaml(data_yaml)
    print(f"Loaded data.yaml: {data_yaml}\n")
    print("Declared classes (names):")
    names = cfg.get('names')
    if names is None:
        print("  (no 'names' in data.yaml)")
    else:
        for i, n in enumerate(names):
            print(f"  {i}: {n}")

    # find label directories relative to data.yaml
    base_dir = os.path.dirname(os.path.abspath(data_yaml))
    # common locations
    possible = [
        os.path.join(os.path.dirname(__file__), 'train', 'labels'),
        os.path.join(os.path.dirname(__file__), 'valid', 'labels'),
        os.path.join(os.path.dirname(__file__), 'test', 'labels'),
        os.path.join(base_dir, 'labels'),
    ]
    label_files = []
    for p in possible:
        if os.path.isdir(p):
            label_files.extend(find_label_files([p]))

    # fallback: scan common train/valid/test
    if not label_files:
        for root in [os.path.join(os.path.dirname(__file__), 'train', 'labels'),
                     os.path.join(os.path.dirname(__file__), 'valid', 'labels'),
                     os.path.join(os.path.dirname(__file__), 'test', 'labels')]:
            if os.path.isdir(root):
                label_files.extend(find_label_files([root]))

    if not label_files:
        print("No label files found under train/labels, valid/labels, or test/labels")
        sys.exit(1)

    print(f"Found {len(label_files)} label files (scanning)...")
    counts, examples = scan_labels(label_files)

    print("\nLabel counts by class id:")
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls}: {cnt} annotations")

    print("\nExample files per class (up to 5 each):")
    for cls, files in examples.items():
        print(f"  Class {cls}:")
        for f in files:
            print(f"    - {os.path.relpath(f)}")

    # check if any class ids exceed declared names
    max_cls = max(counts.keys()) if counts else -1
    if names is not None and max_cls >= len(names):
        print('\nWARNING: Found label class id(s) >= declared number of class names in data.yaml')
        print(f"  Max class id found: {max_cls}, declared names count: {len(names)}")
        print("  This indicates a mismatch: annotations use different class indices than the provided data.yaml names mapping.")

    print('\nDone.')
