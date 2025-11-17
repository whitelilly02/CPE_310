import os

LABEL_DIR = r"C:\CPE_310\Project_helmet\dataset_test\train\labels"

old_dataset_map = {0:0, 1:1, 2:2, 3:3}
new_dataset_map = {0:4, 1:5}

def update_label_indexes(label_dir, old_map, new_map, new_dataset_files):
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(label_dir, filename)

        # เลือก map ตาม dataset
        mapping = new_map if filename in new_dataset_files else old_map

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_index = int(parts[0])
            if old_index in mapping:
                parts[0] = str(mapping[old_index])
            else:
                print(f"⚠️ Warning: {filename} has unknown index {old_index}")
                continue
            new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w") as f:
            f.writelines(new_lines)

        print(f"✅ Updated: {filename}")

# 🔥 เรียกใช้งาน
new_dataset_files = ["bicycle_001.txt", "motorcycle_001.txt"]  # เพิ่มรายชื่อไฟล์ dataset ใหม่
update_label_indexes(LABEL_DIR, old_dataset_map, new_dataset_map, new_dataset_files)
