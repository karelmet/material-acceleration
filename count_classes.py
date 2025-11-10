import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter

path_data = os.getcwd() + '/DataCadot/'

# Choose split: 'train' or 'valid'
split = 'train'

# Get image paths
images_paths = sorted(glob.glob(path_data + f'images/{split}/*.jpg'))[:12]

# Load class names (you may need to adjust this path)
# Common YOLO class file locations
class_names_file = path_data + 'data.yaml'
classes = ['Basketball Field', 'Building', 'Crosswalk', 'Football Field',
           'Graveyard', 'Large Vehicle', 'Medium Vehicle', 'Playground',
           'Roundabout', 'Ship', 'Small Vehicle', 'Swimming Pool',
           'Tennis Court', 'Train'
           ]

def load_yolo_annotations(label_path, img_width, img_height):
    """Load YOLO format annotations from txt file."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            # chaque ligne contient 5 éléments : class_id et 2 coordonnées pour la box, width et height
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Convert from YOLO format (normalized center coords) to pixel coords
            x_center = cx * img_width
            y_center = cy * img_height
            box_width = w * img_width
            box_height = h * img_height
            
            # Convert to corner coordinates
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            
            annotations.append({
                'class_id': class_id,
                'bbox': [x1, y1, box_width, box_height]
            })
    
    return annotations

# Count category occurrences across all splits
category_counts = Counter()
all_label_paths = glob.glob(path_data + f'labels/{split}/*.txt')

for label_path in all_label_paths:
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                category_counts[class_name] += 1

# Print statistics
print(f"\nDataCadot Dataset Statistics ({split} split):")
print(f"Total images: {len(glob.glob(path_data + f'images/{split}/*.jpg'))}")
print(f"Total label files: {len(all_label_paths)}")

if classes:
    print(f"\nAvailable classes: {classes}")

print(f"\nCategory Distribution in {split.capitalize()} Set:")
if category_counts:
    for class_name, count in sorted(category_counts.items()):
        print(f"  {class_name}: {count}")
else:
    print("  No annotations found")

print(f"\nTotal annotations: {sum(category_counts.values())}")