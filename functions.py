import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter


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

def plot_sample_images(path_data, split, images_paths, classes):
    """Print images and corresponding boxes with annotations. """
    fig, axes = plt.subplot(3, 4, figsize=(16,12))
    axes = axes.flatten()

    colors = plt.cm.tab20.colors
    for idx, img_path in enumerate(images_paths):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size
        ax.imshow(img)

        # Get corresponding label file
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = path_data + f'labels/{split}/' + label_filename

        # Load annotations
        annotations = load_yolo_annotations(label_path, img_width, img_height)

        # Draw bounding boxes
        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = bbox
            class_id = ann['class_id']
            class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        
            # Get color for this class
            color = colors[class_id % len(colors)]
        
            # Create rectangle
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
            # Add label
            ax.text(x, y - 5, class_name, color='white', fontsize=8,
               bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))
    
        ax.set_title(f"{img_filename}\n{len(annotations)} objects", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(images_paths), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()