import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_image_with_annotations(image, annotations, class_names=None, figsize=(8, 8)):
    """
    Display an image with bounding boxes and class labels.

    Args:
        image (PIL.Image): The image to display.
        annotations (List[dict]): Output from load_yolo_annotations.
            Each annotation is a dict containing:
               {
                   "class_id": int,
                   "bbox": [x1, y1, w, h]  # pixel coordinates
               }
        class_names (list or None): Optional list of class names.
        figsize (tuple): Figure size.
    """

    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    colors = plt.cm.tab20.colors

    for ann in annotations:
        x, y, w, h = ann['bbox']
        class_id = ann['class_id']
        label = class_names[class_id] if class_names else f"class_{class_id}"

        color = colors[class_id % len(colors)]

        # draw rectangle
        rect = patches.Rectangle(
            (x, y), w, h, 
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        # draw label text
        ax.text(
            x, y -5,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none')
        )
    plt.axis('off')
    plt.show()

def yolo_boxes_to_annotations(boxes: np.ndarray, img_width, img_height):
    """
    Convertit un tableau (N,5) YOLO normalisé [cls, xc, yc, w, h]
    en liste de dicts :
        {"class_id": int, "bbox": [x, y, w, h]} en pixels
    """
    annotations = []
    if boxes.size == 0:
        return annotations

    for cls, xc, yc, bw, bh in boxes:
        # remettre en pixels
        bw_pix = bw * img_width
        bh_pix = bh * img_height
        x_center = xc * img_width
        y_center = yc * img_height

        x = x_center - bw_pix / 2.0
        y = y_center - bh_pix / 2.0

        ann = {
            "class_id": int(cls),
            "bbox": [float(x), float(y), float(bw_pix), float(bh_pix)],
        }
        annotations.append(ann)

    return annotations