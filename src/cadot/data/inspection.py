import torch
from collections import Counter

def get_class_weights_from_dataset(dataset, num_classes, threshold=0.1):
    """
    Calculate class weights based on class frequency in dataset.
    
    Args:
        dataset: Your dataset object
        num_classes: Total number of classes
        threshold: Classes below this frequency ratio are considered rare
    
    Returns:
        class_weights: Tensor of weights for each class
        rare_class_indices: List of rare class indices
    """
    # Count class occurrences
    class_counts = Counter()
    for sample in dataset:
        labels = sample['cls']  # Adjust based on your dataset structure
        class_counts.update(labels.tolist())
    
    # Convert to tensor
    counts = torch.zeros(num_classes)
    for cls_id, count in class_counts.items():
        counts[int(cls_id)] = count
    
    # Calculate frequencies
    total = counts.sum()
    frequencies = counts / total
    
    # Identify rare classes
    rare_class_indices = torch.where(frequencies < threshold)[0].tolist()
    common_class_indices = torch.where(frequencies > threshold * 3)[0].tolist()
    
    # Calculate inverse frequency weights
    class_weights = torch.ones(num_classes)
    class_weights = 1.0 / (frequencies + 1e-6)
    class_weights = class_weights / class_weights.mean()  # Normalize
    
    return class_weights, rare_class_indices, common_class_indices

# Usage
class_weights, rare_indices, common_indices = get_class_weights_from_dataset(train_dataset, num_classes=80)
print(f"Rare classes: {rare_indices}")
print(f"Class weights: {class_weights}")


# DataCadot dataset statistics (Train split):
# Total images: 3234
# Total label files: 3234

# Category distribution in Train set:
#0   Basketball Field: 15
#1   Building: 30764
#2   Crosswalk: 3751
#3   Football Field: 77
#4   Graveyard: 130
#5   Large Vehicle: 841
#6   Medium Vehicle: 3865
#7   Playground: 44
#8   Roundabout: 37
#9   Ship: 300
#10   Small Vehicle: 35149
#11   Swimming Pool: 60
#12   Tennis Court: 98
#13   Train: 67

# Total annotations: 75198