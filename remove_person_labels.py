import os

def remove_class_from_labels(directory, class_to_remove):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = [line for line in lines if not line.startswith(f'{class_to_remove} ')]

            with open(file_path, 'w') as file:
                file.writelines(new_lines)

# Specify your directories
train_labels_dir = 'dataset_club_only/train/labels'
valid_labels_dir = 'dataset_club_only/valid/labels'

# Class index to remove (0 for 'person')
class_to_remove = 0

# Remove 'person' class from label files
remove_class_from_labels(train_labels_dir, class_to_remove)
remove_class_from_labels(valid_labels_dir, class_to_remove)
