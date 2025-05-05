import os
import shutil
import random

def split_dataset(source_dir, output_dir, val_ratio=0.2):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        src_folder = os.path.join(source_dir, cls)
        images = os.listdir(src_folder)
        random.shuffle(images)

        split_index = int(len(images) * (1 - val_ratio))
        train_images = images[:split_index]
        val_images = images[split_index:]

        for subset, image_list in [('train', train_images), ('val', val_images)]:
            dest_folder = os.path.join(output_dir, subset, cls)
            os.makedirs(dest_folder, exist_ok=True)
            for img in image_list:
                shutil.copy2(os.path.join(src_folder, img), os.path.join(dest_folder, img))

        print(f"{cls}: {len(train_images)} train / {len(val_images)} val images")

source_folder = './serpapi_images'
output_folder = './dataset'
split_dataset(source_folder, output_folder, val_ratio=0.2)  # 80% train, 20% val
