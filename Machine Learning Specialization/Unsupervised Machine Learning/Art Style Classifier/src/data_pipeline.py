import os
import zipfile
import shutil
from pathlib import Path
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def extract_data(zip_path, output_path, images_per_class, random_selection):
    # === UNZIP ===
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("extracted")

    # === DOWNSAMPLE ===
    input_root = Path("extracted/images")
    output_root = Path(output_path)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for class_dir in input_root.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.*"))
            if random_selection:
                images = random.sample(images, min(images_per_class, len(images)))
            else:
                images = images[:images_per_class]
            
            target_dir = output_root / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in images:
                shutil.copy(img_path, target_dir / img_path.name)

    # === DELETE EXTRACTED FOLDER ===
    shutil.rmtree("extracted")

    print(f"✅ Reduced dataset created at: {output_root.resolve()}")


def zip_folder(folder_path):
    shutil.make_archive(folder_path, 'zip', folder_path)

def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    data_path = Path(data_dir).with_suffix('.zip')
    # If backup zip is provided and data folder exists, zip it and delete the folder
    if data_path.exists():
        print(f"Extracting {data_dir}...")
        unzip_folder(data_path, data_dir)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

    train = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training'
    )

    val = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )
    return train, val


def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
