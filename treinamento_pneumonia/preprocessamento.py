from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def carregar_dados(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Carrega dados com preprocess_input do ResNet50 e augmentation realista.
    REMOVIDO: .cache() e .prefetch() → incompatíveis com DirectoryIterator
    """
    
    # === DATA AUGMENTATION (treino) ===
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # === VALIDAÇÃO E TESTE (sem augmentation) ===
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # === GERADORES ===
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=42
    )

    # === PRINT INFORMATIVO ===
    print(f"\nDADOS CARREGADOS:")
    print(f"  Treino: {train_gen.samples} imagens")
    print(f"  Validação: {val_gen.samples} imagens")
    print(f"  Teste: {test_gen.samples} imagens")
    print(f"  Classes: {list(train_gen.class_indices.keys())}\n")

    return train_gen, val_gen, test_gen