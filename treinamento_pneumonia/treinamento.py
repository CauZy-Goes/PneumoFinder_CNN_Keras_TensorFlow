from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def treinar_modelo(model, train_gen, val_gen, epochs=50):
    """
    Treina o modelo com dataset balanceado (1200 NORMAL + 1200 PNEUMONIA).
    - Sem class_weight (não é mais necessário)
    - Callbacks otimizados para máxima performance
    - Ideal para transfer learning com ResNet50

    Retorna:
    - history: Histórico do treinamento
    """
    
    # === Callbacks Avançados ===
    callbacks = [
        ModelCheckpoint(
            filepath='best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=12,              # Mais paciência com dataset balanceado
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("Dataset balanceado (1200 + 1200) → class_weight DESATIVADO")
    print("Iniciando treinamento com ResNet50 + transfer learning...")

    # === Treinamento ===
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
        # class_weight REMOVIDO (não necessário!)
    )

    return history