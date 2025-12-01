from tensorflow.keras.models import Sequential  # Importa o modelo sequencial (camadas empilhadas linearmente)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Importa as camadas necessárias
from tensorflow.keras.applications import ResNet50  # Importa o modelo pré-treinado ResNet50 para transfer learning
from tensorflow.keras.optimizers import Adam  # Importa o otimizador Adam com taxa de aprendizado ajustável

def criar_modelo(input_shape=(224, 224, 3)):
    """
    Cria e compila um modelo de rede neural convolucional (CNN) avançado usando transfer learning com ResNet50
    para classificação binária, melhorando a acurácia em tarefas como detecção de pneumonia em raios-X torácicos.

    Parâmetros:
    - input_shape: Tupla que define o formato das entradas do modelo (altura, largura, canais de cor).
                   O default é (224, 224, 3), indicando imagens coloridas de 224x224.

    Retorna:
    - model: Um modelo CNN compilado e pronto para treinamento, com acurácia potencialmente superior a 95% em datasets como Chest X-Ray Pneumonia.
    """
    # Carrega o modelo base ResNet50 pré-treinado no ImageNet, sem a camada de saída superior
    # Isso permite transfer learning: usa features aprendidas em milhões de imagens para melhorar performance em datasets menores
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Congela as camadas do modelo base inicialmente para evitar atualizar pesos pré-treinados
    # (Pode descongelar depois para fine-tuning com base_model.trainable = True)
    base_model.trainable = False

    model = Sequential([  # Define um modelo sequencial
        base_model,  # Adiciona o ResNet50 como backbone para extração de features avançadas
        
        # Camada de pooling global: Reduz a dimensionalidade da saída do base_model (melhor que Flatten para reduzir parâmetros e overfitting)
        GlobalAveragePooling2D(),
        
        # Camada totalmente conectada: Mais neurônios (256) para aprender relações complexas das features extraídas
        # Função de ativação ReLU para não-linearidade

        Dense(1024, activation='relu'),
        Dropout(0.5),

        # Dense(256, activation='relu'),
        
        # # Dropout: Desativa 50% dos neurônios durante o treinamento para prevenir overfitting e melhorar generalização
        # Dropout(0.5),
        
        # Camada de saída: 1 neurônio para classificação binária (NORMAL ou PNEUMONIA)
        # Função de ativação Sigmoid para probabilidades entre 0 e 1
        Dense(1, activation='sigmoid')
    ])

    # Compila o modelo:
    # - Otimizador: Adam com taxa de aprendizado baixa (0.0001) para convergência estável em transfer learning
    # - Função de perda: Binary Crossentropy para classificação binária
    # - Métrica: Accuracy para medir taxa de acerto
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model  # Retorna o modelo criado