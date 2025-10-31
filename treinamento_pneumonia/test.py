from preprocessamento import carregar_dados
from tensorflow.keras.models import load_model
from avaliacao import avaliar_modelo, plotar_historico




train_dir = "dataset\\train"
val_dir = "dataset\\val"
test_dir = "dataset\\test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen, val_gen, test_gen = carregar_dados(train_dir, val_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
BEST_MODEL_PATH = 'best_model.keras'

model = load_model(BEST_MODEL_PATH)  # Garante avaliação com o MELHOR modelo
avaliar_modelo(model, test_gen)

# === PASSO 6: PLOTAR HISTÓRICO COMBINADO ===
print("6. Gerando gráficos de treinamento...")
# Combina histórico do transfer learning + fine-tuning
if 'history' in locals():
    full_history = {}
    for key in history.history.keys():
        full_history[key] = history.history[key] + history_fine.history[key]
    plotar_historico(full_history)
else:
    plotar_historico(history_fine)