# rode no Python
from tensorflow.keras.models import load_model
model = load_model("best_model.keras")
print(model.layers[0])  # Deve ser <ResNet50>
print(len(model.layers[0].layers))  # ~175 camadas