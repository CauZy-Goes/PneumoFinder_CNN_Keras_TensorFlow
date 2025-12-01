import os
import shutil
import random

# === CONFIGURAÇÕES ===
origem_normal = "dataset/NORMAL"       # pasta com as 1800 imagens normais
origem_pneumonia = "dataset/PNEUMONIA" # pasta com as 1800 imagens de pneumonia
destino_base = "dataset2"

# Quantidades desejadas
qtd_train = 1200
qtd_val = 400
qtd_test = 200

# === FUNÇÃO PRINCIPAL ===
def distribuir_imagens(origem, destino_base, classe):
    # Garante que o diretório de destino existe
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(destino_base, split, classe), exist_ok=True)

    # Lista e embaralha as imagens
    imagens = [f for f in os.listdir(origem) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imagens)

    if len(imagens) < (qtd_train + qtd_val + qtd_test):
        raise ValueError(f"Número insuficiente de imagens em {origem}: {len(imagens)}")

    # Divide as imagens
    train_imgs = imagens[:qtd_train]
    val_imgs = imagens[qtd_train:qtd_train + qtd_val]
    test_imgs = imagens[qtd_train + qtd_val:qtd_train + qtd_val + qtd_test]

    # Copia os arquivos
    def copiar(lista, split):
        destino = os.path.join(destino_base, split, classe)
        for img in lista:
            shutil.copy(os.path.join(origem, img), os.path.join(destino, img))

    copiar(train_imgs, "train")
    copiar(val_imgs, "val")
    copiar(test_imgs, "test")

    print(f"{classe}: {len(train_imgs)} treino, {len(val_imgs)} validação, {len(test_imgs)} teste")


# === EXECUÇÃO ===
if __name__ == "__main__":
    random.seed(42)  # garante reprodução
    distribuir_imagens(origem_normal, destino_base, "NORMAL")
    distribuir_imagens(origem_pneumonia, destino_base, "PNEUMONIA")

    print("\n✅ Dataset organizado em:", os.path.abspath(destino_base))
