import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import seaborn as sns

def avaliar_modelo(model, test_gen):
    """
    Avalia o modelo com MÁXIMA PRECISÃO e FORMATAÇÃO PROFISSIONAL para TCC.
    
    MELHORIAS:
    - Predição CORRETA com reset do gerador
    - Métricas: Accuracy, F1, AUC-ROC, Sensibilidade, Especificidade
    - Matriz de confusão com heatmap bonito
    - Relatório formatado com 4 casas decimais
    """
    print("\n" + "="*60)
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("="*60)

    # === 1. Reset do gerador (OBRIGATÓRIO!) ===
    test_gen.reset()
    
    # === 2. Avaliação direta ===
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Acurácia no Teste : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Perda no Teste    : {test_loss:.4f}")

    # === 3. Predições CORRETAS (com reset) ===
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # === 4. Métricas Avançadas ===
    report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'], digits=4, output_dict=True)
    auc = roc_auc_score(y_true, y_pred_prob)

    print(f"\nAUC-ROC           : {auc:.4f}")
    print(f"F1-Score (média)  : {report['weighted avg']['f1-score']:.4f}")
    print(f"Sensibilidade     : {report['PNEUMONIA']['recall']:.4f}")
    print(f"Especificidade    : {report['NORMAL']['recall']:.4f}")

    # === 5. Matriz de Confusão com Heatmap ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'], 
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Matriz de Confusão - Conjunto de Teste', fontsize=14, pad=20)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Predição do Modelo', fontsize=12)
    plt.tight_layout()
    plt.show()

    # === 6. Curva ROC ===
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # === 7. Relatório Completo ===
    print("\nRELATÓRIO DE CLASSIFICAÇÃO DETALHADO")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'], digits=4))

    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'auc': auc,
        'f1_weighted': report['weighted avg']['f1-score'],
        'cm': cm
    }


def plotar_historico(history_dict):
    """
    Plota gráficos PROFISSIONAIS com histórico combinado (transfer learning + fine-tuning).
    
    Parâmetros:
    - history_dict: dict com chaves 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    """
    if not isinstance(history_dict, dict):
        history_dict = history_dict.history

    epochs = range(1, len(history_dict['accuracy']) + 1)

    plt.figure(figsize=(15, 6))

    # === Acurácia ===
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['accuracy'], 'b-o', label='Treino', linewidth=2, markersize=4)
    plt.plot(epochs, history_dict['val_accuracy'], 'r-o', label='Validação', linewidth=2, markersize=4)
    plt.title('Acurácia ao Longo das Épocas', fontsize=14, pad=15)
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)

    # === Perda ===
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['loss'], 'b-o', label='Treino', linewidth=2, markersize=4)
    plt.plot(epochs, history_dict['val_loss'], 'r-o', label='Validação', linewidth=2, markersize=4)
    plt.title('Perda ao Longo das Épocas', fontsize=14, pad=15)
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()