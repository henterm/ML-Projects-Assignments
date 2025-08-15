# Подключите библиотеки из requirements

trainer = OpenEyesTrainer()
trainer.model = trainer.build_model()
trainer.model.load_weights("mobilenet_openeyes.weights.h5")

# Загрузка новых изображений
folder = "your/test/folder"  # <- Укажи путь к своей папке
image_paths = glob(os.path.join(folder, "*.jpg"))

y_true = [1 if "open" in p.lower() else 0 for p in image_paths]

# 3. Предсказания
y_scores = [trainer.predict_proba(p) for p in image_paths]

# 4. Расчёт EER, порога и метрик
eer, threshold = compute_eer(y_true, y_scores)
y_pred = [1 if s >= threshold else 0 for s in y_scores]
accuracy = accuracy_score(y_true, y_pred)

# 5. Вывод результатов
print(f"\n[RESULTS]")
print(f"EER: {eer:.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Accuracy: {accuracy:.4f}\n")
print(classification_report(y_true, y_pred))

# 6. ROC-кривая
fpr, tpr, _ = roc_curve(y_true, y_scores)
auc = np.trapz(tpr, fpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()