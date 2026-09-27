import matplotlib.pyplot as plt
import pickle

# load saved history if you saved it
with open("training_history.pkl", "rb") as f:
    history, fine_tune_history = pickle.load(f)

train_loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

plt.figure(figsize=(8,6))
plt.plot(train_loss, label="Train Loss", marker='o')
plt.plot(val_loss, label="Validation Loss", marker='o')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()