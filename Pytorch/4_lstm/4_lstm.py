import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product
import numpy as np

# Örnek metin
text = """ Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performansı harika.
Kesinlikle tavsiye ederim ve öneririm!"""

# --------------------------
# PRE-PROCESSING
# --------------------------
words = text.replace(".", "").replace("!", "").lower().split()

# Kelime frekansına göre sıralama ve indeksleme
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# Eğitim verisi: (kelime, bir sonraki kelime)
data = [(words[i], words[i+1]) for i in range(len(words) - 1)]

# --------------------------
# LSTM MODEL TANIMI
# --------------------------
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # (seq_len, embedding_dim)
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        output = self.fc(lstm_out[-1])  # sadece son output kullanılacak
        return output

# --------------------------
# Yardımcı Fonksiyon
# --------------------------
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

# --------------------------
# HYPERPARAMETER TUNING
# --------------------------
embedding_sizes = [8, 16]
hidden_sizes = [32, 64]
learning_rates = [0.01, 0.005]

best_loss = float("inf")
best_params = {}
print("Hyperparameter tuning başlatıldı...\n")

for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: Embedding: {emb_size}, Hidden: {hidden_size}, Learning Rate: {lr}")
    
    model = LSTM(len(vocab), emb_size, hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_tensor = prepare_sequence([word], word_to_ix)
            target_tensor = prepare_sequence([next_word], word_to_ix)
            output = model(input_tensor)
            loss = loss_function(output.view(1, -1), target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} Loss: {epoch_loss:.4f}")
    
    print(f"Toplam Loss: {epoch_loss:.4f}\n")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_params = {
            "embedding_dim": emb_size,
            "hidden_dim": hidden_size,
            "learning_rate": lr
        }

print("En iyi hiperparametreler:")
print(best_params)
