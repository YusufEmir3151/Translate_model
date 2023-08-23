import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader, Dataset

# GPU kullanımı (varsa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri dosyalarınızın yolları
train_data_path = ("/kaggle/input/training-translate/train_v1.json")
valid_data_path = ("/kaggle/input/training-translate/validation_v1.json")

# Hyperparametreler
max_length = 512
batch_size = 16
learning_rate = 2e-5
epochs = 2

# T5 modeli ve tokenizasyon aracını yükleyin
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Birden fazla GPU kullanımı için DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
else:
    print("Don't use GPU")

model.to(device)  # Modeli cihaza taşı


# Veri setini özelleştirilmiş bir sınıf ile tanımlayın
class TranslationDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8-sig") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        translation = self.data[idx]["translation"]
        source_text = translation["en"]
        target_text = translation["tr"]
        source_encoded = tokenizer.encode(source_text, max_length=max_length, truncation=True, padding="max_length")
        target_encoded = tokenizer.encode(target_text, max_length=max_length, truncation=True, padding="max_length")
        return {"source_ids": torch.tensor(source_encoded), "target_ids": torch.tensor(target_encoded)}


print("Yükleme işlemleri")
train_dataset = TranslationDataset(train_data_path)
valid_dataset = TranslationDataset(valid_data_path)

# Veri yükleyicileri oluşturun
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Optimizasyon ve kayıp fonksiyonunu tanımlayın
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

print("Eğitim")
# Eğitim döngüsü
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_processed = 0  # İşlenen toplam veri sayısı

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        source_ids = batch["source_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        outputs = model(input_ids=source_ids, labels=target_ids)
        loss = outputs.loss.mean()
        loss.mean().backward()
        optimizer.step()
        total_loss += loss.item()

        # Her batch işlendiğinde işlenen veri sayısını güncelle
        total_processed += len(batch)

        # Veri yükleme sayısını ve işlenen veri sayısını ekrana yazdır
        print(
            f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Processed: {total_processed}/{len(train_dataset)} - Loss: {loss:.4f}")

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {average_loss:.4f}")

    # Doğrulama
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for batch in valid_loader:
            source_ids = batch["source_ids"].to(device)  # Verileri cihaza gönder
            target_ids = batch["target_ids"].to(device)  # Verileri cihaza gönder

            outputs = model(input_ids=source_ids, labels=target_ids)
            loss = outputs.loss
            valid_loss += loss.item()

        average_valid_loss = valid_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {average_valid_loss:.4f}")

# Modelinizi kaydedin
model_save_path = "/kaggle/working/trained_translation_model_v1"  # Değiştirilmiş yol
model.module.save_pretrained(model_save_path)  # DataParallel sarmalayıcısını çıkarmak için .module kullan
