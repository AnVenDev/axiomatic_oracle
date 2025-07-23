from blockchain_publisher import publish_ai_prediction, batch_publish_predictions
from sample_property import sample_response, multiple_samples

# Singola pubblicazione
print("📌 Esecuzione singola:")
result = publish_ai_prediction(sample_response)
print(result)

# Pubblicazione batch
print("\n📦 Esecuzione batch:")
batch_results = batch_publish_predictions(multiple_samples)
for res in batch_results:
    print(res)