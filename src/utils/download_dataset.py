from datasets import load_dataset

# carrega automaticamente do cache local,
# e baixa só se ainda não estiver no cache
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")

print(dataset)
print(dataset["train"][0])
