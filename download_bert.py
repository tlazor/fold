from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = 'bert-base-multilingual-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)