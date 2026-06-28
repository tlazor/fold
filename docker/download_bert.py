from transformers import AutoTokenizer, AutoModelForMaskedLM

for model_name in (
    "bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
):
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    AutoModelForMaskedLM.from_pretrained(model_name)
