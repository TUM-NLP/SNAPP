import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

model_path = "../models/hsb-de/nllb-finetuned-stage3_reverse"

tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="deu_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_file = "../data/hsb-de/dev.de-hsb.de"
output_file = "../data/hsb-de/dev.de-hsb.stage3.pred.hsb"

with open(input_file, "r", encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f.readlines()]


batch_size = 16
translations = []
chv_token_id = tokenizer.convert_tokens_to_ids("hsb_Latn")

for i in tqdm(range(0, len(src_sentences), batch_size)):
    batch = src_sentences[i:i+batch_size]
    
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=chv_token_id,
            max_length=128
        )
    
    batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    translations.extend(batch_translations)

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in translations:
        f.write(sentence + "\n")

print(f"Done! Translations saved to: {output_file}")