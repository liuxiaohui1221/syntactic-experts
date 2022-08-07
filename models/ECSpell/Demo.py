import torch
import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoModelForTokenClassification, BertModel, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,type(device))
model_dir='./bert/results/checkpoint'
model=AutoModel.from_pretrained(model_dir).to(device)
model.eval()
# print(model)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
block_texts='这块民表带带相传'
inputs = tokenizer(block_texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    outputs= model(**inputs)
    print(outputs)
    # decode_tokens = tokenizer.decode(torch.argmax(outputs[:2], dim=-1), skip_special_tokens=True)
    # print(decode_tokens)