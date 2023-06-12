import torch

model_path = "export/bert_lm_mask"
model = torch.load(model_path).to(device)
device = torch.device("cuda:0")
text = "The capital of Fracne, [MASK], contains the eiffel tower."
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("dataset/bert-base-uncased")
text_encode = tokenizer.encode_plus(text)

input_ids = torch.tensor(text_encode["input_ids"], device=device)
input_ids = torch.reshape(input_ids, [1, -1])
input_seg = torch.ones(len(input_ids), dtype=torch.long, device=device)
input_seg = torch.reshape(input_seg, [1, -1])
with torch.no_grad():
    _, out_mask = model.forward(input_ids, input_seg)
