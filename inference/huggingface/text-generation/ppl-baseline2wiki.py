import torch
from tqdm import tqdm
import deepspeed
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher
from argparse import ArgumentParser
from deepspeed.accelerator import get_accelerator

parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="model_name")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8", "bfloat16"], help="data-type")
parser.add_argument("--num_inputs", default=1, type=int, help="number of test inputs")
parser.add_argument("--min_length", default=200, type=int, help="minimum tokens generated")
parser.add_argument("--max_length", default=300, type=int, help="maximum tokens generated")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--use_kernel", default=True, help="enable kernel-injection")
args = parser.parse_args()

def string_similarity(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    similarity_ratio = matcher.ratio()
    return similarity_ratio

eos_id = 2
input_length = 512
output_length = 512
device = get_accelerator().device_name()

checkpoint = args.model
data_set_path = "~/my_project/DSE/DataSets/wikitext2/"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
test_rawdata = load_from_disk("~/my_project/DSE/DataSets/wikitext2")
encod_ids = tokenizer("\n\n".join(test_rawdata["text"]), return_tensors="pt")['input_ids'][0] # get ids

wiki_ids = encod_ids[input_length:input_length+output_length]
wiki_contents = tokenizer.decode(wiki_ids)

seq_len = len(encod_ids)
ppl = 0
predict_content = str()
# for begin_loc in tqdm(range(output_length), desc="Calculating Perplexity: ", ncols=100):
for begin_loc in range(output_length):
    end_loc = min(begin_loc+input_length, seq_len)
    input_ids = encod_ids[begin_loc:end_loc]
    xi_id = encod_ids[end_loc]
    input_content = tokenizer.decode(input_ids) # decode back to the content

    input_encode = tokenizer(input_content, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_encode)
        last_word_logits = outputs.logits[0, -1, :]
        last_word_logits[eos_id] = -float("inf")
        last_word_probs = torch.softmax(last_word_logits, dim=-1)
        max_prob_idx = torch.argmax(last_word_probs, dim=-1)
        predict_word = tokenizer.decode(max_prob_idx)
        predict_content = predict_content + predict_word
        prob_xi = torch.log(last_word_probs[xi_id])
        ppl = ppl + prob_xi
ppl = torch.exp(-1*ppl/output_length)
similarity = string_similarity(wiki_contents, predict_content)
print(f"wikitext2: {wiki_contents}")
print(f"{'-'*60}")
print(f"baseline output: {predict_content}")
print(f"{'-'*60}")
print(f"The perplexity of baseline on wikitext2 is: {ppl}")
print(f"The similarity ratio is: {similarity*100}%")
