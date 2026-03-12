'''
This project is for fine-tuning a large language model (LLM) to generate Tang poems.
The code is adapted from the NTU course.
Difference from the course. We use smaller LLM models, like Qwen2.5-3B-Instruct and MediaTek-Research/Llama-Breeze2-3B-Instruct,
so we can implement the fine-tuning in our local computer which has one RTX 3060 Ti (8GB).
Note that we are using LoRA to fine-tune the model, which is a parameter-efficient fine-tuning method. 
This means that we only update a small portion of the model's parameters during fine-tuning, 
which can significantly reduce the computational resources required for training.
Author: Dr. KaHou Leong
Date: 12/3/2026
'''

import os
import sys
import argparse
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers, datasets
from peft import PeftModel
from colorama import *

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

# model_name = "taide/TAIDE-LX-7B-Chat" # Name of the model you want to use. 
model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "MediaTek-Research/Llama-Breeze2-3B-Instruct"

"""## Fix Random Seeds
There may be some randomness involved in the fine-tuning process. We fix random seeds to make the result reproducible.
"""

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# This is desigened for QWEN2.5-3B-Instruct.
def generate_training_data(data_point):
    """
    (1) Goal:
        - This function is used to transform a data point (input and output texts) to tokens that our model can read

    (2) Arguments:
        - data_point: dict, with field "instruction", "input", and "output" which are all str

    (3) Returns:
        - a dict with model's input tokens, attention mask that make our model causal, and corresponding output targets
    Note: This is designed for Qwen2.5-3B-Instruct and is different than the NTU template which is designed for taide/TAIDE-LX-7B-Chat
    """
    system_prompt = (
        # "You are a helpful assistant and good at writing Tang poem. "
        "你是一個樂於助人的助手且擅長寫唐詩，一切對話都是用繁體中文進行的。"
    )

    user_prompt = f'{data_point["instruction"]}\n{data_point["input"]}'.strip()
    assistant_text = data_point["output"].strip() # For fine tuning

    # Prompt without assistant answer: used to count how many tokens to mask
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # leaves the prompt ready for assistant response
    )

    # Full conversation including target answer
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]

    len_prompt_tokens = len(prompt_ids)

    labels = [-100] * len_prompt_tokens + full_tokens[len_prompt_tokens:]

    # Keep labels same length as input_ids
    # labels = labels[: len(full_tokens)]

    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": [1] * len(full_tokens),
    }

# This is desigened for QWEN2.5-3B-Instruct.
def evaluate(instruction, generation_config, max_len, input="", verbose=True):
    """
    Generate a response with Qwen2.5-Instruct chat template.
    """
    system_prompt = (
        # "You are a helpful assistant and good at writing Tang poem. "
        "你是一個樂於助人的助手且擅長寫唐詩，一切對話都是用繁體中文進行的。"
    )

    user_prompt = f"{instruction}\n{input}".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    generation_output = model.generate(
        input_ids=inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    ) # type: ignore

    # Only decode newly generated tokens
    # sequences: (batch_size, input_length + generated_length)
    # So, [0] is for the first sample. The shape is (input_length + generated_length)
    generated_ids = generation_output.sequences[0][inputs.shape[-1]:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if verbose:
        print(output)

    return output



"""## Inference before Fine-tuning
Let's first see what our model can do without fine-tuning.

The following code block takes about **2** minutes to run if you use the default setting, but it may vary depending on the condition of Colab.
"""

""" It is recommmended NOT to change codes in this cell """

cache_dir = "./cache"
# Here we use 4-bit quantization with NF$ format
# double quant means the scaling factor for LoRA is also stored in NF4
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
from huggingface_hub import login
# login() # Some models, like Taide, requires access to use.

# Set Hyperarameters for Fine-tuning

num_train_data = 4096 # 設定用來訓練的資料數量，可設置的最大值為5000。在大部分情況下會希望訓練資料盡量越多越好，這會讓模型看過更多樣化的詩句，進而提升生成品質，但是也會增加訓練的時間
                      # 使用預設參數(1040): fine-tuning大約需要25分鐘，完整跑完所有cell大約需要50分鐘
                      # 使用最大值(5000): fine-tuning大約需要100分鐘，完整跑完所有cell大約需要120分鐘

""" You may want (but not necessarily need) to change some of these hyperparameters """

output_dir = "./trained_result"  # 設定作業結果輸出目錄 (如果想要把作業結果存在其他目錄底下可以修改這裡，強烈建議存在預設值的子目錄下，也就是Google Drive裡)
ckpt_dir = "./exp2" # 設定model checkpoint儲存目錄 (如果想要將model checkpoints存在其他目錄下可以修改這裡)
num_epoch = 1  # 設定訓練的總Epoch數 (數字越高，訓練越久，若使用免費版的colab需要注意訓練太久可能會斷線)
LEARNING_RATE = 3e-4  # 設定學習率

""" It is recommmended NOT to change codes in this cell """

cache_dir = "./cache"  # 設定快取目錄路徑
from_ckpt = False  # 是否從checkpoint載入模型的權重，預設為否
ckpt_name = None  # 從特定checkpoint載入權重時使用的檔案名稱，預設為無
dataset_dir = "./project_data_main/Tang_training_data.json"  # 設定資料集的目錄或檔案路徑
logging_steps = 20  # 定義訓練過程中每隔多少步驟輸出一次訓練誌
save_steps = 65  # 定義訓練過程中每隔多少步驟保存一次模型
save_total_limit = 3  # 控制最多保留幾個模型checkpoint
report_to = "none"  # 設定上報實驗指標的目標，預設為無
MICRO_BATCH_SIZE = 4  # 定義微批次的大小
BATCH_SIZE = 16  # 定義一個批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 計算每個微批次累積的梯度步數
CUTOFF_LEN = 256  # 設定文本截斷的最大長度
LORA_R = 8  # 設定LORA（Layer-wise Random Attention）的R值
LORA_ALPHA = 16  # 設定LORA的Alpha值
LORA_DROPOUT = 0.05  # 設定LORA的Dropout率
VAL_SET_SIZE = 0  # 設定驗證集的大小，預設為無
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"] # 設定目標模組，這些模組的權重將被保存為checkpoint
device_map = "auto"  # 設定設備映射，預設為"auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 獲取環境變數"WORLD_SIZE"的值，若未設定則預設為1
ddp = world_size != 1  # 根據world_size判斷是否使用分散式數據處理(DDP)，若world_size為1則不使用DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

## Start Fine-tuning


# create the output directory you specify
os.makedirs(output_dir, exist_ok = True)
os.makedirs(ckpt_dir, exist_ok = True)


"""##  Testing
The fine-tuning process is done. We then want to test whether our model can do the task that we wanted it to do before but failed.

We need to first load the fine-tuned model for checkpoint we saved.
"""

""" It is recommmended NOT to change codes in this cell """

# find all available checkpoints
ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if (ckpt.startswith("checkpoint-")):
        ckpts.append(ckpt)

# list all the checkpoints
ckpts = sorted(ckpts, key = lambda ckpt: int(ckpt.split("-")[-1]))
print("all available checkpoints:")
print(" id: checkpoint name")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

""" You may want (but not necessarily need) to change the check point """

id_of_ckpt_to_use = 2  # 要用來進行推理的checkpoint的id(對應上一個cell的輸出結果)
                        # 預設值-1指的是上列checkpoints中的"倒數"第一個，也就是最後一個checkpoint
                        # 如果想要選擇其他checkpoint，可以把-1改成有列出的checkpoint id中的其中一個

ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])

""" You may want (but not necessarily need) to change decoding parameters """
# 你可以在這裡調整decoding parameter，decoding parameter的詳細解釋請見homework slides
max_len = 512   # 生成回復的最大長度
temperature = 0.5  # 設定生成回覆的隨機度，值越小生成的回覆越穩定
top_p = 0.3  # Top-p (nucleus) 抽樣的機率閾值，用於控制生成回覆的多樣性
# top_k = 5 # 調整Top-k值，以增加生成回覆的多樣性和避免生成重複的詞彙

"""The following code block takes about **2** minutes to run if you use the default setting, but it may vary depending on the condition of Colab."""

""" It is recommmended NOT to change codes in this cell """

test_data_path = "./project_data_main/Tang_testing_data.json"
output_path = os.path.join(output_dir, "results.txt")

cache_dir = "./cache"  # 設定快取目錄路徑
seed = 42  # 設定隨機種子，用於重現結果
no_repeat_ngram_size = 3  # 設定禁止重複 Ngram 的大小，用於避免生成重複片段

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 使用 tokenizer 將模型名稱轉換成模型可讀的數字表示形式
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# 從預訓練模型載入模型並設定為 8 位整數 (INT8) 模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},  # 設定使用的設備，此處指定為 GPU 0
    cache_dir=cache_dir
)

# 從指定的 checkpoint 載入模型權重
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

"""The following code block takes about **4** minutes to run if you use the default setting, but it may vary depending on the condition of Colab."""

""" It is recommmended NOT to change codes in this cell """

results = []

# 設定生成配置，包括隨機度、束搜索等相關參數
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2
)

# 讀取測試資料
with open(test_data_path, "r", encoding = "utf-8") as f:
    test_datas = json.load(f)

# 對於每個測試資料進行預測，並存下結果
with open(output_path, "w", encoding = "utf-8") as f:
  for (i, test_data) in enumerate(test_datas):
      predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose = False)
      f.write(f"{i+1}. "+test_data["input"]+predict+"\n")
      print(f"{i+1}. "+test_data["input"]+predict)

"""## **IMPORTANT**: Submit above 15 poems to DaVinci Assistant.
You can find these poems in your "results.txt". The grading of this homework will be based on the evaluation results of DaVinci Assistant on these poems.

## See how the fine-tune model do compared to model without fine-tuning

We now check what our model can do on the same examples we saw in the "Inference before Fine-tuning" section above.

The following code block takes about **40** seconds to run if you use the default setting, but it may vary depending on the condition of Colab.
"""

# using the same demo examples as before
test_tang_list = ['相見時難別亦難，東風無力百花殘。', '重帷深下莫愁堂，臥後清宵細細長。', '芳辰追逸趣，禁苑信多奇。']

# inference our fine-tuned model
demo_after_finetune = []
for tang in test_tang_list:
  demo_after_finetune.append(f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n'+evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose = False))

# print and store the output to text file
for idx in range(len(demo_after_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_after_finetune[idx])
  print("-" * 80)


"""## Reference

[Tang Poem Dataset](https://github.com/chinese-poetry/chinese-poetry/tree/master/%E5%85%A8%E5%94%90%E8%AF%97?fbclid=IwAR2bM14S42T-VtrvMi3wywCqKfYJraBtMl7QVTo0qyPMjX9jj9Vj3JepFBA)
"""