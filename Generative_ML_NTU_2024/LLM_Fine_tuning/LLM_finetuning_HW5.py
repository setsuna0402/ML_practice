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
import time
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
# Load the pretrained LLM model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage = True
)

# Create tokenizer and set eos_token (end of sentence token)
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

# decoding parameters for influence
max_len = 256 # Max length of output
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1, # low temperature means the output is more stable, hight temperatue means it is more divergent
    num_beams=1, # Beam search width 
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)

# demo examples
test_tang_list = ['相見時難別亦難，東風無力百花殘。', '重帷深下莫愁堂，臥後清宵細細長。', '芳辰追逸趣，禁苑信多奇。']

# get the model output for each examples
demo_before_finetune = []
for tang in test_tang_list:
  demo_before_finetune.append(f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n'+evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose = False))

# print and store the output to text file
for idx in range(len(demo_before_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_before_finetune[idx])
  print("-" * 80)

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

# 根據 from_ckpt 標誌，從 checkpoint 載入模型權重
if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

# 將模型準備好以使用 INT8 訓練
# model = prepare_model_for_int8_training(model) # tis depicated, use prepare_model_for_kbit_training instead
# Ensure the model is ready for k-bit training (e.g., 4-bit or 8-bit quantization)
# It upcasts the weights to fp16 and makes other necessary adjustments for training with quantized weights.
model = prepare_model_for_kbit_training(model)

# Use LoraConfig to configure LORA model and get_peft_model to get the LORA model
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# set pad_token_id to 0 for tokenizer
tokenizer.pad_token_id = 0

# Load the training data 
with open(dataset_dir, "r", encoding = "utf-8") as f:
    data_json = json.load(f)
with open("tmp_dataset.json", "w", encoding = "utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent = 2, ensure_ascii = False)

data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")

# 將訓練數據分為訓練集和驗證集（若 VAL_SET_SIZE 大於 0）
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None

# 使用 Transformers Trainer 進行模型訓練
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,  # 使用混合精度訓練
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,  # 是否使用 DDP，控制梯度更新策略
        report_to=report_to,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 禁用模型的 cache 功能
model.config.use_cache = False

# 若使用 PyTorch 2.0 版本以上且非 Windows 系統，進行模型編譯
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

start_time = time.time()  # Record the start time of training
# 開始模型訓練
trainer.train()
end_time = time.time()  # Record the end time of training
# 將訓練完的模型保存到指定的目錄中
model.save_pretrained(ckpt_dir)

# 印出訓練過程中可能的缺失權重的警告信息
print("\n If there's a warning about missing keys above, please disregard :)")

print("Fine tuning is done!")
print("Total training time: {:.2f} minutes".format((end_time - start_time) / 60))
exit()
