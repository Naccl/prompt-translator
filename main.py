import re
import os
import string
import csv
import time
import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# The directory to store the models
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_name = 'Helsinki-NLP/opus-mt-zh-en'
# 自定义翻译映射表，用于将prompt常用词语翻译为更合适的英文结果
custom_csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom.csv')
# batch_size太大可能反而会变慢，测试16比较合适，如果显存不够，可以适当减小
dataloader_batch_size = 16
# 对于prompt来说，通常是单词或短句，token长度设为32比较合适，速度和翻译结果都比较好，但如果输入是长句子，就会出现翻译一半漏掉关键字的情况
model_max_length = 32


class ZhEnTranslatorDataset(Dataset):
    def __init__(self, zh_str_arr, tokenizer, device):
        self.zh_str_arr = zh_str_arr
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.zh_str_arr)

    def __getitem__(self, idx):
        zh_str = self.zh_str_arr[idx]
        encoded_input = self.tokenizer.encode_plus(zh_str, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input['input_ids'].squeeze(0).to(self.device)
        attention_mask = encoded_input['attention_mask'].squeeze(0).to(self.device)
        return input_ids, attention_mask


def collate_fn(batch):
    input_ids_batch, attention_mask_batch = zip(*batch)
    input_ids_padded = pad_sequence(input_ids_batch, batch_first=True)
    attention_mask_padded = pad_sequence(attention_mask_batch, batch_first=True)
    return input_ids_padded, attention_mask_padded


class ZhEnTranslator:
    def __init__(self, cache_dir=cache_dir, model_name=model_name):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"Running model on: {'GPU' if str(self.device) == 'cuda:0' else 'CPU'}")

    def translate(self, zh_str_arr: list) -> list:
        # 将中文字符串数组转为dataset加快翻译速度
        dataset = ZhEnTranslatorDataset(zh_str_arr, self.tokenizer, self.device)
        dataloader = DataLoader(dataset, batch_size=dataloader_batch_size, shuffle=False, collate_fn=collate_fn)

        en_str_arr = []

        for batch_input_ids, batch_attention_mask in dataloader:
            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)
            translations = self.model.generate(batch_input_ids, attention_mask=batch_attention_mask, num_beams=4, max_length=model_max_length)

            for translation in translations:
                en_str = self.tokenizer.decode(translation, skip_special_tokens=True)
                en_str_arr.append(en_str)
        return en_str_arr


class Processor():
    def __init__(self) -> None:
        self.translator = ZhEnTranslator()
        # 加载自定义翻译映射表
        self.custom_cache = self.load_csv(custom_csv_file)

    def load_csv(self, csv_file) -> dict:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            cache = dict(reader)
        return cache

    def process(self, zh_text: str) -> str:
        """翻译入口函数"""
        pre_text_array = self.pre_process_text(zh_text)
        translated_text_array = self.translate(pre_text_array)
        return ','.join(translated_text_array)

    def pre_process_text(self, text: str) -> list:
        """预处理文本"""
        # 将中文全角标点符号替换为半角标点符号
        text = text.translate(str.maketrans('，。！？；：‘’“”（）【】', ',.!?;:\'\'\"\"()[]'))
        # 按逗号分割成数组
        text_array = text.split(',')
        # 预处理结果数组
        pre_text_array = []
        # 对数组中每个字符串进行处理
        for i in range(len(text_array)):
            # 如果字符串以 < 开头 > 结尾，则是Lora，跳过不处理
            if text_array[i].startswith('<') and text_array[i].endswith('>'):
                pre_text_array.append(text_array[i])
            # 判断是否只包含英文字符
            if all(char in string.printable for char in text_array[i]):
                pre_text_array.append(text_array[i])
            # 判断是否有中文
            if re.search('[\u4e00-\u9fff]', text_array[i]):
                # 切分字符串
                split = self.split_string(text_array[i])
                # 将切分后的三部分存入预处理结果数组，中间部分为待翻译的文本
                pre_text = [split[0], split[1], split[2]]
                pre_text_array.append(pre_text)

        return pre_text_array

    def split_string(self, text: str) -> tuple:
        """切分字符串，第一个汉字、数字、字母和最后一个汉字、数字、字母为分界点，切分成三部分，将特殊字符和待翻译文本分隔开，便于更精准匹配映射表"""
        # 查找第一个汉字、数字或字母的索引
        first_index = re.search(r'[\u4e00-\u9fff0-9a-zA-Z]', text).start()

        # 查找最后一个汉字、数字或字母的索引
        last_index = max([i for i, c in enumerate(text) if re.match(r'[\u4e00-\u9fff0-9a-zA-Z]', c)])

        # 切分字符串
        part1 = text[:first_index]
        part2 = text[first_index:last_index + 1]
        part3 = text[last_index + 1:]
        return part1, part2, part3

    def translate(self, pre_text_array: list) -> list:
        """自定义翻译映射 + 模型翻译"""
        # 收集待模型翻译的文本
        pre_translate_array = []
        for i in range(len(pre_text_array)):
            # 如果元素不是数组，则是不需要翻译的文本，留在pre_text_array中
            if isinstance(pre_text_array[i], list):
                # 判断能否被自定义翻译
                custom_res = self.custom_translate(pre_text_array[i][1])
                if custom_res is not None:
                    # 能被自定义翻译，直接替换
                    pre_text_array[i] = pre_text_array[i][0] + custom_res + pre_text_array[i][2]
                else:
                    # 不能被自定义翻译，加入待翻译数组
                    pre_translate_array.append(pre_text_array[i][1])

        # 调用模型翻译
        en_str_array = self.translator.translate(pre_translate_array)

        # 删除翻译结果中的句号
        for i in range(len(en_str_array)):
            en_str_array[i] = self.remove_trailing_dot(en_str_array[i])

        for i in range(len(pre_text_array)):
            if isinstance(pre_text_array[i], list):
                pre_text_array[i] = pre_text_array[i][0] + en_str_array.pop(0) + pre_text_array[i][2]

        return pre_text_array

    def custom_translate(self, text: str) -> str:
        """自定义翻译映射"""
        if text in self.custom_cache:
            return self.custom_cache[text]
        else:
            return None

    def remove_trailing_dot(self, text: str) -> str:
        """Removes a trailing dot from the text"""
        if text.endswith("."):
            return text[:-1]
        return text


if __name__ == "__main__":
    processor = Processor()
    while True:
        zh_text = input("input: ")
        start_time = time.time()
        translated_text = processor.process(zh_text)
        end_time = time.time()
        trans_log = f'''
==================================================
{zh_text}
================= elapsed: {round(end_time - start_time, 2)} ==================
{translated_text}
==================================================
'''
        print(trans_log)
