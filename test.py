import os
import json
import re
from tqdm import tqdm
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from threading import Thread
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import docx

# 下载 punkt 数据包
nltk.download('punkt')
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Initialize the tokenizer and model
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def check_sentence_for_errors(sentence):
    prompt = (
        "阅读以下长文本，识别并判断其中是否存在错误。错误类型包括："
        "常识错误、数值单位错误、逻辑矛盾、时间矛盾、数值前后矛盾、数据不完整、计算错误、语句重复。"
        "请根据这些错误类型，逻辑模糊以及不完整不算逻辑错误。无需检测格式错误，语法错误，判断给定的句子是否包含错误，并非常简要地说明原因。"
        "请先回答是或者否，并返回以逗号分割的最小错误子句因包含全部错误子句，错误子句应该在输入文本中完全相同，只检测上诉错误不。"
        "返回示例，是@错误子句@错误类别@错误原因。"
        f"输入句子: {sentence}\n"
    )
    history = []
    max_length = 8192
    top_p = 0.8
    temperature = 0.6
    stop = StopOnTokens()

    history.append([prompt, ""])

    messages = []
    messages.append({"role": "user", "content": prompt})

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": False,
        "temperature": 0,
        "top_p": 0,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token

    # print(response.split("@"))
    # 尝试将响应解析为JSON

    return response.split("@")


def check_logic(sentence, context):
    prompt = (
        "请结合上文判断当前句子是否有明显的逻辑错误，只回答是或者否"
        f"上文: {context}\n"

        f"输入句子: {sentence}\n"
    )
    max_length = 8192
    stop = StopOnTokens()
    messages = []
    messages.append({"role": "user", "content": prompt})

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": False,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.9,
        "eos_token_id": model.config.eos_token_id,
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token

    # print(response.split("@"))
    # 尝试将响应解析为JSON

    return True if "是" in response[0:4] else False


def read_docx(filename, path):
    doc = docx.Document(os.path.join(path, filename))
    dic = {}
    dic["id"] = filename[:-5]
    dic["sents"] = []
    accumulated_text = ""
    paragraphs = doc.paragraphs
    for i, paragraph in tqdm(enumerate(paragraphs)):
        text = paragraph.text.strip()

        # Skip empty paragraphs
        if not text:
            continue

        # Accumulate short paragraphs
        if len(text) < 12:
            accumulated_text += " " + text
            continue

        # If there is accumulated text, prepend it to the current paragraph
        if accumulated_text:
            text = accumulated_text.strip() + " " + text
            accumulated_text = ""

        # Check the combined paragraph length
        if len(text) < 20:
            accumulated_text = text
            continue

        if re.match(r'^\d', text):
            text = text[6:]  # Only take the content after the 5th character

        full_sentence = f" {text}".strip()
        response = check_sentence_for_errors(full_sentence)

        if "是" in response[0]:
            # print("++++++++++++++++++++++++++++++++++++++++"*3)
            # print(text, response)

            if len(response) < 3:
                continue
            logic = True
            if response[2] != "语句重复" and response[2] != "数值单位错误":

                if i > 5:
                    logic = check_logic(full_sentence, paragraphs[i - 5:i])
                else:
                    logic = check_logic(full_sentence, paragraphs[0:i])
                # print("======" * 30)
                # print(text, response)
                # print("logic:",logic)

            if logic:
                print("=====================" * 30)
                print(paragraph.text, '\n', response)
                parts = re.split(',', response[1])
                for err_sent in parts:
                    dic["sents"].append(err_sent)

        # sentences = sent_tokenize(paragraph.text)  # 将段落分割成句子
        # for sentence in sentences:
        #     full_sentence = f" {sentence}".strip()
        #     response = check_sentence_for_errors(full_sentence)
        #
        #     if "是" in response[0]:
        #         print("++++=" * 30)
        #         print(sentence, response)
        #         parts = re.split(',', response[1])
        #         for err_sent in  parts:
        #             dic["sents"].append( err_sent)
        #     else:
        #         non_error_sentences.append(sentence)

    # 打开文件以追加方式写入处理结果
    json_str = json.dumps(dic, ensure_ascii=False)
    with open('1.json', 'a', encoding='utf-8') as f:
        f.write(json_str + '\n')

    # print("处理完成并保存到 1.json")


def process_files(path):
    result = os.listdir(path)
    for data in tqdm(result[:]):

        if 'docx' in data:
            read_docx(data, path)


if __name__ == "__main__":
    data_path = './round1_test_data'
    file_path = "1.json"

    # 尝试删除文件，如果文件存在则删除
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} 已删除")

    # 创建一个新的空文件
    with open(file_path, 'w') as file:
        file.write('')
        print(f"{file_path} 已创建")
    process_files(data_path)
