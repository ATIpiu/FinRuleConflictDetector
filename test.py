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
    # prompt = f"""
    #     阅读以下长文本，识别并判断其中是否存在错误。错误类型包括：常识错误、数值单位错误、逻辑矛盾、时间矛盾、数值前后矛盾、数据不完整、计算错误、语句重复。请根据这些错误类型，判断给定的句子是否包含错误，并说明原因。
    #
    #     示例：
    #
    #     常识错误:
    #     七、合同签订日期:2022-07-32
    #     返回: 是
    #     原因：7月没有32号
    #
    #     数值单位错误:
    #     合同金额:40.0000000 元
    #     返回: 是
    #     原因：金额单位应为万元而不是元
    #
    #     逻辑矛盾:
    #     第五条 保险财产的下列损失本公司也负责赔偿:e(1)被保险人自有的供电、供水、供气设备因第四条所列灾害或事故遭受损害，引起停电、水、停气以致直接造成保险财产的损失;“(
    #     2)在发生第四条所列灾害或事故时，为了抢救财产或防止灾害蔓延，采取合理的、必要的措施而造成保险财产的损失。e(3)核子辐射或污染:e第六条
    #     发生保险事故时，为了减少保险财产损失，被保险入对保险财产采取施救、保护、整理措施而支出的合理费用，由本公司负责赔偿。-除外责任 由于下列原因造成保险财产的损失，本公司不负责赔偿:(1)战争、军事行动或暴乱:“(
    #     2)核子辐射或污染:(3)被保险人的故意行为。e第八条本公司对下列损失也不负责赔偿
    #     返回: 是
    #     原因：前后矛盾，先说负责赔偿核子辐射或污染，后又说不负责赔偿
    #
    #     时间矛盾:
    #     投标文件递交截止时间 2024-05-23 09:00:00(北京时间，24小时制)
    #     开标时间: 2024-05-22 09:00:00(北京时间，24小时制)
    #     返回: 是
    #     原因：开标时间早于投标截止时间
    #
    #     数值前后矛盾:
    #     在恒生12个一级子行业中，医疗保健行业周涨跌幅排名为第14位。
    #     返回: 是
    #     原因：12个一级子行业中不可能有第14位
    #
    #     数据不完整:
    #     此次接续采购周期为年。
    #     返回: 是
    #     原因：缺少具体年份
    #
    #     计算错误:
    #     髋关节产品系统年度采购需求量285995个(陶瓷-陶瓷类髋关节102264个、陶瓷聚乙烯类173303个、合金聚乙烯类1042个)。
    #     返回: 是
    #     原因：各类髋关节产品总和与年度需求量不符
    #
    #     语句重复:
    #     部分产品复活价格线价格高于集采中标价格，部分产品复活价格线价格高于集采中标价格。
    #     返回: 是
    #     原因：语句重复
    #
    #     请根据以上示例判断输入的句子是否包含错误，并回答“是”或“否”，并说明原因。
    #
    #     输入句子: {sentence}
    #     """
    prompt = (
        "阅读以下长文本，识别并判断其中是否存在错误。错误类型包括："
        "常识错误、数值单位错误、逻辑矛盾、时间矛盾、数值前后矛盾、数据不完整、计算错误、语句重复等。"
        "请根据这些错误类型，逻辑模糊以及不完整不算逻辑错误。无需检测格式错误，语法错误，判断给定的句子是否包含错误，并非常简要地说明原因。"
        "请先回答是或者否，并返回以逗号分割的最小错误子句因包含全部错误子句，错误子句应该和源文本中完全相同，只检测上诉错误不。"
        "返回示例，是@错误子句@原因。"
        f"输入句子: {sentence}\n"
    )
    history = []
    max_length = 8192
    top_p = 0.8
    temperature = 0.6
    stop = StopOnTokens()

    history.append([prompt, ""])

    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

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
        # "top_p": top_p,
        # "temperature": temperature,
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

def read_docx(filename, path):
    doc = docx.Document(os.path.join(path, filename))
    dic = {}
    dic["id"] = filename[:-5]
    dic["sents"] = []
    non_error_sentences = []

    paragraphs = doc.paragraphs
    total_paragraphs = len(paragraphs)

    for idx, paragraph in enumerate(paragraphs):
        text = paragraph.text
        if len(text) < 12:
            continue
        if re.match(r'^\d', text):
            text = text[4:]  # 只取第5个字符之后的内容

        # 获取上下文
        context = []
        if idx > 1:
            context.extend([paragraphs[idx - 2].text, paragraphs[idx - 1].text])
        else:
            context.extend([paragraphs[i].text for i in range(idx)])

        if idx < total_paragraphs - 2:
            context.extend([paragraphs[idx + 1].text, paragraphs[idx + 2].text])
        else:
            context.extend([paragraphs[i].text for i in range(idx + 1, total_paragraphs)])

        context_text = " ".join(context).strip()
        full_sentence = f" {context_text}{text}".strip()
        response = check_sentence_for_errors(full_sentence )

        if "是" in response[0]:
            print("++++=" * 30)
            print(text, response)
            if len(response) == 1:
                continue

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
    with open('result/test_data.json', 'a', encoding='utf-8') as f:
        f.write(json_str + '\n')

    # print("处理完成并保存到 test_data.json")


def process_files(path):
    result = os.listdir(path)
    for data in tqdm(result[:]):

        if 'docx' in data:
            read_docx(data, path)

if __name__ == "__main__":
    data_path = './round1_test_data'
    file_path = "result/test_data.json"

    # 尝试删除文件，如果文件存在则删除
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} 已删除")

    # 创建一个新的空文件
    with open(file_path, 'w') as file:
        file.write('')
        print(f"{file_path} 已创建")
    process_files(data_path)
