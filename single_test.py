import os
import torch
from threading import Thread
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel

LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "local_model")
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto").eval()


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def check_sentence_for_errors(prompt, sentence,context):
    prompt = (
        "阅读以下长文本，识别并判断其中是否存在错误。错误类型包括："
        "常识错误、数值单位错误、逻辑矛盾、时间矛盾、数值前后矛盾、数据不完整、计算错误、语句重复等。"
        "请根据这些错误类型，逻辑模糊以及不完整不算逻辑错误。判断给定的句子是否包含错误，并非常简要地说明原因。"
        "请先回答是或者否，并返回以逗号分割的最小错误子句因包含全部错误子句,错误子句应该和源文本中完全相同。书写错误也算为逻辑错误。"
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

    response = response.strip()
    return response.split("@")


if __name__ == "__main__":
    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    last_prompt = ""
    while True:
        # prompt = input("请输入 prompt: ")
        # if prompt:
        #     last_prompt = prompt
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        context=input("context")
        result = check_sentence_for_errors("", user_input,context)
        print(f"GLM-4: {result}")
# 请帮我检查我的句子是否存在逻辑漏洞，回答我是或者否，并简明解释原因
# 你现在需要帮我检测文本中的逻辑错误,方便我进行修改。阅读以下长文本，识别并判断其中是否存在错误。错误类型包括：常识错误、数值单位错误、逻辑矛盾、时间矛盾、数值前后矛盾、数据不完整、计算错误、语句重复等。请根据这些错误类型，判断给定的句子是否包含错误，并说明原因。请先回答是或者否，书写错误也算为逻辑错误
