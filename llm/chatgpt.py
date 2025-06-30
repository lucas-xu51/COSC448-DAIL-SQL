import json.decoder
import time
from openai import OpenAI, RateLimitError, OpenAIError
from utils.enums import LLM

# global variable to hold the OpenAI client
client = None

def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    global client
    client = OpenAI(api_key=OPENAI_API_KEY)  # read from environment variable


def ask_completion(model, batch, temperature):
    response = client.completions.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response.choices]
    return dict(
        response=response_clean,
        **response.usage.model_dump()
    )


def ask_chat(model, messages: list, temperature, n):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=200,
        n=n
    )
    response_clean = [choice.message.content for choice in response.choices]
    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        **response.usage.model_dump()
    )


def ask_llm(model: str, batch: list, temperature: float, n: int):
    n_repeat = 0
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                assert n == 1
                response = ask_completion(model, batch, temperature)
            elif model in LLM.TASK_CHAT:
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]}]
                response = ask_chat(model, messages, temperature, n)
                response['response'] = [response['response']]
            break
        except RateLimitError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for RateLimitError")
            time.sleep(1)
        except json.decoder.JSONDecodeError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError")
            time.sleep(1)
        except OpenAIError as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for OpenAIError: {e}")
            time.sleep(1)
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for Exception: {e}")
            time.sleep(1)

    return response
