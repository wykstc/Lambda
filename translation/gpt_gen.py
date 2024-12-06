from openai import OpenAI
import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os
from tqdm import tqdm
import re
import time
import json
import random
import pandas as pd
import pickle

client = OpenAI(api_key='YOUR_API_KEY')

EN = open("augEN.pkl", 'rb')
ENInfo = pickle.load(EN)

df_ids = pd.read_csv('orgENDE.csv',sep='\t', header=None, names=['ID','EN','DE'])




def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


parser = argparse.ArgumentParser("")
parser.add_argument("--temperature", default=0.2, type=float, help="which seed to use")
parser.add_argument("--top_p", default=0.5, type=float, help="top_p for sampling")
parser.add_argument("--n_sample", default=6000, type=int, help="number of examples to be generated")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--max_tokens", default=512, type=int, help="which seed to use")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

args = parser.parse_args()


async def dispatch_OpenAI_requests(
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return async_responses

prompt_template = ("Firstly, refer to the ground truth translation of English sentence: {}, and its corresponding German translation: {}."
                   "Then, generate corresponding translation in German of sentence: {}.")
def main(args):
    i = 0
    for idx, row in df_ids.iterrows():
        i = i + 1
        print(i)
        id = row['ID']
        ENORGtxt = row['EN']
        FRORGtxt = row['DE']
        ENAUG = ENInfo[str(id)]

        prompt_input = prompt_template.format(ENORGtxt,FRORGtxt,ENAUG)

        try:
            response = asyncio.run(
                dispatch_OpenAI_requests(
                    messages_list=[
                        [{"role": "user", "content": prompt_input}],
                    ],
                    model="gpt-3.5-turbo",
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                )
            )
        except openai.RateLimitError:
            print(f"RateLimitError for class {i}.")
            time.sleep(10)
            continue
        except openai.APIError:
            print("APIError for class {i}.")
            time.sleep(10)
            continue

        ans = [completion.choices[0].message.content for completion in response]
        ans_str = '\n'.join(ans) + '\n\n'

        with open('enrichedSentenceGermanREF.txt', 'a', encoding='utf-8') as f:
            f.write(str(id) + ': ' + ans_str + '\n')

if __name__ == '__main__':
    main(args)

