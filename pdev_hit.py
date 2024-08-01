from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import getpass
from pprint import pprint
import json
import csv

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Your API key here")
llm = ChatOpenAI(
    model = "gpt-4o", 
    temperature = 0.3,
    max_tokens = 1024
)

"""
Read each lines in the the jsonlines file and assign it to file_input.
"""
prompts = []
file_input = './dataset/hit/hit_over_18_gpt.json'
file_output = './dataset/hit/answer_hit_over_18_gpt.csv'
with open(file_input) as file:
    for line in file:
        prompts.append(json.loads(line))

"""
Pattern
"""
pattern = """
[[human 1 | animate obj]] hit [[human 2 | phys obj]]
[[vehicle | human 1 = driver]] hit [[phys obj | human 2 = casualty]]
[[projectile | human 1 = shooter]] hit [[phys obj = target | human 2 = target | victim]]
[[human = ball player]] hit [[phys obj = ball]] [adv [direction]]
[[human = ball player]] hit [[phys obj]] [with ([[artifact = ball]])]
[[human = ball player]] hit [[numerical value]]
[[eventuality | abstract entity]] hit [[numerical value]] | {target | objective}
[[inanimate]] hit [[phys obj]]
[[event = bad]] hit ([[activity]]) | ([[institution]]) | ([[human]]) | ([[location]]) []
idiom [[human]] hit {winning streak | form}
[[human]] hit {head}
[[human 1]] hit [at, back [[human 2]]]
[[human 1]] hit ([[human 2]])[back]
[[human 1]] hit [out, at ([[phys obj]]) | [[human 2]] | [[]]]
[[human 1]] hit [out, at  ([[proposition]] | [[human 2]] | [[]]]
[[human]] hit [on | upon [[concept]]]
[[eventuality = bad]] hit [home]
idiom [[anything]] hit {the deadlines}
idiom [[human]] hit {[the deck) | (the ground)}
idiom [[human]] hit {the road}
[[human | event]] hit {[MOD] note}
[[human 1]] hit [[human 2]] [for [[entity = valued]]]
[[concept | event = experience]] | {it} hit [[human]]
idiom [[human | institution]] hit {stride}
idiom [[human 1]] hit ([[human 2]]) [below {belt}]
idiom [[human | institution | activity]] hit {brick wall | buffers}
[[human | institution]] hit [[time_period = bad]]
idiom [[human]] hit {brake}
phrasal [[human 1]] hit {it} [off, with [[human 2]] | ([[]])]
idiom {jaw} hit {floor | carpet}
idiom [[human]] hit {jackpot}
idiom [[human]] hit {the roof}
idiom [[artifact = product]] hit {market | shops | stores | streets | news stands | screen}
idiom [[human]] hit {nail} [on {the head}]
[[human | institution | action]] hit {mark | target}
idiom [[human]] hit {bottle}
[[human = recording artist]] hit [[[-]] {big}]
idiom [[inanimate = food]] hit {the spot}
idiom {head} hit {the pillow}
idiom [[human]] hit {the block}
"""

"""
Example
"""
example = """
Are you hitting me? = [[human 1 | animate obj]] hit [[human 2 | phys obj]]
You hit your sister. = [[human 1 | animate obj]] hit [[human 2 | phys obj]]
You'd think that a bomb had hit this house. = [[projectile | human 1 = shooter]] hit [[phys obj = target | human 2 = target | victim]]
You have to hit them back again. = [[human 1]] hit ([[human 2]])[back]
Did you hit your head? = [[human]] hit [head]
"""

"""
Prompt Template
"""
prompt_template = """
    You are going to annotate some data according to Pattern Dictionary of English Verbs by Patrick Hanks. The verb is \"hit\" and the list of the pattern for the verb are found below:
 
    [[human 1 | animate obj]] hit [[human 2 | phys obj]]
    [[vehicle | human 1 = driver]] hit [[phys obj | human 2 = casualty]]
    [[projectile | human 1 = shooter]] hit [[phys obj = target | human 2 = target | victim]]
    [[human = ball player]] hit [[phys obj = ball]] [adv [direction]]
    [[human = ball player]] hit [[phys obj]] [with ([[artifact = ball]])]
    [[human = ball player]] hit [[numerical value]]
    [[eventuality | abstract entity]] hit [[numerical value]] | [target | objective]
    [[inanimate]] hit [[phys obj]]
    [[event = bad]] hit ([[activity]]) | ([[institution]]) | ([[human]]) | ([[location]]) []
    idiom [[human]] hit [winning streak | form]
    [[human]] hit [head]
    [[human 1]] hit [at, back [[human 2]]]
    [[human 1]] hit ([[human 2]])[back]
    [[human 1]] hit [out, at ([[phys obj]]) | [[human 2]] | [[]]]
    [[human 1]] hit [out, at  ([[proposition]] | [[human 2]] | [[]]]
    [[human]] hit [on | upon [[concept]]]
    [[eventuality = bad]] hit [home]
    idiom [[anything]] hit [the deadlines]
    idiom [[human]] hit [[the deck) | (the ground)]
    idiom [[human]] hit [the road]
    [[human | event]] hit [[MOD] note]
    [[human 1]] hit [[human 2]] [for [[entity = valued]]]
    [[concept | event = experience]] | [it] hit [[human]]
    idiom [[human | institution]] hit [stride]
    idiom [[human 1]] hit ([[human 2]]) [below [belt]]
    idiom [[human | institution | activity]] hit [brick wall | buffers]
    [[human | institution]] hit [[time_period = bad]]
    idiom [[human]] hit [brake]
    phrasal [[human 1]] hit [it] [off, with [[human 2]] | ([[]])]
    idiom [jaw] hit [floor | carpet]
    idiom [[human]] hit [jackpot]
    idiom [[human]] hit [the roof]
    idiom [[artifact = product]] hit [market | shops | stores | streets | news stands | screen]
    idiom [[human]] hit [nail] [on [the head]]
    [[human | institution | action]] hit [mark | target]
    idiom [[human]] hit [bottle]
    [[human = recording artist]] hit [[[-]] [big]]
    idiom [[inanimate = food]] hit [the spot]
    idiom [head] hit [the pillow]
    idiom [[human]] hit [the block]

    Follow the examples of the annotation below:
    Are you hitting me? = [[human 1 | animate obj]] hit [[human 2 | phys obj]]
    You hit your sister. = [[human 1 | animate obj]] hit [[human 2 | phys obj]]
    You'd think that a bomb had hit this house. = [[projectile | human 1 = shooter]] hit [[phys obj = target | human 2 = target | victim]]
    You have to hit them back again. = [[human 1]] hit ([[human 2]])[back]
    Did you hit your head? = [[human]] hit [head]
    
    Annotate the data below and keep it short:
    {utterance}
"""

"""
Chain
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
runnable = prompt | llm | StrOutputParser()

for i, prompt in enumerate(prompts):
    print(f"Answering question {i+1}: {prompt['utterance']}")

"""
Structured Output
"""
response = runnable.batch(prompts)

with open(file_output, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Response'])
    writer.writerows([[item] for item in response])
print("Output saved to", file_output)