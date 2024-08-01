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
file_input = './dataset/break/break_over_18_gpt.json'
file_output = './dataset/break/answer_break_over_18_gpt.csv'
with open(file_input) as file:
    for line in file:
        prompts.append(json.loads(line))

"""
Prompt Template
"""
prompt_template = """
    You are going to annotate some data according to Pattern Dictionary of English Verbs by Maarouf and Baisa (2013). The verb is \"break\" and the list of the pattern for the verb are found below:
 
    [[human | animate | event]] break [[artifact | phys obj]]
    [[artifact | phys obj]] break
    [[human]] break [[rule | agreement]]
    [[human | institution | event]] break [[state of affairs]]
    [[human | animal ]] break [[body part = bone / tooth]]
    [[human 1 | animal 1 | event]] break [[body part == bone / tooth]]
    [[body part = bone / tooth]] break
    phrasal [[vehicle | device | human = driver | operator]] break [down]
    phrasal [[human]] break [down]
    phrasal [[activity = collaborative | goal-oriented | state of affairs | relationships]] break [down]
    phrasal [[human | animal | eventuality]] breal [[phys obj = barrier]] [down]
    phrasal [[institution | action | human]] break [[abstract entity = obstacle]] [down]
    phrasal [[phys obj 1 | stuff 1 | body | process]] break [[phys obj 2 | stuff 2]] [[down, into [[phys obj 3 = constituent parts]]]
    phrasal [[phs obj 1 | stuff]] break [down, into [[phys obj 2 = plural | multiple]]]
    phrasal [[human]] break [[abstract entity 1 = complex predicate]] [[into, down | up [[abstract entity 2 = component (plural)]]]
    phrasal [[anything = complex predicate]] break [into, down | up [[anything 2 = component (plural | multiple)]]]
    phrasal [[entity]] break [apart]
    phrasal [[human | eventuality | animal]] break [[entity]] [apart]
    phrasal [[phys obj part]] break [[from, away [[phys obj]]]
    phrasal [[human 1 | human group 1]] break [from, away [[human group 2 | human 2 = leader | concept]]]
    phrasal [[human 1 | human group 1 | vehicle]] break [awal | off, from [[human 2 | human group 2]]]
    phrasal [[human = burglar]] break [into, in [[building | road vehicle]]]
    phrasal [[human]] break [in]
    phrasal [[human]] break [[horse]] [in]
    phrasal [[human | animal]] break [[phys obj]] [off]
    phrasal [[phys obj part]] break [off]
    phrasal [[human | institution]] break [[activity = collaborative | goal-oriented | relationship | state of affairs]] [off]
    phrasal [[human | document]] break [off]
    [[human 1]] break [news] [to [[human 2]]]
    [news | scandal | story] break
    idiom [[human]] break [the ice]
    idiom [[human]] break [the mould]
    idiom [[human | eventuality]] break [record]
    idiom [[eventuality | human 1]] break [[[human 2]]'s heart]
    [wave | sea] break [over | against [[phys obj]]]
    [[human | institution]] break [free | loose] [from [[eventuality = constraint]]]
    idiom [(all) hell | pandemonium] break [loose]
    idiom [[human 1 | institution 1]] break [ranks] [with [[human 2 | institution 2]]]
    [[human | activity]] break [even]
    [dawn | day] break
    phrasal [[eventuality = bad]] break [out]
    phrasal [[human | institution | animal]] break [out, of [[location = bad | eventuality = bad]]]
    [[human]] break [into, out in | into [a rash | a fever | a sweat ]]
    phrasal [[human]] break [[alcoholic drink]] [out]
    phrasal [[human group = military force | violent | human]] break [through [[phys obj = obstacle]]]
    phrasal [human | institution]] break [through [[abstract entity = obstacle]]]
    phrasal [[light]] break [through]
    phrasal [[watercourse]] | [lava] break [through [[land]]]
    phrasal [[phys obj | stuff]] break [up]
    phrasal [[human | eventuality]] break [[phys obj]] [up]
    phrasal [[eventuality]] break [up]
    phrasal [[human | event 1]] break [[event 2]] [up]
    phrasal [[human | institution 1 | eventuality]] break [[institution 2 | state of affairs]] [up]
    phrasal [[institution | state of affairs]] break [up]
    idiom [[human | institution]] break [new ground]
    phrasal [[human | eventuality]] break [bond | link | circuit | chain]
    [[human | institution | eventuality]] break [[abstract entity = connection]] [between, with [[anything | anything 1 (and) anything 2]]]
    phrasal [[human = couple]] break [up]
    [[human | institution]] break [code]
    [[human | institution]] break [with [[activity = habitual | traditional | concept = widely accepted]]]
    [[human 1 | institution 1]] break [with [[human 2 | institution 2]]]
    [voice] break
    [[human]] break [into [[speech sound]] | [song | giggles | tears]]
    [[human | institution]] break [into [[activity = commercial | desirable]]]
    [[human]] break [habit | pattern | cycle]
    [[human | eventuality]] break [[anything = monotonous | continuous]] [up]
    phrasal [[human group = schoolchildren | institution = school]] break [up, for [[time period = vacation]]]
    [[eventuality | human 1]] break [[human 2]] | [spirit | will]
    [[phys obj]] break [fall]
    [[human group]] break [for [[food | beverage | time period]]]
    [weather] break
    [[human group = workers | institution]] break [strike]
    [[human]] break [diet | fast]
    [[human]] break [journey]
    [[watercourse = river]] break [banks]
    idiom [waters] break
    idiom [[eventuality | human]] make or break [[anything]]
    idiom [[eventuality 1 | human 1]] break [the back] [of [[eventuality 2 = bad]]]
    idiom [[activity | artifact]] will not break [the bank]
    idiom [[human group = christian]] break [bread]
    [[human]] break [wind]
    [[phys obj]] break [[surface]]
    idiom [the straw that] break [the camel's back]
    idiom [water] break

    Follow the examples of the annotation below:
    She's breaking things = [[human | animate | event]] break [[artifact | phys obj]]
    Your waters have broke = idiom [water] break
    Is it broken? = [[artifact | phys obj]] break
    He's going to break my microphone = [[human | animate | event]] break [[artifact | phys obj]]
    He broke his leg? = [[body part = bone / tooth]] break
    
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