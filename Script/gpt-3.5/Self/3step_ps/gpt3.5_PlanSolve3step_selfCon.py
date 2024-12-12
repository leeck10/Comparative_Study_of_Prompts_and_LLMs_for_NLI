import argparse
import json
import openai
from tqdm import tqdm
import time
from collections import Counter
OPEN_API_KEY = ""
openai.api_key = OPEN_API_KEY
Plan = "Step1. Understand the premise and hypothesis// Step2. Determine if there is a difference between the information that can be obtained from the premises and the information that can be obtained from the hypothesis.// Step3. Based on what you understand, determine whether the premise falls under 'contradiciton', 'entailment', or 'neutral'."

def score_for_input(args, premise, hypothesis, plan):
    print(1)
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    print(2)
    prompts = None
    prefix = "Solve the problem step by step, following a plan to predict whether the hypothesis is 'contradiction' 'entailment' or 'neutral' with respect to the premises, as shown in the example."
    
    _pred =""
    print(3)
    # Chat 기반 모델 
    if args.model_type == 'gpt-3.5-turbo-16k':
        print(4)
        if args.task == 'knowledge':
            if args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}\nExample\n{example} '},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} Plan: {plan} Solution :'}]

        if prompts is None:
            raise Exception(f'score_for_input() not implemented for {args.task}!')
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-16k',
                    messages=prompts,
                    max_tokens=512,
                    temperature=0.5,
                    #top_p=1,
                    n=3,
                    #stop='\n',
                )
                break
            except Exception as e:
                print(e)
                time.sleep(60)
                    
        _pred = response.choices[0].message.content 

    return response.choices[0].message.content, response.choices[1].message.content, response.choices[2].message.content 


def check_equal_frequency(my_list):
    # 요소의 빈도 수를 계산
    freq = Counter(my_list)
    
    # 요소의 종류가 하나뿐이면 False 반환
    if len(freq) < 2:
        return False
    
    # 모든 값이 동일한지 확인
    first_count = None
    for count in freq.values():
        if first_count is None:
            first_count = count
        elif first_count != count:
            return False
    return True

def self_con(args, premise, hypothesis,plan):
    di = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    un = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
    labellist = ['contradiction', 'entailment', 'neutral']
    while True:
        path1, path2, path3 = score_for_input(args, premise, hypothesis,plan) 
        ans = [path1, path2, path3]
        sen = f'ans1:{path1}\nans2:{path2}\nans3:{path3}'
        print(sen)

        new = []
        for ii in ans:
            sen = ii.replace("'", "")
            sen1 = sen.replace(".", "")
            sen2 = sen1.replace('"', "")
            sen3 = sen2.replace('*', "")
            words = sen3.split()
            predict = words[len(words)-1].lower()
            if predict not in labellist:
                continue
            else:
                new.append(di[predict])
        print("========================================",len(new),new)
        if len(new)==3 and check_equal_frequency(new)==False:
            
            break
    coun = Counter(new)
    print(un[coun.most_common(1)[0][0]])
    return sen, un[coun.most_common(1)[0][0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['baseline', 'knowledge'])
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=3, help="In-Context의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/chl/Self/3step_ps/prompt4inference_ver.ps_4shot.txt', help="In-Context 환경, 태스크 예시 경로")
    parser.add_argument('--label_type', type=str,)
    args = parser.parse_args()
    
    if args.task == 'knowledge':
        args.input_path = '/home/cyh/GKP-master/chl/Self/input.json'
    args.output_path = f'./esnli_selfCon_3step_{args.shot}shot.json'
    data = []
    new_li = []
    with open(args.input_path ,'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 현재 줄을 JSON 객체로 파싱
                d = json.loads(line)
                data.append(d)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")


    with open(args.output_path, 'w') as out_f:  
        for d in data:
            empty = {
                "premise": "",
                "hypothesis": "",
                "label": "",
                "Plan": "",
                "Solve": "",
                "answer": ""
            }

            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            empty['label'] = d['label']
            empty['Plan'] = Plan

            Solve, answer = self_con(args, d['premise'], d['hypothesis'], Plan)
            empty['Solve'] = Solve
            empty['answer'] = answer
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )

if __name__ == '__main__':
    main()