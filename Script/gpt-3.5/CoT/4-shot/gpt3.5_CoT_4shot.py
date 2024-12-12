import argparse
import json
import openai
from tqdm import tqdm
import time
from collections import Counter
OPEN_API_KEY = ""
openai.api_key = OPEN_API_KEY

def score_for_input(args, premise, hypothesis):
    print(1)
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    print(2)
    prompts = None
    prefix = "Think and infer step by step as an example whether the relationship between premise and hypothesis corresponds to 'contradiction', 'entailment', or 'neutral'."
    
    _pred =""
    print(3)
    # Chat 기반 모델 
    if args.model_type == 'gpt-3.5-turbo-16k':
        print(4)
        if args.task == 'knowledge':
            if args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}\nExample\n{example} '},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} Answer:'}]

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
                    n=5,
                    #stop='\n',
                )
                break
            except Exception as e:
                print(e)
                time.sleep(60)
                    
        _pred = response.choices[0].message.content 
        print(response)

    return response.choices[0].message.content, response.choices[1].message.content, response.choices[2].message.content, response.choices[3].message.content, response.choices[4].message.content

def self_con(args, premise, hypothesis):
    path1,path2,path3,path4,path5=score_for_input(args, premise, hypothesis)
    di={
        'contradiction':0,
        'entailment':1,
        'neutral':2
    }
    ans=[path1,path2,path3,path4,path5]
    new=[]
    for i in ans:
        sen = i.replace("'","")
        sen1 = sen.replace(".","")
        sen2 = sen1.replace('"',"")
        
        words = sen2.split()
        
        predict = words[len(words)-1].lower()
        new.append(predict)
    sen = f'ans1:{path1}\nans2:{path2}\nans3:{path3}\nans4:{path4}\nans5:{path5}\n'
    print(sen)
    final=[]
    for ii in new:
        n = ii.lower()
        final.append(di[n])
    counter = Counter(final)

    # 가장 빈번한 요소 찾기 (가장 많은 빈도수를 가진 요소 1개)
    most_common_element = counter.most_common(1)[0][0]
    
    if most_common_element == 0:
        return sen,'contradiction'
    elif most_common_element == 1:
        return sen,'entailment'
    elif most_common_element == 2:
        return sen,'neutral'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['baseline', 'knowledge'])
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=3, help="In-Context의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/chl/Self/CoT/data4prompt_4shot.txt', help="In-Context 환경, 태스크 예시 경로")
    parser.add_argument('--label_type', type=str,)
    args = parser.parse_args()
    
    if args.task == 'knowledge':
        args.input_path = '/home/cyh/GKP-master/chl/Self/input.json'
    args.output_path = f'./self_con_esnli_CoT_{args.shot}shot.json'
    data=[]
    new_li = []
    with open(args.input_path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                # 현재 줄을 JSON 객체로 파싱
                d = json.loads(line)
                data.append(d)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
            
            
    with open(args.output_path, 'w') as out_f:  
        for d in data:
            empty={
                "premise": "",
                "hypothesis": "",
                "Answer":""
            }
            
            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            
            Answer = score_for_input(args,d['premise'], d['hypothesis'])
            empty['Answer'] = Answer
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )
    
if __name__ == '__main__':
    main()