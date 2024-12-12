import argparse
import json
import openai
from tqdm import tqdm
import time

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
    prefix = 'Infer that the relationship between premises and hypothesis is ‘contradiction’, ‘entailment’, or ‘neutral’.'
    _pred =""
    print(3)
    # Chat 기반 모델 
    if args.model_type == 'gpt-3.5-turbo-16k':
        print(4)
        prefix = 'Infer that the relationship between premises and hypothesis is ‘‘contradiction’’, ‘entailment’, or ‘neutral’.'
        if args.task == 'knowledge':
            if args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}'},
                            {"role": "user", "content": f'premise: {premise} hypothesis: {hypothesis} Label :'}]

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
                    n=1,
                    #stop='\n',
                )
                break
            except Exception as e:
                print(e)
                time.sleep(10)
                    
        _pred = response.choices[0].message.content 
        print(_pred)

    return _pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['baseline', 'knowledge'])
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=0, help="In-Context의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/chl/Plan_Solve/baseline/data4prompt_4shot.txt', help="In-Context 환경, 태스크 예시 경로")
    args = parser.parse_args()
    
    if args.task == 'knowledge':
        args.input_path = '/home/cyh/GKP-master/chl/Plan_Solve/baseline/new.json'
    args.output_path = f'/home/cyh/GKP-master/chl/Plan_Solve/baseline/Inference_baseline_zero-shot2222.json'
    
    data=[]
    new_li = []
    with open(args.input_path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                data.append(d)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                
    with open(args.output_path, 'w') as out_f:  
        for d in data:
            empty={
                "premise": "",
                "hypothesis": "",
                "keywords": "",
                "label": "",
                "knowledge": "",
                "Label" : ""
            }
            
            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            empty['keywords'] = d['keywords']
            empty['label'] = d['label']
            empty['knowledge'] = d['knowledge']
            
            label = score_for_input(args,d['premise'], d['hypothesis'])
            empty['Label'] = label
            new_li.append(empty)
            out_f.write(json.dumps(empty) +'\n' )
    
if __name__ == '__main__':
    main()