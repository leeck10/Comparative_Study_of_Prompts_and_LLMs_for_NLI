import argparse
import json
import openai
from tqdm import tqdm
import time
OPEN_API_KEY = ""
openai.api_key = OPEN_API_KEY

Plan = "Step1. Determine whether there is a superordinate or synonymous relationship between the words that make up the premise and the words that make up the hypothesis.// Step2. Determined whether the words forming the premise and the words forming the hypothesis have an antonym relationship or a negative relationship, or whether they have no relationship with each other.// Step3. Determine whether a hypothesis has been formed by adding additional information to the information obtained from the premises.// Step4. Based on the judgments, infer whether the premise corresponds to 'contradiction', 'entailment', or 'neutral'."
def score_for_input(args, premise, hypothesis, plan):
    print(1)
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    print(2)
    prompts = None
    prefix = "Solve the problem step by step, following a plan to predict whether the hypothesis is 'contradiction' 'entailment' or 'neutral' with respect to the premises. Explain the inference process step by step, and inform the inference result at the end, such as 'Answer: Inferred label'."
    _pred =""
    print(3)
    # Chat 기반 모델 
    if args.model_type == 'gpt-3.5-turbo-16k':
        print(4)
        if args.task == 'knowledge':
            if args.shot > 0:
                prompts = [{"role": "system", "content": f'{prefix}'},
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
                    n=1,
                    #stop='\n',
                )
                break
            except Exception as e:
                print(e)
                time.sleep(60)
                    
        _pred = response.choices[0].message.content 
        print(_pred)

    return _pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['baseline', 'knowledge'])
    parser.add_argument('--model-type', type=str, default='gpt3')
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--shot', type=int, default=3, help="In-Context의 개수")
    parser.add_argument('--example-path', type=str, default='/home/cyh/GKP-master/chl/Plan_Solve/only_PS/plainPS/ps_4step/4-shot/prompt4inference_ver2_4shot.txt', help="In-Context 환경, 태스크 예시 경로")
    parser.add_argument('--label_type', type=str,)
    args = parser.parse_args()
    
    if args.task == 'knowledge':
        args.input_path = '/home/cyh/GKP-master/chl/Plan_Solve/only_PS/plainPS/ps_2step/4-shot/22_esnli_knowledge_3shot.knowledge.gpt-3.5-turbo-16k.json'
    args.output_path = f'./esnli_4Step_PS_zero-shot.json'
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
                "label": "",
                "Plan":"",
                "Solution":""
            }
            
            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            empty['label'] = d['label']
            empty['Plan'] = Plan
            
            Solution = score_for_input(args,d['premise'], d['hypothesis'],Plan)
            empty['Solution'] = Solution
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )
    
if __name__ == '__main__':
    main()