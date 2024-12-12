from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_id = "microsoft/Phi-3-mini-4k-instruct"
#model_id = "microsoft/Phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
Plan = "Step1. Understand the premise and hypothesis// Step2. Determine if there is a difference between the information that can be obtained from the premises and the information that can be obtained from the hypothesis.// Step3. Based on what you understand, determine whether the premise falls under 'contradiciton', 'entailment', or 'neutral'."

def response(args, premise, hypothesis):
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    prefix = "Solve the problem step by step, following a plan to predict whether the hypothesis is contradiction” “entailment” or “neutral” with respect to the premises, as shown in the example."
    messages = [
        {"role": "user", "content": f'{prefix}'},
        {"role": "assistant", "content": f'Example:{example}'},
        {"role": "user", "content": f'premise:{premise}\nhypothesis:{hypothesis}\nPlan:{Plan}\nSolution:'},
    ]
    print((pipe(messages, **generation_args)[0]))
    return (pipe(messages, **generation_args)[0]['generated_text'])

def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument('--input-path',type=str)
    parser.add_argument('--example-path',type=str)
    parser.add_argument('--output-path',type=str)
    parser.add_argument('--shot', type=int, default=4, help="In-Context의 개수")
    args = parser.parse_args()
    
    data=[]
    new_li=[]
    
    with open(args.input_path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                data.append(d)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                
                
    with open(args.output_path,'w') as out_f:
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
            
            Solution = response(args,d['premise'], d['hypothesis'])
            empty['Solution'] = Solution
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )
    
if __name__ == '__main__':
    main()
            
            