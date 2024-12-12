from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "microsoft/Phi-3-mini-4k-instruct"
#model_id = "microsoft/Phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda:0", 
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
    "temperature": None,
    "do_sample": False,
}

def response(args, premise, hypothesis):
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    prefix = "Refer to the explanation and predict whether the relationship between premise and hypothesis is 'contradiction', 'entailment', or 'neutral', but never add any explanation other than the label name as the example"
    '''
    messages = [
        {"role": "system", "content": f'{prefix}'},
        {"role": "user", "content": f'premise:{premise} hypothesis:{hypothesis}, Label: '},
        ]
    '''
    messages = [
        {"role": "user", "content": f'{prefix}'},
        {"role": "assistant", "content": f'Example:{example}'},
        {"role": "user", "content": f'premise:{premise}\nhypothesis:{hypothesis}\nLabel: '},
        ]
    #print(messages[0]["content"])
    print((pipe(messages, **generation_args)[0]['generated_text']))
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
                "answer":""
            }
            
            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            empty['label'] = d['label']
            
            answer = response(args,d['premise'], d['hypothesis'])
            empty['answer'] = answer
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )
    
if __name__ == '__main__':
    main()
