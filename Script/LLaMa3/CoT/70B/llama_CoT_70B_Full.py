from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #load_in_4bit = True,
    device_map="auto",
)

def response(args, premise, hypothesis):
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    
    prefix = "Think and infer step by step as an example whether the relationship between premise and hypothesis corresponds to 'contradiction', 'entailment', or 'neutral'."
    messages = [
        {"role": "system", "content": f'{prefix}\nExample\n{example}'},
        {"role": "user", "content": f'premise:{premise} hypothesis:{hypothesis}, Answer:'},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    print(result)
    return result


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
                "Answer":""
            }
            
            empty['premise'] = d['premise']
            empty['hypothesis'] = d['hypothesis']
            empty['label'] = d['label']
            
            Answer= response(args,d['premise'], d['hypothesis'])
            empty['Answer'] = Answer
            new_li.append(empty)
            out_f.write(json.dumps(empty) + '\n' )
    
if __name__ == '__main__':
    main()
            
            