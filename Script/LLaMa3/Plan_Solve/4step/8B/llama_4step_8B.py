from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
Plan = "Step1. Determine whether there is a superordinate or synonymous relationship between the words that make up the premise and the words that make up the hypothesis.// Step2. Determined whether the words forming the premise and the words forming the hypothesis have an antonym relationship or a negative relationship, or whether they have no relationship with each other.// Step3. Determine whether a hypothesis has been formed by adding additional information to the information obtained from the premises.// Step4. Based on the judgments, infer whether the premise corresponds to 'contradiction', 'entailment', or 'neutral'."

def response(args, premise, hypothesis):
    with open(args.example_path) as f:
        example_string = f.read().split('{shot}')[:(args.shot)]
    if args.shot > 0:
        example = ''.join(example_string)
    
    prefix = "Solve the problem step by step, following a plan to predict whether the hypothesis is contradiction” “entailment” or “neutral” with respect to the premises, as shown in the example."
    messages = [
        {"role": "system", "content": f'{prefix}\nExample\n{example}'},
        {"role": "user", "content": f'premise:{premise} hypothesis:{hypothesis}, Plan:{Plan} Solution:'},
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
        max_new_tokens=256,
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
            
            