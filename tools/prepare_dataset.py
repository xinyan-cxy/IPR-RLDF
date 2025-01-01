import os
import json
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter",
        default=1,
        help="num of iter",
    )
    parser.add_argument(
        "--dir",
        default="../output",
        help="directory of detection results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sdv2",
        help="base model (sdxl or sdv2)",
    )
    args = parser.parse_args()
    return args

def main(args):
    data_one = []
    with open(f"{args.dir}/diffusion_correctness_{args.model}/diffusion_correctness_epoch0.jsonl", 'r') as file:
        for line in file:
            data_one.append(json.loads(line))
            
    if int(args.iter)!=0:
        with open(f"{args.dir}/diffusion_correctness_{args.model}/diffusion_correctness_epoch{args.iter}.jsonl", 'r') as file:
            for line in file:
                data_one.append(json.loads(line))

    training_dataset = f"{args.dir}/{args.model}_dataset/train"
    if os.path.exists(training_dataset):
        shutil.rmtree(training_dataset)  
    os.makedirs(training_dataset, exist_ok=True)
    with open(os.path.join(training_dataset, 'metadata.jsonl'), 'w') as file:
        for item in data_one:
            file.write(json.dumps(item) + '\n')
            
    for filename in os.listdir(f"{args.dir}/{args.model}_dataset/0"):
        if filename.endswith('.png'):
            file_path = os.path.join(f"{args.dir}/{args.model}_dataset/0", filename)
            output_path = os.path.join(training_dataset, filename)
            shutil.copyfile(file_path, output_path)
    
    if int(args.iter)!=0:        
        for filename in os.listdir(f"{args.dir}/{args.model}_dataset/{args.iter}"):
            if filename.endswith('.png'):
                file_path = os.path.join(f"{args.dir}/{args.model}_dataset/0", filename)
                output_path = os.path.join(training_dataset, filename)
                shutil.copyfile(file_path, output_path)

    print("Dataset preparation is done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
