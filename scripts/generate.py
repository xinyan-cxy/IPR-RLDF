from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import json, argparse, os, random
from itertools import islice
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt.",
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=0,
        help="num of iter.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../output", 
        help="output path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='',
        help="path of finetuned ckpt"
    )
    parser.add_argument(
        "--use_finetuned_ckpt",
        action='store_true', 
        default=False,
        help="use the finetuned sd ckpt",
    )
    parser.add_argument(
        "--use_lora",
        action='store_true', 
        default=False,
        help="use lora",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sdv2",
        help="base model (sdxl or sdv2)",
    )
    args = parser.parse_args()
    return args

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
    
def generate_sd(args):
    viz_ids = [
                19100, 24653, 29131, 8606, 17652, 6603, 26515, 22815, 7904, 6486, 26363, 
                22495, 18253, 12812, 20714, 29841, 23283, 29120, 23113, 810, 9942, 22356, 
                3792, 7257, 29971, 20086, 20727, 10321, 2084, 27141, 30955, 29633, 23544, 
                13352, 27244, 19973, 7646, 21186, 7366, 17831, 8001, 12373, 12046, 8966, 
                7264, 15896, 29727, 5257, 4254, 8754, 17066, 7170, 26186, 16226, 8341,
                10516, 25814, 887, 19792, 24514, 3937, 27667, 19794, 7335, 21865, 5416, 
                14686, 31510, 27552, 18714, 14405, 4381, 23780, 22884, 22461, 21636, 14555, 
                18915, 10811, 19134, 3344, 13642, 21645, 16896, 22927, 6431, 29065, 1824, 
                14972, 8963, 13984, 26053, 22416, 11271, 28697, 17604, 18051, 5015, 15407, 
                6465
                ]
    
    with open('../text_spatial_rel_phrases.json', 'r') as f:
        text_data = json.load(f)
        data = []
        names = []
        for uniq_id in viz_ids:
            free_form_prompt = text_data[uniq_id]["text"]
            image_id = str(uniq_id)
            for _ in range(args.batch_size):
                data.append(free_form_prompt)
            names.append(image_id)
        data = list(chunk(data, args.batch_size))
        
    if args.model=="sdv2":
        print("using sdv2")
        # model = "/data/chenxy/models/stable-diffusion-2-1"
        model = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        pipe.to("cuda")
    else:
        print("using sdxl")    
        # model = "/data/chenxy/models/stable-diffusion-xl-base-1.0" 
        model = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        pipe.to("cuda")

    for data_slice, image_id in tqdm(zip(data, names), total=len(data)):
        for i, free_form_prompt in enumerate(data_slice):
            if args.model=="sdv2":
                with suppress_stdout_stderr():
                    image = pipe(prompt=free_form_prompt).images[0]
            else:
                with suppress_stdout_stderr():
                    image = pipe(free_form_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
                    
            out_path = os.path.join(args.out_path, f"{args.model}_dataset/{str(args.iter_num)}") 
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, f"{args.iter_num}_{image_id}_{i}.png") )

    
def generate_finetuned_sd(args):
    viz_ids = [
                19100, 24653, 29131, 8606, 17652, 6603, 26515, 22815, 7904, 6486, 26363, 
                22495, 18253, 12812, 20714, 29841, 23283, 29120, 23113, 810, 9942, 22356, 
                3792, 7257, 29971, 20086, 20727, 10321, 2084, 27141, 30955, 29633, 23544, 
                13352, 27244, 19973, 7646, 21186, 7366, 17831, 8001, 12373, 12046, 8966, 
                7264, 15896, 29727, 5257, 4254, 8754, 17066, 7170, 26186, 16226, 8341,
                10516, 25814, 887, 19792, 24514, 3937, 27667, 19794, 7335, 21865, 5416, 
                14686, 31510, 27552, 18714, 14405, 4381, 23780, 22884, 22461, 21636, 14555, 
                18915, 10811, 19134, 3344, 13642, 21645, 16896, 22927, 6431, 29065, 1824, 
                14972, 8963, 13984, 26053, 22416, 11271, 28697, 17604, 18051, 5015, 15407, 
                6465
                ]
    
    with open('../text_spatial_rel_phrases.json', 'r') as f:
        text_data = json.load(f)
        data = []
        names = []
        for uniq_id in viz_ids:
            free_form_prompt = text_data[uniq_id]["text"]
            image_id = str(uniq_id)
            for _ in range(args.batch_size):
                data.append(free_form_prompt)
            names.append(image_id)
        data = list(chunk(data, args.batch_size))
    
    model_path = args.ckpt_path
    if args.model=="sdv2":
        if args.use_lora == False:
            print("using finetuned sdv2, not using lora")
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
            pipe.to("cuda")
        else:
            print("using finetuned sdv2, using lora")
            # pipe = StableDiffusionPipeline.from_pretrained("/data/chenxy/models/stable-diffusion-2-1", torch_dtype=torch.float16)
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
            pipe.unet.load_attn_procs(model_path)
            pipe.to("cuda")
    else:
        print("using finetuned sdxl, using lora")
        # pipe = DiffusionPipeline.from_pretrained("/data/chenxy/models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.load_lora_weights(model_path)
        
    for data_slice, image_id in tqdm(zip(data, names), total=len(data)):
        for i, free_form_prompt in enumerate(data_slice):
            if args.use_lora == False:
                with suppress_stdout_stderr():
                    image = pipe(prompt=free_form_prompt).images[0]
            else:
                with suppress_stdout_stderr():
                    image = pipe(free_form_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
                    
            out_path = os.path.join(args.out_path, f"{args.model}_dataset/{str(args.iter_num)}") 
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, f"{args.iter_num}_{image_id}_{i}.png") )

    
if __name__ == "__main__":
    args = parse_args()
    if args.use_finetuned_ckpt:
        generate_finetuned_sd(args)
    else:
        generate_sd(args)