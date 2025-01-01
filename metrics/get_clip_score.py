from tqdm import tqdm
from PIL import Image
import torch
import numpy  as np
import json, argparse, os, random
from tqdm import tqdm
from itertools import islice
from torchmetrics.functional.multimodal import clip_score
from functools import partial

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
        default=3,
        help="num of iter.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/data/chenxy/chenxy/output_add/imagereward/refl_lora_data/3/",
        help="image path",
    )
    args = parser.parse_args()
    return args

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def calculate_clip_score(image, prompts, clip_score_fn):
    # import pdb;pdb.set_trace()
    # images_int = (np.asarray(images[0]) * 255).astype("uint8")
    images_int = np.asarray(image).astype("uint8")
    images_int = np.expand_dims(images_int, axis=0)
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def get_clip_score(args):
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
    with open('text_spatial_rel_phrases.json', 'r') as f:
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
        
    clip_score_fn = partial(clip_score, model_name_or_path="/data/chenxy/chenxy/models/clip-vit-large-patch14")

    tot_num = 0
    tot_clip_score = 0.0
    for data_slice, image_id in tqdm(zip(data, names), total=len(data)):
        for i, free_form_prompt in enumerate(data_slice):
            tot_num += 1
            image_path = args.image_path + str(args.iter_num) + "_" + image_id + f"_{i}.png" 
            # image_path = args.image_path + image_id + f"_{i}.png" 
            image = Image.open(image_path)
            score = calculate_clip_score(image, free_form_prompt, clip_score_fn)
            tot_clip_score += score
    print(tot_clip_score/tot_num)

if __name__ == "__main__":
    args = parse_args()
    get_clip_score(args)