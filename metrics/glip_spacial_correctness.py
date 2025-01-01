"""
Used to check if the generated image (by stable diffusion) 
is consistent with the prompt.
"""
import torch
import json, os, random
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter",
        default=3,
        help="num of iter",
    )
    parser.add_argument(
        "--change_prompt",
        action="store_true",  
        help="Alter the prompt to the correct prompt",
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
    opt = parser.parse_args()
    return opt

def main(opt):
    # relations of spacial location
    RELATIONS = ["to the left of", "to the right of", "above", "below"]

    # calculate the center of a bounding box
    def BoxMid(s : list):
        x1 = s[0]
        y1 = s[1]
        x2 = s[2]
        y2 = s[3]
        return [(x1 + x2)/2.0, (y1 + y2)/2.0]

    def RelationMatchesPos(pos1: list, pos2: list, relation: str):
        if relation == RELATIONS[0]:
            return pos1[0] < pos2[0]
        if relation == RELATIONS[1]:
            return pos1[0] > pos2[0]
        if relation == RELATIONS[2]:
            return pos1[1] < pos2[1]
        if relation == RELATIONS[3]:
            return pos1[1] > pos2[1]
        return False
    
    def PosMatchesRelation(relation: str):
        if relation == RELATIONS[0]:
            return RELATIONS[1]
        elif relation == RELATIONS[1]:
            return RELATIONS[0]
        elif relation == RELATIONS[2]:
            return RELATIONS[3]
        return RELATIONS[2]


    # test if an image matches the prompt
    def IsCorrect(pos1: list, pos2: list, relation: str):
        center1 = BoxMid(pos1)
        center2 = BoxMid(pos2)
        return RelationMatchesPos(center1, center2, relation)

    def GetCorrectIds(change_prompt=False):
        correctness_data = {}
        tot_num = 0
        tot_correct = 0.0
        glip_dir = os.path.join(opt.dir, f"glip_all_metadata_{opt.model}/glip_dataset_metadata_epoch"+str(opt.iter)+".json")
        with open(glip_dir, 'r') as f:
            glip_data = json.load(f)
            
        correctness_data = []
        for key in glip_data:
            tot_num += 1
            id_now = glip_data[key]["unique_id"]
            prompt = glip_data[key]["prompt"]
            obj1 = glip_data[key]["obj1"]
            obj2 = glip_data[key]["obj2"]
            rel = glip_data[key]["relation"]
            object_names = glip_data[key]["object_names"]
            glip_boxes = glip_data[key]["glip_boxes"]
            if (len(object_names) != 2) or (obj1 not in object_names) or (obj2 not in object_names):
                if not change_prompt:
                    data = {
                        "file_name": f"{opt.iter}_{key}.png",
                        "uniq_id": id_now,
                        "image_id": key,
                        "correctness": False,
                        "prompt": prompt,
                        "reason": "number",
                    }
                    correctness_data.append(data)
                else:
                    count1 = 0
                    count2 = 0
                    for o in object_names:
                        if o == obj1:
                            count1 += 1
                        elif o == obj2:
                            count2 += 1
                    if count1!=0 and count2 != 0:
                        prompt = str(count1)+" "+obj1+" and "+ str(count2)+" "+obj2
                        data = {
                            "file_name": f"{opt.iter}_{key}.png",
                            "uniq_id": id_now,
                            "image_id": key,
                            "correctness": False,
                            "prompt": prompt,
                            "reason": "number",
                        }
                        correctness_data.append(data)
                    elif count1==0:
                        prompt =str(count2)+" "+obj2
                        data = {
                            "file_name": f"{opt.iter}_{key}.png",
                            "uniq_id": id_now,
                            "image_id": key,
                            "correctness": False,
                            "prompt": prompt,
                            "reason": "number",
                        }
                        correctness_data.append(data)
                    else:
                        prompt = str(count1)+" "+obj1
                        data = {
                            "file_name": f"{opt.iter}_{key}.png",
                            "uniq_id": id_now,
                            "image_id": key,
                            "correctness": False,
                            "prompt": prompt,
                            "reason": "number",
                        }
                        correctness_data.append(data)
                    
                continue
            idx1 = object_names.index(obj1)
            idx2 = object_names.index(obj2)
            bounding_box1 = glip_boxes[idx1]
            bounding_box2 = glip_boxes[idx2]
            correctness = IsCorrect(bounding_box1, bounding_box2, rel)
            if not change_prompt:
                data = {
                        "file_name": f"{opt.iter}_{key}.png",
                        "uniq_id": id_now,
                        "image_id": key,
                        "correctness": correctness,
                        "prompt": prompt,
                        "reason": "location",
                }
                correctness_data.append(data)
            else:
                if correctness:
                    data = {
                        "file_name": f"{opt.iter}_{key}.png",
                        "uniq_id": id_now,
                        "image_id": key,
                        "correctness": correctness,
                        "prompt": prompt,
                        "reason": "location",
                    }
                    correctness_data.append(data)
                else:
                    true_relation = PosMatchesRelation(rel)
                    prompt = "a " + obj1 + " "+ true_relation + " a "+obj2
                    data = {
                        "file_name": f"{opt.iter}_{key}.png",
                        "uniq_id": id_now,
                        "image_id": key,
                        "correctness": correctness,
                        "prompt": prompt,
                        "reason": "location",
                    }
                    correctness_data.append(data)
            if correctness:
                tot_correct += 1
        output_dir = os.path.join(opt.dir, f"diffusion_correctness_{opt.model}")
        os.makedirs(output_dir, exist_ok=True)
        outdir = os.path.join(output_dir, "diffusion_correctness_epoch" + str(opt.iter)+".jsonl")
            
        with open(outdir, 'w') as f:
            for data_entry in correctness_data:
                json.dump(data_entry, f)
                f.write('\n')

        print(tot_correct/tot_num)

    GetCorrectIds(opt.change_prompt)

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
