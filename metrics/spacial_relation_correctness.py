"""
Used to check if the generated image (by stable diffusion) 
is consistent with the prompt.
"""
import torch
import json, os, random
import argparse

NUMBER2ENGLISH = ['zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_path",
        default=3,
        help="Output path of the generated sd folder",
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

    def GetCorrectIds():
        correctness_data = {}
        tot_num = 0
        number_error = 0.0
        tot_l_r_num = 0
        tot_a_b_num = 0
        l_r_corr = 0.0
        a_b_corr = 0.0
        glip_dir = "/data/chenxy/chenxy/output_add/imagereward/glip_all_metadata_refl/glip_dataset_metadata_epoch"+str(opt.sd_path)+".json"
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
            if rel == RELATIONS[0] or rel == RELATIONS[1]:
                tot_l_r_num += 1
            else:
                tot_a_b_num += 1
            if (len(object_names) != 2) or (obj1 not in object_names) or (obj2 not in object_names):
                number_error+=1
                continue
            idx1 = object_names.index(obj1)
            idx2 = object_names.index(obj2)
            bounding_box1 = glip_boxes[idx1]
            bounding_box2 = glip_boxes[idx2]
            correctness = IsCorrect(bounding_box1, bounding_box2, rel)
            if correctness:
                if rel == RELATIONS[0] or rel == RELATIONS[1]:
                    l_r_corr+=1
                else:
                    a_b_corr+=1

        print("number accuracy: ", 1 - number_error/tot_num)
        print(tot_l_r_num)
        print("left-right accuracy: ", l_r_corr/tot_l_r_num)
        print(tot_a_b_num)
        print("above-below accuracy: ", a_b_corr/tot_a_b_num)

    GetCorrectIds()

if __name__ == "__main__":
    opt = parse_args()
    main(opt)

# number to english