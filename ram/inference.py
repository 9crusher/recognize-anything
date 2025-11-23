'''
 * The Inference of RAM++ Model (Open-Set)
 * Written by Xinyu Huang
'''
import torch


def inference_ram_openset(image, model):

    with torch.no_grad():
        tags = model.generate_tag_openset(image)

    return tags[0]
