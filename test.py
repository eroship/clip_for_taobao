import json

import torch
import clip
import cv2
import numpy as np
from data_utils import *

def get_model(name, device):
    model_clip = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    for mc in model_clip:
        if name == mc:
            model, preprocess = clip.load(name, device)
            return model, preprocess

    model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
    checkpoint = torch.load(name)

    #checkpoint['model_state_dict']["input_resolution"] = 224#input_resolution
    #checkpoint['model_state_dict']["context_length"] = 77 #model.context_length
    #checkpoint['model_state_dict']["vocab_size"] = model.vocab_size
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, preprocess

def test(dataset):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    #model, preprocess = clip.load('RN50x64', device)
    model, preprocess = get_model('./model/39.pt', device)

    #img, optional_tags, optional_tags_english = dataset.get(imgname)
    data_path = dataset.data_path
    with open(data_path, 'rt', encoding='utf-8') as f:
        img_tag_dict = json.load(f)
    num = len(img_tag_dict)

    i=0
    num_item = 0
    for name, data in img_tag_dict.items():
        num_item = num_item + 1
        #print(name)
        img_tags = data['imgs_tags']
        optional_tags = data['optional_tags']
        img, optional_tags_english = dataset[i]

        text = [f"A photo of a {color} color cloth" for color in optional_tags_english]
        text_tockens = clip.tokenize(text).to(device)

        img_list = []
        for img_tag in img_tags:
            imgname = list(img_tag.keys())[0]
            #print(imgname)
            img, optional_tags_english = dataset[i]
            i = i+1
            img = img.convert('RGB')
            img = preprocess(img).unsqueeze(0).to(device)
            img_list.append(img)

        img_input = torch.stack(img_list).squeeze()
        #print(img_list[0].shape)
        #print(torch.stack(img_list).squeeze().shape)

        with torch.no_grad():
            text_features = model.encode_text(text_tockens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            #print(text_features)
        with torch.no_grad():
            image_features = model.encode_image(img_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #print(image_features)
            #logits_per_image, logits_per_text = model(img, text_tockens)
            #probs = logits_per_image.softmax(dim=-1).to('cpu').numpy()
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        tag_pred = np.argmax(similarity, axis=0)
        for t,img_tag in enumerate(img_tags):
            imgname = list(img_tag.keys())[0]
            img_tag_dict[name]["imgs_tags"][t][imgname] = optional_tags[tag_pred[t]]

        #print(similarity)
        #print(optional_tags)
        #print(optional_tags_english)
        #print(img_tag_dict[name])
        #print(probs)
        if num_item%100==0:
            print('already handle ', num_item,'in ', num, ' items')

    #json_data = json.dumps(img_tag_dict)
    with open('./test_pred_39.json','a',encoding='utf-8') as f:
        #f.write(json_data)
        json.dump(img_tag_dict, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    torch.cuda.set_device(0)
    tag_english_path = './emb_dataset/standard_tags_english.txt'
    standard_tag_path = './emb_dataset/standard_tags.txt'
    tokenizer = Tokenizer()
    tokenizer.build_tag_english_dict(tag_english_path, standard_tag_path)

    data_dir = './thumbnail'
    dataset_test = mydataset(data_dir, False, tokenizer)
    test(dataset_test)







