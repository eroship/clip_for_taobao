import torch
import clip
import json
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from re import split
import pandas as pd
import csv

class Tokenizer(object):
    def __init__(self, emb_path=None, json_path=None):
        self.standard_color = ['红','黄','蓝','绿','橙','青','紫','白','灰','黑','粉','橘','银','金','栗','驼','咖','卡其','杏','棕','米','靛','褐']
        #用来在词语中搜索
        self.standard_color2 = ['红色', '黄色', '蓝色', '绿色', '橙色', '青色', '紫色', '白色', '灰色', '黑色', '粉色','橘色','银色','金色','栗色','驼色','咖色','卡其色','杏色','浅杏色','棕色','米色','靛','靛蓝','肉色','褐色']
        #用来在emb_table中搜索
        self.emb_path = emb_path
        self.json_path = json_path
        self.tag_english_dict = {}
        self.standard_tag_list = []
        #print('finish reading embedding table, num of emb_words: ', len(self.emb_words))

    def __len__(self):
        return len(self.tag_english_dict)

    def __call__(self, tag):
        return self.tag_english_dict.get(tag, 'None')

    #读取json文件，构造三个dict
    # 图片文件名-颜色标签对应的dict，颜色标签-标号dict，标号-向量dict
    def if_tag_exist(self, tag):
        if np.where(np.array(self.standard_tag_list)==tag)[0].__len__() > 0:
            return True
        else:
            return False


    def build_tag_english_dict(self, tag_english_path, standard_tag_txt_path):
        standard_tags = pd.read_csv(standard_tag_txt_path, header=None, encoding='utf-8', sep=' ', quoting=csv.QUOTE_NONE,
                                error_bad_lines=False)
        english_tags = pd.read_csv(tag_english_path, header=None, encoding='utf-8', sep='\n', quoting=csv.QUOTE_NONE,
                                error_bad_lines=False)
        self.standard_tag_list = []
        standard_tags = np.array(standard_tags[0])
        english_tags = np.array(english_tags[0])
        #(standard_tags.__len__())
        #print(english_tags.__len__())
        #print()
        for standard_tag,english_tag in zip(standard_tags, english_tags):
            self.tag_english_dict[standard_tag] = english_tag.lower()
            self.standard_tag_list.append(standard_tag)

    def generate_standard_tag_txt(self, standard_tag_txt_path):
        #生成一个包含所有标准tag的txt文件，用作翻译
        emb_table = pd.read_csv(self.emb_path, header=None, encoding='utf-8', sep=' ', quoting=csv.QUOTE_NONE,
                                error_bad_lines=False)
        standard_tags = np.array(emb_table[0])
        for standard_tag in standard_tags:
            if len(standard_tag) == 1:
                standard_tag = standard_tag+'色'
            self.standard_tag_list.append(standard_tag)

        self.standard_tag_list = set(self.standard_tag_list)
        for standard_tag in self.standard_tag_list:
            with open(standard_tag_txt_path, 'a', encoding='utf-8') as f:
                f.write(standard_tag + '\n')



    def get_color_emb_table(self, standard_tags_list_duplicate, save_path):
        #原始数据集的emb_table太大了，只需要取出所有可能用到的颜色的词语然后保存
        emb_path = self.emb_path
        print('num of tags: ', len(standard_tags_list_duplicate))
        names = os.listdir(emb_path)
        tag_not_matched = []
        for i, standard_tag in enumerate(standard_tags_list_duplicate):
            print('tag： ',i,' ',standard_tag)
            ifFound = False
            for name in names:
                path = emb_path+name
                emb_table = pd.read_csv(path, header=None, encoding='utf-8', sep=' ', quoting=csv.QUOTE_NONE,
                                        error_bad_lines=False)
                words = np.array(emb_table[0])
                loc = np.where(words == standard_tag)[0]
                if len(loc)!=0:
                    loc = loc[0]
                    print(emb_table.iloc[loc])
                    emb_list = list(emb_table.iloc[loc])
                    for t,val in enumerate(emb_list):
                        emb_list[t] = str(val)

                    emb_str = ' '.join(emb_list)
                    print(emb_str)
                    with open(save_path, 'a', encoding='utf-8') as f:
                        f.write(emb_str +' '+'\n')

                    ifFound = True
                    break

            if ifFound == False:
                tag_not_matched.append(standard_tag)
        self.emb_path = save_path
        return tag_not_matched

    def get_standard_tag_from_json(self):
        #读取json文件，整理出所有图片对应的标准标签
        json_path = self.json_path
        standard_tags_list = []
        with open(json_path, 'rt', encoding='utf-8') as f:
            img_tag_dict = json.load(f)
        for name, data in img_tag_dict.items():
            img_tags = data['imgs_tags']
            for img_tag in img_tags:
                imgname = list(img_tag.keys())[0]
                label = img_tag[imgname]
                standard_tag = standardize(label)
                standard_tags_list.append(standard_tag)
                if len(standard_tag)==1:
                    standard_tags_list.append(standard_tag+'色')
        return standard_tags_list


class mydataset(Dataset):
    def __init__(self, data_dir, train, tokenizer, preprocess=None):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.train = train
        self.data_path = Path(data_dir)/'train_all.json' if train else Path(data_dir)/'test_all.json'
        self.imgname_list = []

        self.img_tag_list = []
        self.img_tag_english_list = []
        self.img_optionaltags_english_list = []
        self.img_optionaltags_list = []

        self.preprocess = preprocess

        self.dataset_init()

    def __len__(self):
        return len(self.imgname_list)

    def __getitem__(self, item):
        imgname = self.imgname_list[item]
        if self.train == True:
            img_path = self.data_dir + '/train/' + imgname.split('_')[0] + '/' + imgname
            img = Image.open(img_path)
            tag_english = self.img_tag_english_list[item]
            tag_english = 'A photo of a '+tag_english+' color cloth'
            #print(tag_english)
            img = self.preprocess(img)
            text = clip.tokenize(tag_english).squeeze()
            return img, text
        else:
            img_path = self.data_dir + '/test/' + imgname.split('_')[0] + '/' + imgname
            #print(img_path)
            img = Image.open(img_path)
            #optional_tags = self.img_optionaltags_dict[imgname]
            optional_tags_english = self.img_optionaltags_english_list[item]
            return img, optional_tags_english

    '''def get(self, imgname):
        if self.train == True:
            img_path = self.data_dir + '/train/' + imgname.split('_')[0] + '/' + imgname
            img = Image.open(img_path)
            tag_english = self.img_tag_english_dict[imgname]
            return img, tag_english
        else:
            img_path = self.data_dir + '/test/' + imgname.split('_')[0] + '/' + imgname
            #print(img_path)
            img = Image.open(img_path)
            #optional_tags = self.img_optionaltags_dict[imgname]
            optional_tags_english = self.img_optionaltags_english_dict[imgname]
            return img, optional_tags_english'''

    def dataset_init(self):
        with open(self.data_path, 'rt', encoding='utf-8') as f:
            img_tag_dict = json.load(f)
        for name, data in img_tag_dict.items():
            img_tags = data['imgs_tags']
            optional_tags = data['optional_tags']
            #print(optional_tags)
            optional_tags_english = []
            for optional_tag in optional_tags:
                opt_tag = self.get_tag_for_label(optional_tag)
                optional_tags_english.append(self.tokenizer(opt_tag))


            for img_tag in img_tags:
                imgname = list(img_tag.keys())[0]
                self.imgname_list.append(imgname)
                self.img_optionaltags_list.append(optional_tags)
                self.img_optionaltags_english_list.append(optional_tags_english)

                if self.train == True:
                    label = img_tag[imgname]
                    tag = self.get_tag_for_label(label)
                    self.img_tag_english_list.append(self.tokenizer(tag))
                    self.img_tag_list.append(tag)

    def get_tag_for_label(self, label):
        tag = standardize(label)
        #print('standard: ',tag)
        if len(tag) == 1:
            tag = tag + '色'
        if self.tokenizer.if_tag_exist(tag) == False:
            if_found = False
            for std_color in self.tokenizer.standard_color:
                if label.count(std_color) > 0:
                    if_found = True
                    tag = std_color + '色'
                    break
            if if_found == False:
                tag = '<BOS>'
        return tag






def standardize(tag):
    #去除所有括号、数字、字母、横斜杠、加减号
    #直接在括号处截断，带来的问题：【白色】单件衬衫，这样的就没了
    #所以后续需要在标准色库中再搜索一遍
    sp = '\(|\[|【|{|（|<|《| '
    tag = split(sp,tag)[0]
    tag = tag.split('+')[0]

    char_to_delete = []
    for idx, c in enumerate(tag):
        asc = ord(c)
        if asc<128:
            tag = tag.replace(c, '')

    if tag.count('色')>0:
        tag = tag.split('色')[0] + '色'

    return tag

if __name__ == '__main__':
    emb_path = './emb_dataset/data_separate/'
    json_path = './thumbnail/train_all.json'
    tokenizer = Tokenizer(emb_path, json_path)
    stadard_tags_list = tokenizer.get_standard_tag_from_json()
    standard_tags_list_duplicate = set(stadard_tags_list + tokenizer.standard_color2)
    standard_tags_list_duplicate = sorted(standard_tags_list_duplicate, key=lambda i: len(i), reverse=False)
    for i,val in enumerate(standard_tags_list_duplicate):
        if len(val)>3:
            standard_tags_list_duplicate = standard_tags_list_duplicate[0:i]
            break
    print('length of tags: ', len(standard_tags_list_duplicate))

    print(standard_tags_list_duplicate)
    tags_emb_path = './emb_dataset/tags_emb_table.txt'
    tag_not_matched = tokenizer.get_color_emb_table(standard_tags_list_duplicate, tags_emb_path)
    print(tag_not_matched)
