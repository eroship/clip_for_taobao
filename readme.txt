运行环境：
需要安装clip，安装方法见：https://github.com/openai/CLIP

项目文件夹下包含文件夹：
model、thumbnail、emb_dataset

数据集文件夹：
thumbnail：即作业提供的最小尺寸数据集
emb_dataset：其中保存了：
	文件standard_tags.txt：所有商品的标准颜色标签
	文件standard_tags_english.txt：所有商品的标准颜色标签的英文

模型文件夹：model

代码文件：
test.py:
用于测试模型，它会输出一个预测的json文件保存到项目路径下，其中：
tese函数中的语句model, preprocess = get_model('./model/39.pt', device)用于读取模型，'./model/39.pt'可以换成'ViT-B/32'等clip模型
语句with open('./test_pred_39.json','a',encoding='utf-8') as f表示把预测结果写进'./test_pred_39.json'，可以改成其他的名称
直接运行test.py即可得到表示测试结果的json文件

data_utils.py：
定义了Tokenizer和mydataset

train.ipynb：
因为训练过程是在kaggle服务器上运行的，因此是ipynb格式。
其中第一部分是安装clip，需要在有网络的情况下运行
倒数第二个代码块中有几个路径：
standard_tag_path = '../input/tag-txt/standard_tags.txt'
tag_english_path = '../input/tag-txt/standard_tags_english.txt'
data_dir = '../input/dataset-img-tag/thumbnail'
get_model('../input/model-pretrain/24.pt', device)
这些路径是kaggle服务器上存储数据集的路径，如果是在自己的电脑上运行，需要更改成：
standard_tag_path = './emb_dataset/standard_tags.txt'
tag_english_path = './emb_dataset/standard_tags_english.txt'
data_dir = './thumbnail'
get_model('/model/24.pt', device)
其中get_model函数也可以改成get_model('ViT-B/32', device)来加载clip模型

直接运行train.ipynb会在当前的load的model的基础上训练，每三个epoch保存一次模型，在当前的项目路径下


	