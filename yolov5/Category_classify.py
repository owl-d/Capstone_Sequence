# 지금부터 하려고 하는 작업에 대한 대략적인 개요를 적고자 한다.
# Data를 저장하는 곳은 바탕화면 또는 어느 폴더의 CustomData에 저장되어있다.
# 이를 서버에 올려 사용한다면 json 파일에 과자,면류,유제품,음료,주류에 해당하는 폴더 앞 index를 저장
# txt에서 나온 index와 detect.py 실행으로 나온 결과와 대입했을 때 해당하는 개체들이 많은 class를 부여한다.

import os
import sys
from pathlib import Path
import json
import yaml
from glob import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

text_path = glob('./runs/detect/task' + '/*.txt')
json_path = './classify.json'
label2name_json_path = './label2name.json'
yaml_path = ROOT / 'file_path/data.yaml'
# 이 부분은 detect.py 실행 후 입력으로 받을 예정
# labels = ["코카스프라이트500ML","코카스프라이트500ML","하이트)필라이트후레쉬(캔)500ML","광동)옥수수수염차500ML","테라(캔)500ML",'롯데밀키스500ML']

#-----------------------------------------------------------------------------------------------------------------------

# detect.py를 통해 나온 txt를 가져다가 이를 labels라는 목록으로 변환하는 코드
labels = []
data = yaml.load(open(yaml_path,'r',encoding = 'utf-8'),Loader=yaml.FullLoader)
labels.extend(data['names'])
# print(labels)
index = list(range(1,len(labels)))
index2label_dict = dict(zip(index,labels))

'''
# ./classify.json 파일 생성하는 코드 : 해당하는 품목의 제품이 유제품인지, 음료인지 나누어놓은 json 파일입니다.

#-----------------------------------------------------------------------------------------------------------------------
CustomData_path = '/home/parkjuntae/바탕화면/CustomData/Train_data'

file_data = OrderedDict()  # json 파일 생성을 위한 함수

# 목록 불러오기
upper_folder = glob(CustomData_path +'/**')
upper_folder = [str(os.path.basename(x)) for x in upper_folder]
print(upper_folder)
_folder_list =[]
for fold_name in upper_folder:
    temp_path = CustomData_path + '/' + str(fold_name)
    temp_list = glob(temp_path +'/**')
    # name = [os.path.basename(x).split('_')[0] for x in temp_list]
    # label = [os.path.basename(x).split('_')[-1] for x in temp_list]
    # file_data[fold_name] = dict(zip(name,label))
    file_data[fold_name] = [os.path.basename(x).split('_')[-1] for x in temp_list]
with open(json_path, 'w', encoding = 'utf-8') as make_json:
    json.dump(file_data,make_json,ensure_ascii=False,indent='\t')

#-----------------------------------------------------------------------------------------------------------------------
# ./label2name.json 파일 생성하는 코드 : 해당하는 품목의 제품이 유제품인지, 음료인지 나누어놓은 json 파일입니다.

CustomData_path = '/home/parkjuntae/바탕화면/CustomData/Train'

file_data = OrderedDict()  # json 파일 생성을 위한 함수

# 목록 불러오기
upper_folder = glob(CustomData_path +'/**')
upper_folder = [str(os.path.basename(x)) for x in upper_folder]
print(upper_folder)
_folder_list =[]
name = [os.path.basename(x).split('_')[0] for x in upper_folder]
label = [os.path.basename(x).split('_')[-1] for x in upper_folder]
with open(label2name_json_path, 'w', encoding = 'utf-8') as make_json:
    json.dump(dict(zip(name,label)),make_json,ensure_ascii=False,indent='\t')

#----------------------------------------------------------------------------------------------------------------------
'''
json_data = json.load(open(label2name_json_path, 'r',encoding = 'utf-8'))

label_to_name = []
for idx in labels:
    label_to_name.append(json_data[idx])

def Category(text_path = 'runs/detect/task/labels/',json_path = './classify.json'):
    with open(json_path, 'r',encoding = 'utf-8') as f:
        json_data = json.load(f)

    text_path = glob(text_path+'*.txt')
    print(text_path)
    max_name = []
    for path in text_path:
        with open(path, 'r') as tf:
            strings = tf.readlines()
        print(strings)
        pre_label = list(int(x.split(' ')[0]) for x in strings)
        labels = list(label_to_name[x] for x in pre_label)
        print(labels)
        _len_list = []
        for index in list(json_data.keys()):
            intersection = [index for idx in labels if len(list(set(json_data[str(index)]) & {idx}))]
            _len_list.append(len(intersection))
        max_name.append(list(json_data.keys())[_len_list.index(max(_len_list))])

    return max_name

print(Category())
