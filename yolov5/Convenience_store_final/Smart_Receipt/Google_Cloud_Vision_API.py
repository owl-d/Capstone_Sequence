#pip install --upgrade google-cloud-vision

# Linux --------------------------------------------------------------------------------

# export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
# KEY_PATH를 서비스 계정 키가 포함된 JSON 파일의 경로로 바꿉니다.
# 예를 들면 다음과 같습니다.
# export GOOGLE_APPLICATION_CREDENTIALS="/home/user/Downloads/service-account-file.json"

# Windows --------------------------------------------------------------------------------

# PowerShell:
# $env:GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
# KEY_PATH를 서비스 계정 키가 포함된 JSON 파일의 경로로 바꿉니다.
# 예를 들면 다음과 같습니다.
# $env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\username\Downloads\service-account-file.json"

# 명령 프롬프트:
# set GOOGLE_APPLICATION_CREDENTIALS=KEY_PATH

# ----------------------------------------------------------------------------------------
# reference : https://konlpy.org/ko/latest/install/

# Linux Mecab 설치 코드
# !sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
# !python3 -m pip install --upgrade pip
# !python3 -m pip install konlpy
# !sudo apt-get install curl git
# !bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# ----------------------------------------------------------------------------------------

import io
import os
import cv2
import numpy as np
# file_path = './receipt_image.jpg'
file_path = "down/result.jpg"
# save_path = "./save_text.txt"
save_path = 'receipt_text/GCVtext_new_14.txt'

# ----------------------------------------------------------------------------------------
# ETRI 정보가 들어온 경우 이를 형태소 분류기로 분류해 해당 단어를 형태소로 나누고 이 내용이
Input_ETRI = ['블랙아메리카노','칸타타헤이즐넛',"빅컵얼음"]


def GCV_model(file_path, save_path):

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr1-346505-37a54a8fe96a.json"

    # Imports the Google Cloud client library
    from google.cloud import vision

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.abspath(file_path)
    # img_array = np.fromfile(file_name, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # crop_img = cv2.resize(img, dsize=(1024, 768), fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs text detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations

    print('Texts:')
    print(type(texts))
    print(texts)

    # 파일 내용을 지우기 위한 용도
    with open(save_path, "w", encoding="UTF-8") as file:
        pass

    for text in texts:
        with open(save_path, "a",encoding="UTF-8") as file:
            file.write(text.description + '\n')
            # vertices = (['({},{})'.format(vertex.x, vertex.y)
            #              for vertex in text.bounding_poly.vertices])
            # file.write('bounds: {}'.format(','.join(vertices)) + '\n')
            file.close()

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

GCV_model(file_path,save_path)

# 이 밑 코드부터는 ETRI API로 생성된 text를 받았다고 가정하고 구성할 것이다.
# 예를 들어 CU,세븐일레븐,GS25,이마트24 편의점이 있다고 하면

with open(save_path, "r+",encoding="UTF-8") as file:
    strings = file.readlines()

def CU(Input_ETRI,strings):
    # for voice_text in Input_ETRI:
    del_index = []
    # enable = False
    for i, string in enumerate(strings):
        if 'POS' in string.strip(): #and not enable:
            del_index.append(i)
            # enable = True

        if "과세" in string.strip().strip(' ') or '부가' in string.strip().strip(' '):
            del_index.append(i)

    strings = strings[del_index[0] + 1:del_index[1]]

    print(strings)

    # length = 0
    # enable = False
    # for i, string in enumerate(strings):
    #     if string.strip().isdigit() and not enable:
    #         length = i
    #         enable = True

    digit_list1 = []
    digit_list2 = []
    text_list = []

    match_digit = [] # Input으로 들어온 ETRI정보와 일치하는 물품의 가격
    match_text = [] # Input으로 들어온 ETRI정보와 일치하는 물품의 이름

    for string in strings:
        new_string = string.strip().replace(',','').replace(' ','').replace('.','').replace('-','')
        if new_string.isdigit() or new_string == '증정' or new_string == '증점':
            if new_string == '증정' or new_string == '증점':
                digit_list2.append('증정')
            elif len(new_string) <= 2:
                digit_list1.append(int(new_string))
            else:
                digit_list2.append(int(new_string))
        else:
            text_list.append(new_string)
    print(digit_list1) # 숫자 중 개수 정보 리스트
    print(digit_list2) # 숫자 중 가격 정보 리스트
    print(text_list) # 감지한 전체 상품 리스트
    for voice_text in Input_ETRI:
        for j, text in enumerate(text_list):
            if voice_text in text:
                print('물품 :',text_list[j])
                match_digit.append(text_list[j])
                # print('개수 :',digit_list1[j])
                print('가격 :',digit_list2[j])
                match_text.append(digit_list2[j])

CU(Input_ETRI,strings)