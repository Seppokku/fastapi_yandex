import torch
from torchvision.models import resnet18
import torchvision.transforms as T
import json
import joblib
import os
from transformers import AutoTokenizer, AutoModel
import re
import string
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002B50-\U00002B55"  # Miscellaneous Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "]+",
    flags=re.UNICODE,
)


def clean(text):
    text = text.lower()  # нижний регистр
    text = re.sub(r"http\S+", " ", text)  # удаляем ссылки
    text = re.sub(r"@\w+", " ", text)  # удаляем упоминания пользователей
    text = re.sub(r"#\w+", " ", text)  # удаляем хэштеги
    text = re.sub(r"\d+", " ", text)  # удаляем числа
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"<.*?>", " ", text)  #
    text = re.sub(r"[️«»—]", " ", text)
    text = re.sub(r"[^а-яё ]", " ", text)
    text = text.lower()
    text = emoji_pattern.sub(r"", text)
    return text


def load_classes():
    '''
    Returns IMAGENET classes
    '''
    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = load_classes()
    return labels[i]

def load_pt_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model = resnet18()
    model.load_state_dict(torch.load('weights/resnet18-weights.pt', map_location='cpu'))
    model.eval()
    return model

def load_sklearn_model():
    clf = joblib.load('weights/logistic_regression_model.pkl')
    return clf

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    print(trnsfrms(img).shape)
    return trnsfrms(img).unsqueeze(0)

def load_tokenizer():
    model = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    return model

def load_bert():
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
    return model

