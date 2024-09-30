import os
import json

import torch
import torchvision.datasets
from PIL import Image
from torchvision import transforms, models

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')


def patient_test(classes_name, i, classes, predicted, labels, patient_trueresult, patient_falseresult):
    # 收集各病例的图片预测结果数量
    for prefix in ['CF', 'GF']:
        for j in range(1, 101):
            class_label = f'{prefix}{j}'
            if classes_name[i].split(' ')[0] == class_label:
                if classes[predicted] == classes[labels]:
                    patient_trueresult[class_label] += 1
                else:
                    patient_falseresult[class_label] += 1


def patient_val(classes_name, i, classes, predicted, labels, patient_trueresult, patient_falseresult):
    # 收集各病例的图片预测结果数量
    for prefix in ['CF', 'GF']:
        for j in range(1, 101):
            class_label = f'{prefix}{j}'
            if classes_name[i].split(' ')[0] == class_label:
                if classes[predicted] == classes[labels]:
                    patient_trueresult[class_label] += 1
                else:
                    patient_falseresult[class_label] += 1