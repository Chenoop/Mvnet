import os
import json
import torch
import torchvision.datasets
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn.metrics as sm
import matplotlib
matplotlib.use('AGG')
from CFGF100_new import patient_test
import pandas as pd
# from model import mobile_vit_small, mobile_vit_x_small


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         # transforms.CenterCrop(224),
         # transforms.Resize((128, 128)),
         # transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # All_data400photos: [0.142, 0.142, 0.142], [0.178, 0.178, 0.178]

    # All_photoes_512 [0.179, 0.179, 0.179], [0.218, 0.218, 0.218]

    # test_path = "C:\\Users\\scli2021_09\\Desktop\\dengke\\Anal\\Anal_224\\test"
    # test_path = 'D:\\images\\test'
    # test_path = "C:\\Users\\scli2021_09\\Desktop\\dengke\\Anal\\All_data400photos\\test"
    # test_path = "D:\\All_data400photos\\test"
    test_path = "E:\\data224\\jpg_end\\test"
    # test_path = "D:\\dengke\\Anal\\externaldata\\external_test_zhang"
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    classes = test_dataset.classes
    test_name = []
    for i in range(len(test_dataset.imgs)):
        test_name.append(test_dataset.imgs[i][0])
    print(test_name)
    classes_name = []
    classes_name1 = []
    for i in range(len(test_name)):
        classes_name.append(test_name[i].split('\\')[5].split('.')[0])
        classes_name1.append(test_name[i].split('\\')[5].split('.')[0])  # 9 images:4
    print(classes_name)
    # print(len(classes_name))

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, 2)
    model.to(device)
    print(model)

    # model_weight_path = "D:\\PycharmProjects\\save_weight\\data224_mobilenetv2_UT.pth"
    # model_weight_path = './changing_model.pth'
    model_weight_path = "D:\\tao\\pth_tao\\\warmup_data224_mobilenetv2_T_mydataset.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # weights_dict = torch.load(model_weight_path, map_location=device)
    # # load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    # # model.load_state_dict(load_weights_dict, strict=False)

    def imshow(inp, title, ylabel):
        """Imshow for Tensor."""
        inp = inp.cpu()
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.show()
        plt.ylabel('GroundTruth: {}'.format(ylabel))
        plt.title('predicted: {}'.format(title))

    # 统计测试集两种肛瘘各病例的图片数量
    patient_name = []
    for i in range(len(classes_name)):
        patient_name.append(classes_name1[i].split(' ')[0])
    print(patient_name)
    mp = {}
    for i in patient_name:
        if patient_name.count(i) > 1:
            mp[i] = patient_name.count(i)
    test_data = json.dumps(mp, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
    # print(mp)
    print(test_data)
    test_data = json.loads(test_data)
    # for kv in test_data.items():
    #     print(kv)
    #     print(kv[0])

    patient_trueresult = {}
    patient_falseresult = {}
    for kv in test_data.items():
        a = kv[0]
        patient_trueresult[a] = 0
        patient_falseresult[a] = 0
    # print(patient_trueresult)
    # print(patient_falseresult)
    # print(patient_falseresult['CF1'])

    false_onepatient = 0
    true_onepatient = 0
    correct = 0
    total = 0
    i = 0
    j = 0
    filename = 'D:\\tao\\test_again\\external_CFGF_test_data224_mobilenetv2_T_mydataset.txt'
    filename1 = 'D:\\tao\\test_again\\external_CFGF_test_data224_mobilenetv2_T_mydataset）.csv'
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # print(images.numpy())
            images = images.to(device)
            out = torchvision.utils.make_grid(images)
            # torch.squeeze(model(img.to(device))).cpu()
            outputs = torch.squeeze(model(images).to(device)).cpu()
            _, predicted = torch.max(outputs.data, 0)
            # print(predicted)
            # print(i, 'class_name:', ''.join('%10s' % classes_name[i]), '.Predicted:', ''.join('%5s' % classes[predicted]), '  GroundTruth:',
            #           ''.join('%5s' % classes[labels]))
            patient_test(classes_name, i, classes, predicted, labels, patient_trueresult, patient_falseresult)
            # with open(filename, 'a') as file_object:
            #     file_object.write("classes_name:"+classes_name[i]+"    ")
            #     file_object.write("predicted: "+classes[predicted]+"   ")
            #     file_object.write("truelabels:"+classes[labels]+"\n")

            if j % 1 == 0:  # 设置每个窗口显示1张
                plt.figure()
                # j = j % 5
            # plt.subplot(1, 1, 1)
            # imshow(out, title=[classes[predicted]], ylabel=[classes[labels]])
            j = j + 1
            i = i + 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(true_onepatient)
            # print(false_onepatient)
            # print("Accuracy of the network on the test images:{: .1f}%".format((100 * correct / total)))
            # if i == 0:
            #     break
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        print('The information of numbers of CF and GF prediction true: ')
        # for kv in patient_trueresult.items():
        #     print(kv)
        # print(patient_trueresult.items())
        with open(filename, "a+") as f:
            f.write('The information of numbers of CF and GF prediction true: ' + '\n')
            for k, v in patient_trueresult.items():
                f.write(str(k) + ": " + str(v) + '\n')
        f.close()

        scores = {}
        for k1, k2 in zip(patient_trueresult.keys(), patient_falseresult.keys()):
            if k1 == k2:
                single_score = (patient_trueresult[k1]/(patient_trueresult[k1] + patient_falseresult[k2]))
                single_score = round(single_score, 2)
                scores[k1] = single_score

        sub_scores = {}
        for key, value in scores.items():
            num = 1 - value
            num = round(num, 3)
            sub_scores[key] = num

        print('The information of numbers of CF and GF prediction false: ')
        for kv in patient_falseresult.items():
            print(kv)

        with open(filename, "a+") as f:
            f.write('The information of numbers of CF and GF prediction false: ' + '\n')
            for k, v in patient_falseresult.items():
                f.write(str(k) + ": " + str(v) + '\n')
        f.close()

        scores = json.dumps(scores, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
        sub_scores = json.dumps(sub_scores, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
        # print(mp)
        print(scores, end=',')
        print(sub_scores, end=',')
        scores = json.loads(scores)
        sub_scores = json.loads(sub_scores)
        # for key, value in scores.items():
        #     print(value, end=',')
        with open(filename, "a+") as f:
            f.write("诊断患病概率：" + '\n')
            for key, value in scores.items():
                f.write(str(key) + ": " + str(value) + "\n")
            f.write("--------------------" + '\n')
        f.close()

        with open(filename, "a+") as f:
            for i, (key, value) in enumerate(scores.items()):
                if i > 100:
                    num = 1 - value
                    num = round(num, 3)
                    f.write(str(num) + '\n')
                    CF_baseline_Acc = "%f" % num
                    list = [CF_baseline_Acc]
                    data = pd.DataFrame([list])
                    data.to_csv(filename1, mode='a', header=False, index=False)
                else:
                    f.write(str(value) + '\n')
                    CF_baseline_Acc = "%f" % value
                    list = [CF_baseline_Acc]
                    data = pd.DataFrame([list])
                    data.to_csv(filename1, mode='a', header=False, index=False)
            f.write("--------------------" + '\n')
        f.close()
            # print(value, end=',')
        # for key, value in sub_scores.items():
        #     print(value, end=',')
        with open(filename, "a+") as f:
            f.write("病人诊断真实患病概率：" + '\n')
            for key, value in sub_scores.items():
                f.write(str(key) + ": " + str(value) + "\n")
            f.write("--------------------" + '\n')
        f.close()

        patient_predictiontrue = []
        patient_predictionfalse = []
        patient_nojudge = []
        for key in patient_trueresult.keys():
            for key1 in patient_falseresult.keys():
                if key == key1 and patient_trueresult[key] > patient_falseresult[key1]:
                    # print(key + '预测正确')
                    patient_predictiontrue.append(key)
                if key == key1 and patient_trueresult[key] <= patient_falseresult[key1]:
                    # print(key + '预测错误')
                    patient_predictionfalse.append(key)
                if key == key1 and patient_trueresult[key] == patient_falseresult[key1]:
                    patient_nojudge.append(key)

        print("肛瘘预测准确病例：")
        print(patient_predictiontrue)
        print("肛瘘预测错误病例：")
        print(patient_predictionfalse)
        print("肛瘘无法预测病例：")
        print(patient_nojudge)

        print("Accuracy of the network on the test patients:{: .2f}%".
              format(
            (100 * (len(patient_predictiontrue)) / (len(patient_predictiontrue) + len(patient_predictionfalse)))))
        print("Accuracy of the network on the test images:{: .2f}%".format((100 * correct / total)))

        test_patient_acc = 100 * (len(patient_predictiontrue)) / (len(patient_predictiontrue) + len(patient_predictionfalse))
        test_patient_acc = np.round(test_patient_acc, 5)
        test_image_acc = 100 * correct / total
        # print(test_image_acc, test_patient_acc)
        with open(filename, "a+") as f:
            f.write("肛瘘预测准确病例：" + '\n')
            str1 = ','.join(patient_predictiontrue)
            f.write(str1 + '\n')
            f.write("肛瘘预测错误病例：" + '\n')
            str2 = ','.join(patient_predictionfalse)
            f.write(str2 + '\n')
            f.write("test_patient_acc: %.1f, test_image_acc: %f" % (test_patient_acc, test_image_acc))
        f.close()


if __name__ == '__main__':
    main()