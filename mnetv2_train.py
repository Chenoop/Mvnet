import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate1
import json
from pytorchtools import EarlyStopping
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
torch.cuda.manual_seed(42)
# from thop import profile, clever_format
# from efficientnet_pytorch import EfficientNet
# from original_model import mobile_vit_xx_small, mobile_vit_x_small, mobile_vit_small
# from model import mobile_vit_xx_small, mobile_vit_x_small, mobile_vit_small
from typing import List, Optional
from torch.optim import Optimizer


def warmup_decorator(lr_s: type, warmup: int) -> type:
    class WarmupLRScheduler(lr_s):
        def __init__(self, optimizer: Optimizer, *args, **kwargs) -> None:
            self.warmup = warmup
            self.lrs_ori: Optional[List[int]] = None
            super().__init__(optimizer, *args, **kwargs)

        def get_lr(self) -> List[float]:
            # recover
            if self.lrs_ori is not None:
                for p, lr in zip(self.optimizer.param_groups, self.lrs_ori):
                    p["lr"] = lr
            #
            last_epoch = self.last_epoch
            lrs = super().get_lr()
            self.lrs_ori = lrs
            # warmup
            scale = 1
            if last_epoch < self.warmup:
                scale = (last_epoch + 1) / (self.warmup + 1)
            return [lr * scale for lr in lrs]
    return WarmupLRScheduler


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # device = torch.device("cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([# transforms.Resize(int(128 * 1.143)),
                                     # transforms.CenterCrop(224),
                                     transforms.Resize((224, 224)),
                                     # transforms.ColorJitter(hue=0.5, brightness=0.5, contrast=0.5, saturation=0.5),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomVerticalFlip(), # 左右翻转
                                     # transforms.RandomRotation(degrees=60),
                                     # AddPepperNoise(0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  #  [0.140, 0.140, 0.140], [0.178, 0.178, 0.178]
        "val": transforms.Compose([# transforms.Resize(int(128 * 1.143)),
                                   #transforms.CenterCrop(128),
                                   transforms.Resize((224, 224)),
                                   # transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 改
    assert os.path.exists(args.data_path), "{} path does not exist.".format(args.data_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # buildings_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in buildings_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)

    data_root = "E:\\data224\\jpg_end"
    # data_root = "D:\\dengke\\Anal\\data224"
    # data_root = "E:\\dengke\\Anal\\All_data400photos"
    # data_root = "C:\\Users\\desktop2207\\Desktop\\dengke\\Anal\\All_data400photos"
    # data_root = "C:\\Users\\scli2021_09\\Desktop\\dengke\\Anal\\anal_oax_data_400"
    # data_root = "C:\\Users\\scli2021_09\\Desktop\\dengke\\Anal\\Anal_224"
    # data_root = "D:\\images\\"
    # data_root = "C:\\Users\\scli2021_09\\Desktop\\dengke\\Anal\\All_data400photos"
    # data_root = "D:\\PycharmProjects\\Subject_research\\data"  # get data root path
    image_path = os.path.join(data_root, "train")
    # image_path = os.path.join(data_root, "building_data/train")  # building data set path
    # 遍历文件夹，一个文件夹对应一个类别
    buliding_class = [cla for cla in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, cla))]
    # 排序，保证顺序一致
    buliding_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(buliding_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=1)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in buliding_class:
        cla_path = os.path.join(image_path, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(image_path, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=data_transform["val"])
    val_num = len(val_dataset)

    # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0
                                               )  # collate_fn=train_dataset.collate_fn

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0
                                             )   # collate_fn=val_dataset.collate_fn

    #改：附加
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_dataset)))
    print("{} images for validation.".format(len(val_dataset)))

    # 如果存在预训练权重则载入
    # model = create_model(num_classes=args.num_classes)
    # # model.classifier = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    # # model.to(device)
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(in_features=1280, out_features=256),
    #     nn.SiLU(),
    #     nn.Linear(256, 2)
    # )
    # model.to(device)
    # print(model)

    # vggmodel
    # class VGGNet(nn.Module):
    #     def __init__(self, num_classes=2):  # num_classes，此处为 二分类值为2
    #         super(VGGNet, self).__init__()
    #         net = models.vgg11(pretrained=False)  # 从预训练模型加载VGG16网络参数
    #         net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
    #         self.features = net  # 保留VGG16的特征层
    #         self.classifier = nn.Sequential(  # 定义自己的分类层
    #             nn.Linear(512 * 7 * 7, 1024),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
    #             nn.ReLU(True),
    #             nn.Dropout(0.3),
    #             nn.Linear(1024, 1024),
    #             nn.ReLU(True),
    #             nn.Dropout(0.3),
    #             nn.Linear(1024, num_classes),
    #         )
    #
    #     def forward(self, x):
    #         x = self.features(x)  # 预训练提供的提取特征的部分
    #         x = x.view(x.size(0), -1)
    #         x = self.classifier(x)  # 自定义的分类部分
    #         return x
    #
    # model = VGGNet().to(device)
    # 改进的MobileNetV2
    # model = MobileNetV2(num_classes=2)

    # model = models.resnet101(pretrained=True)
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    # model_weight_path = 'E:\\PycharmProjects\\weight\\mobilenet_v2.pth'
    # assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # # delete classifier weights
    # # print(model.state_dict())
    # pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    # model.load_state_dict(pre_dict, strict=False)
    # missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    # for param in model.features.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = False
    # model = models.mobilenet_v2(pretrained=False)
    # model = models.shufflenet_v2_x2_0(pretrained=True)

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, 2)
    model.to(device)
    print(model)

    # #计算模型的参数量和计算量
    # input1 = torch.zeros((1, 3, 224, 224)).to(device)
    # flops, params1 = profile(model, inputs=(input1,), verbose=False)
    # flops, params1 = clever_format([flops, params1], "%.3f")
    # print(type(flops), type(params1))
    # print("flops: ", flops, "params: ", params1)

    # print(f"macs = {flops / 1e9}G")
    # print(f"params = {params1 / 1e6}M")
    # print("numbers of layers:", len(list(model.modules())))
    # print("numbers of layers:", len(list(model.parameters())))  # Enet:215

    # vggmodel 权重初始化层数22
    # print("numbers of layers:", len(list(model.parameters())))

    # 指定冻结6层之前(包括第6层)的层数
    # find_tune_at = 10

    # 冻结到第120层
    # for i in range(find_tune_at + 1):
    #     list(model.parameters())[i].requires_grad = False

    # 如何查看神经网络中的层是否被冻结
    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    classes = val_dataset.classes
    val_name = []
    for i in range(len(val_dataset.imgs)):
        val_name.append(val_dataset.imgs[i][0])
    # print(val_name)
    classes_name = []
    classes_name1 = []
    for i in range(len(val_name)):
        # print(classes_name)
        classes_name.append(val_name[i].split('\\')[5].split('.')[0])
        classes_name1.append(val_name[i].split('\\')[5].split('.')[0])  #3 4
    # print(classes_name)
    # print(len(classes_name))
    # 统计验证集两种肛瘘各病例的图片数量
    patient_name = []
    for i in range(len(classes_name)):
        patient_name.append(classes_name1[i].split(' ')[0])
    # print(patient_name)
    mp = {}
    for i in patient_name:
        if patient_name.count(i) > 1:
            mp[i] = patient_name.count(i)
    validate_data = json.dumps(mp, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
    # print(mp)
    # print(validate_data)
    validate_data = json.loads(validate_data)
    # for kv in validate_data.items():
    #     print(kv)
    #     print(kv[0])

    patient_trueresult = {}
    patient_falseresult = {}
    # for kv in validate_data.items():
    #     a = kv[0]
    #     patient_trueresult[a] = 0
    #     patient_falseresult[a] = 0

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 删除不需要权重
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
    #
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
        # for para in model.features.parameters():
        #     para.requires_grad = False

    T_max = 100
    warmup = 10
    eta_min = 1e-4
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(pg, lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-3)
    # optimizer = optim.Adam(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    lrs = warmup_decorator(optim.lr_scheduler.CosineAnnealingLR, warmup)(optimizer, T_max, eta_min)

    patience = 10
    avg_train_losses = []
    avg_val_losses = []
    all_train_acc = []
    all_val_acc = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    j = 0
    best_acc = 0
    save_path = "D:\\tao\\pth_tao\\warmup_data224_mobilenetv2_T_mydataset.pth"
    trainAcc_txt = "D:\\tao\\train_text\\train_warmup_data224_mobilenetv2_T_mydataset.txt"
    valAcc_txt = "D:\\tao\\train_text\\val_warmup_data224_mobilenetv2_T_mydataset.txt"
    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc, train_losses = train_one_epoch(model=model,
                                                             optimizer=optimizer,
                                                             data_loader=train_loader,
                                                             device=device,
                                                             epoch=epoch)

        # scheduler.step()
        print(epoch, lrs.get_last_lr())
        lrs.step()
        train_acc = train_acc.cpu().numpy()
        all_train_acc.append(train_acc)
        print('{} Acc: {:.4f}'.format('train', train_acc))

        # validate
        acc, val_losses, patient_trueresult, patient_falseresult = evaluate1(model=model,
                                                                             data_loader=val_loader,
                                                                             device=device,
                                                                             patient_trueresult=patient_trueresult,
                                                                             patient_falseresult=patient_falseresult,
                                                                             validate_data=validate_data,
                                                                             classes_name=classes_name,
                                                                             classes=classes,
                                                                             j=j)

        train_loss = np.average(train_losses)
        valid_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(valid_loss)
        #
        epoch_len = len(str(args.epochs))
        all_val_acc.append(acc)
        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # print('The information of numbers of CF and GF prediction true: ')
        # for kv in patient_trueresult.items():
        #     print(kv)
        #
        # print('The information of numbers of CF and GF prediction false: ')
        # for kv in patient_falseresult.items():
        #     print(kv)

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

        # print("肛瘘验证预测准确病例：")
        # print(patient_predictiontrue)
        # print("肛瘘验证预测错误病例：")
        # print(patient_predictionfalse)
        # print("肛瘘无法预测病例：")
        # print(patient_nojudge)

        valpatient_acc = (100 * (len(patient_predictiontrue)) / (len(patient_predictiontrue) + len(patient_predictionfalse)))
        print("Accuracy of the network on the validate patients:{: .1f}%".format(valpatient_acc))

        # 训练集Acc loss记录
        output1 = "Step [%d]  train Loss : %f, training accuracy :  %g" % (epoch, np.round(train_loss, 4), np.round(train_acc, 4))
        with open(trainAcc_txt, "a+") as f:
            f.write(output1 + '\n')
            f.close()

        # 验证集Acc loss记录
        output2 = "Step [%d]  val Loss : %f, val accuracy :  %g, val_patient_acc: %.1f" % (epoch, np.round(valid_loss, 4), np.round(acc, 4), valpatient_acc)
        with open(valAcc_txt, "a+") as f:
            f.write(output2 + '\n')
            f.close()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # output3 = "Flops: %s, Params: %s" % (flops, params1)
            # with open(trainAcc_txt, "a+") as f:
            #     f.write(output3 + '\n')
            #     f.close()
            break

        train_losses = []
        val_losses = []
    # model_path = './mobilenetv2_T_model.pth'
    # torch.save(model.state_dict(), model_path)

    # model.load_state_dict(torch.load('checkpoint.pt'))
        # torch.save(model.state_dict(), "./weight_nohflip/five/model_0.02-{}.pth".format(epoch))
        # torch.save(model.state_dict(), "./weights/five_valaugment/model_0.02-{}.pth".format(epoch))
        # torch.save(model.state_dict(), "./weights/images5_nodg_wgh_640/model_0.01-{}.pth".format(epoch))
    path = './jpg'
    # plt.switch_backend('TKAgg')
    # plt.figure()
    # plt.plot(avg_train_losses, 'b', label='train')
    # plt.plot(avg_val_losses, 'r', label='val')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(path, "All_losses_640.jpg"))
    plt.switch_backend('TKAgg')
    # plt.figure(figsize=(6, 3), dpi=100)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(avg_train_losses, 'b', lw=1, label='train')
    plt.plot(avg_val_losses, 'r', lw=1, label='val')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.axis('square')
    # plt.ylim([0, 1])
    plt.legend(["train_loss", "val_loss"])

    # plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(all_train_acc, 'b', lw=1, label='train')
    plt.plot(all_val_acc, 'r', lw=1, label='val')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.title("ACC")
    # plt.axis('square')
    # plt.ylim([0, 1])
    plt.legend(["train_acc", "val_acc"])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00001)  # 0.01
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="E:\\data224\\jpg_end")  # "../data/building_data" ..//All_data400photos

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')  # default=''
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
