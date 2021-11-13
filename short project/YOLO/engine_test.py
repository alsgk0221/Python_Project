import torch
import json
from glob import glob
import cv2
from detect_fire import detect_img
from torchvision.transforms import transforms
from PIL import Image

from models.experimental import attempt_load
from utils.torch_utils import select_device

test_json = "../data/test_frame/sample_fire_scenes.json"
file_paths = glob("../data/test_frame/*")

# save_path에 저장할 위치를 입력
def ensemble_test(model: torch.nn.Module, model_2: torch.nn.Module):
    with open(test_json, "r") as st_json:
        temp = json.load(st_json)

    model.eval()
    model_2.eval()

    total_dict = temp
    with torch.no_grad():
        for file_name in file_paths:
            name = file_name.split('/')[-1] + '.mp4'
            file_path = glob(file_name + "/*")

            classificaton_result = []
            detection_result = []

            # in file video frame
            for i in range(len(file_path)):
                video_name = file_name + '/' + str(i) + '.png'
                img = cv2.imread(video_name)

                # Change data for classfication
                image = cv2.resize(img, (224, 224))
                inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
                inputs = torch.unsqueeze(inputs, 0).float()

                # Enter the data
                pred = model(inputs)
                pred_2 = model_2(inputs)

                # Ensemble classification model result
                pred = (pred + pred_2) / 2
                pred = pred[0]

                # Save classification result
                if pred[0] < pred[1]:
                    classificaton_result.append(pred[1].item())
                else:
                    classificaton_result.append(0)

            # check if only values 0 make list to empty
            check = set(classificaton_result)

            # save result classifiation and detectio
            if len(check) == 1:
                total_dict[name]['scenes'] = []
            else:
                total_dict[name]['scenes'] = classificaton_result

            total_dict[name]['detection_scenes'] = detection_result
        save_path= "../data/results/eff_result.json"
        with open(save_path, 'w') as f:
            json.dump(total_dict, f)

# save_path에 저장할 위치를 입력
def resnet_classification(model: torch.nn.Module):

    with open(test_json, "r") as st_json:
        temp = json.load(st_json)

    class_names = ['Fire', 'Neutral', 'Smoke']

    model.eval()
    total_dict = temp
    with torch.no_grad():
        for file_name in file_paths:
            name = file_name.split('/')[-1] + '.mp4'
            file_path = glob(file_name + "/*")

            classificaton_result = []

            # in file video frame
            for i in range(len(file_path)):
                video_name = file_name + '/' + str(i) + '.png'

                img = Image.open(video_name)

                # Change data for classfication
                prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

                image = prediction_transform(img)[:3, :, :].unsqueeze(0)
                #image = image.cuda()

                # Enter the data
                pred = model(image)

                idx = torch.argmax(pred)
                prob = pred[0][idx].item() * 100
                classificaton_result.append(class_names[idx] + " " + str(prob))

            total_dict[name]['scenes'] = classificaton_result

        save_path = '../data/results/resnet_result.json'
        with open(save_path, 'w') as f:
            json.dump(total_dict, f)


# save_path에 저장할 위치를 입력
def detection_result():

    with open(test_json, "r") as st_json:
        temp = json.load(st_json)

    # for detection settings
    device = select_device('cpu')

    dect_model = attempt_load('../save_model/yolo_best.pt', map_location=device)  # load FP32 model
    dect_model.eval()

    total_dict = temp
    with torch.no_grad():
        for file_name in file_paths:

            name = file_name.split('/')[-1] + '.mp4'
            file_path = glob(file_name + "/*")
            detection_result = []

            # in file video frame
            for i in range(len(file_path)):
                video_name = file_name + '/' + str(i) + '.png'
                img = cv2.imread(video_name)

                # Detection model
                detection_pred, _ = detect_img(img, dect_model)
                detection_result.append(detection_pred)

            total_dict[name]['detection_scenes'] = detection_result

        save_path = '../data/results/detection_result.json'
        with open(save_path, 'w') as f:
            json.dump(total_dict, f)


def detection_result():

    # for detection settings
    device = select_device('cpu')

    dect_model = attempt_load('../save_model/yolo_best.pt', map_location=device)  # load FP32 model
    dect_model.eval()

    with torch.no_grad():
            detection_result = []

            # in file video frame
            #video_name = file_name + '/' + str(i) + '.png'
            img_path = ""
            img = cv2.imread(img_path)

            # Detection model
            detection_pred, _ = detect_img(img, dect_model)
            detection_result.append(detection_pred)

