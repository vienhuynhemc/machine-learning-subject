# import thư viện cần dùng
from PIL import Image
import base64
import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from os.path import dirname, join
from joblib import load
# ----------------------------------------------------

# Load model-----------------------------------
# Load model cnn để sử dụng
m_json = join(dirname(__file__), "model5.json")
json_file = open(m_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
mModelCNN = join(dirname(__file__), "model5.h5")
loaded_model.load_weights(mModelCNN)
modelCNN = loaded_model
# Load model svm để sử dụng
mModelSVM = join(dirname(__file__), "mnist-svm.joblib")
modelSVM = load(mModelSVM)
# ----------------------------------------------

# Ký tự có thể nhận diện
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
              'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# Các biến được sử dụng
image = None
sorted_ctrs = None
list_image = None

# Lấy image từ file


def getImageFromFile(file):
    decoded_data = base64.b64decode(file)
    np_data = np.fromstring(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    height, width, depth = image.shape

    # resizing the image to find spaces better
    image = cv2.resize(image, dsize=(width * 5, height * 4),
                       interpolation=cv2.INTER_CUBIC)

    return image

# Cho ra sorted_ctrs


def getSorted_ctrs(image):
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # adding GaussianBlur
    gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)

    # find contours
    ctrs, hier = cv2.findContours(
        gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return sorted_ctrs

# Cho ra hình ảnh đã được vẽ khung bọc


def getImage(file):
    global image
    image = getImageFromFile(file)
    global sorted_ctrs
    sorted_ctrs = getSorted_ctrs(image)

    dp = image.copy()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(dp, (x - 10, y - 10),
                      (x + w + 10, y + h + 10), (86, 207, 211), 20)

    pil_im = Image.fromarray(dp)
    buff = io.BytesIO()
    pil_im.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return str(img_str, 'utf-8')

# Trả về danh sách các hình được cắt nhỏ


def getImageProcessingCNN(not_value):
    result = list()
    for i in range(len(list_image)):
        pil_im = Image.fromarray(list_image[i])
        buff = io.BytesIO()
        pil_im.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        result.append(str(img_str, 'utf-8'))
    return 'ket_noi'.join(result)


def cnn(not_value):
    global list_image
    global sorted_ctrs
    list_image = list()
    pchl = list()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y - 20:y + h + 40, x - 40:x + w + 40]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.medianBlur(roi,5)
        ret,roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
        roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        list_image.append(roi)
        roi = np.array(roi)
        t = np.copy(roi)
        t = t / 255.0
        t = 1 - t
        t = t.reshape(1, 784)
        pred = modelCNN.predict_classes(t)
        pchl.append(pred)

    pcw = list()
    for i in range(len(pchl)):
        pcw.append(characters[pchl[i][0]])

    predstring = ''.join(pcw)
    return predstring


def svm(not_value):
    global list_image
    global sorted_ctrs
    list_image = list()
    pchl = list()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y - 20:y + h + 40, x - 40:x + w + 40]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.medianBlur(roi,5)
        ret,roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
        roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        list_image.append(roi)
        roi = np.array(roi)
        t = np.copy(roi)
        t = t / 255.0
        t = 1 - t
        t = t.reshape(1, 784)
        pred = modelSVM.predict(t)
        pchl.append(pred)

    pcw = list()
    for i in range(len(pchl)):
        pcw.append(characters[pchl[i][0]])

    predstring = ''.join(pcw)
    return predstring
