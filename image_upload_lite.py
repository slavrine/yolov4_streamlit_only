import streamlit as st
import os
import io
import av
import tempfile
import shutil
import gdown
import zipfile
from pathlib import Path

from memory_profiler import profile


# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto


def load_image():
    uploaded_file = st.file_uploader(label='Pick a file to test')
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        return opencv_image
    else:
        return None


def load_video():
    uploaded_file = st.file_uploader(label='Pick a file to test')
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(uploaded_file)
        return tfile.name
    else:
        return None


def export_from_split_file(source_folder, target_file):

    # create a empty file
    with open(target_file, "wb") as gf:
        gf.write(b"")
    
    # write all buffers to a new file
    export = sorted([i for i in os.listdir(source_folder)])
    # For each source file, we read it, apppending to new file.
    with open(target_file, "ab") as f:
        
        for _ in export:
            tmp = source_folder+"/"+str(_)
            
            with open(tmp, "rb") as f1:
                tmp = f1.read(-1)
            
            f.write(tmp)

def download_file_from_google_drive(id,
                                    destination):

    # create an empty directory                                
    Path('custom').mkdir(exist_ok=True)
    Path(destination).mkdir(exist_ok=True)

    # download split zip files
    gdown.download_folder(id= id, quiet=False)

    #combine split zip files into one zip file
    export_from_split_file("custom", "custom.zip")

    #unzip into the checkpoints directory
    with zipfile.ZipFile("custom.zip", 'r') as zip_ref:
        zip_ref.extractall(destination)

    #clean up extra unnecessary folder+files
    if Path('custom.zip').exists(): os.remove('custom.zip')
    if Path('custom').exists(): shutil.rmtree('custom')
   


# def load_model():
#     model = tf.saved_model.load('./checkpoints/custom-608', tags=[tag_constants.SERVING])
#     return model


@st.cache
def load_model():

    f_checkpoint = Path("./checkpoints/custom-608")
    if not f_checkpoint.exists():
        id = st.text_input('google drive folder id', '')
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_file_from_google_drive(id=id,
                                    destination = 'checkpoints/')

    model = tf.saved_model.load('./checkpoints/custom-608', tags=[tag_constants.SERVING])
    return model

def predict_video (saved_model_loaded, video):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 608

    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video))
    except:
        vid = cv2.VideoCapture(video)

    out = None
    out_file='./detections/results.mp4'
    output_memory_file = io.BytesIO()
    output = av.open(output_memory_file, 'w', format="mp4")
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, codec, fps, (width, height))

    stream = output.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
    stream.width = width  # Set frame width
    stream.height = height  # Set frame height
    # stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
    stream.pix_fmt = 'yuv420p'  # Select yuv420p pixel format for wider compatibility.
    stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).

    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image


        image = utils.draw_bbox(frame, pred_bbox, False, allowed_classes=allowed_classes,
                                    read_plate=False)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        frame_av = av.VideoFrame.from_ndarray(image, format='bgr24')  # Convert image from NumPy Array to frame.
        packet = stream.encode(frame_av)  # Encode video frame
        output.mux(packet)
        # cv2.imwrite('./detections/' + 'crop/' + 'imgs_for_lable/' + str(frame_num) + '.png', image)

        # result = np.asarray(image)
        #
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
    out.release()
    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    output_memory_file.seek(0)
    st.video(output_memory_file)

    # video_file = open(out_file, 'rb')
    # video_bytes = video_file.read()

    st.download_button('Processed_Video', output_memory_file, file_name='Processed_Video.mp4',
                       mime=None, key=None, help=None, on_click=None, args=None,
                       kwargs=None)

def predict_image (saved_model_loaded, image):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 608

    # load image
    original_image = image
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # load_weights
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes,
                            read_plate=False)

    image = Image.fromarray(image.astype(np.uint8))

    st.image(image)
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./detections/' + 'detection' + str(1) + '.png', image)

@profile(precision=4)
def main():
    st.title('Pretrained model demo')
    with st.spinner('Heating up the model...'):
        model = load_model()
    option = st.selectbox('Plz select a file?', ('Image', 'Video'))

    if option=='Image':
        original_image = load_image()
        result = st.button('Run on file')
        if result:
            st.write('Calculating results...')
            predict_image(model, original_image)

    if option == 'Video':
        with st.spinner('loading video...'):
            original_video = load_video()
        result = st.button('Run on file')
        if result:
            with st.spinner('Calculating results...'):
                predict_video(model, original_video)

if __name__ == '__main__':
    main()

