import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#from absl import app, flags, logging
#from absl.flags import FLAGS
import core.utils as utils
# from core.config import cfg
# from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

Fweights = 'data/custom-416'

Fclasses = 'data/classes/obj.names'

#Flags
Ftiny = False
Fmodel = 'yolov4'
FFsize = 416
Fframework = 'tf'
Foutput_format = 'XVID'
Fiou = 0.45
Fscore = 0.25
Fdont_show = True

#global counter for processed clothes
throw_for_frame = [0,0,0,0,0]
global_cloth_counter = 0
workstation_cloth_counter = [0,0,0,0,0]

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(Fclasses, Ftiny, Fmodel)
saved_model_loaded = tf.saved_model.load(Fweights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

def increment(y, flag):
    if flag == 'data':
        y += 30
    elif flag == 'line':
        y += 40
    return y

def draw_on_frames(workstations,frame):
    global global_cloth_counter, workstation_cloth_counter
    frame = add_area(frame)
    font_scale = 1.5
    font_thickness = 2
    color_station = (0,255,0)
    color = (0,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    start_x = 1290
    start_y = 50
    cv2.putText(frame, 'Stations Detected: 5',(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Total Cloth Processed: {}'.format(global_cloth_counter),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'line')
    cv2.putText(frame, '*'*25,(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'line')
    cv2.putText(frame, 'Cloth Processed:',(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y,'data')
    cv2.putText(frame, 'Workstation 1: {}'.format(workstation_cloth_counter[0]),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Workstation 2: {}'.format(workstation_cloth_counter[1]),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Workstation 3: {}'.format(workstation_cloth_counter[2]),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Workstation 4: {}'.format(workstation_cloth_counter[3]),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Workstation 5: {}'.format(workstation_cloth_counter[4]),(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'line')
    cv2.putText(frame, '*'*25,(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'line')
    cv2.putText(frame, 'Target: 50',(start_x,start_y),font,font_scale,color,font_thickness)
    start_y = increment(start_y, 'data')
    cv2.putText(frame, 'Achieved: {}'.format(global_cloth_counter),(start_x,start_y),font,font_scale,color,font_thickness)

    for workstation in workstations:
        for i in workstation:
            img = cv2.rectangle(frame, (workstation[0], workstation[2]), (workstation[1], workstation[3]),color_station, 2)
    return img

def add_area(frame):
    # frame.shape = (1440, 1280, 3)
    # right side area (horizontal stack)
    black_area = np.zeros([1440,352,3], dtype=np.uint8)
    black_area.fill(175)
    frame = np.concatenate((frame, black_area), axis=1)
    return frame

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # print(bb1)
    # print(bb2)
    # print('_____________')
    # # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
       # print('here')
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_status(workstations, rois, iou_threshold):
    # workstations = [x1,x2,y1,y2]
    # roi = [xmin,ymin,xmax,ymax,class_id]
    throwing_cont = [0,0,0,0,0]
    ret_flag = {}
    ious = [0,0,0,0,0]
    for roi in rois:
        if roi[4] == 1:
            for itr, workstation in enumerate(workstations):
                roi_box = {
                    'x1':roi[0],
                    'x2':roi[2],
                    'y1':roi[1],
                    'y2':roi[3]
                }
                workstation_box = {
                    'x1':workstation[0],
                    'x2':workstation[1],
                    'y1':workstation[2],
                    'y2':workstation[3]
                }
                iou = get_iou(workstation_box, roi_box)
                #print(iou)

                if iou > iou_threshold:
                    ious[itr] = iou
                    workstation_number_throwing = ious.index(max(ious))
                    throwing_cont[workstation_number_throwing] = 1
            #print(ious)
            #print(throwing_cont)
            ious = [0,0,0,0,0]
    for itr in range(len(workstations)):
        if ( throwing_cont[itr] == 1 ):
            throw_for_frame[itr] += 1
        else:
            throw_for_frame[itr] = 0

def check_cloth(counter):
    global global_cloth_counter, workstation_cloth_counter
    for itr in range(len(throw_for_frame)):
        if (throw_for_frame[itr] >= 16):
            print(throw_for_frame)
            global_cloth_counter += 1
            workstation_cloth_counter[itr] += 1
            print('Workstation {} processed cloth on frame {}'.format(itr+1, counter))
            throw_for_frame[itr] = 0



def format_boxes(boxes, height, width, threshold):
    # out_boxes -> coordinates
    #out_scores -> scores
    #out_classes -> classes
    #num_boxes -> number of predictions
    out_boxes, out_scores, out_classes, num_boxes = list(boxes)
    rois = []

    for itr in range(num_boxes[0]):
        if out_scores[0][itr] > threshold:
            xmin = int(out_boxes[0][itr][1]* width)
            ymin = int(out_boxes[0][itr][0] * height)
            xmax = int(out_boxes[0][itr][3] * width)
            ymax = int(out_boxes[0][itr][2] * height)
            class_id = int(out_classes[0][itr])
            rois.append([xmin,ymin,xmax,ymax, class_id])
    return rois

def detect_inventory(frame, threshold, get_proc_frame = True):
    image_data = cv2.resize(frame, (FFsize,FFsize))
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
        iou_threshold=Fiou,
        score_threshold=Fscore
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    height, width, _ = frame.shape
    rois = format_boxes(pred_bbox, height, width, threshold)
    if get_proc_frame:
        frame = utils.draw_bbox(frame, pred_bbox)
    return rois, frame