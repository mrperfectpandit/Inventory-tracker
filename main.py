import cv2
import os
import numpy as np
from detect_inventory import detect_inventory, get_status, check_cloth, draw_on_frames


def read_video(in_path, out_path, threshold,iou_threshold, get_proc_frame = True):
    counter = 0
    cap = cv2.VideoCapture(in_path)
    while cap.isOpened():
        ret , frame = cap.read()
        if ret:
            rois, frame = detect_inventory(frame, threshold)
            get_status(workstations, rois, iou_threshold)
            check_cloth(counter)
            frame = draw_on_frames(workstations, frame)
            if get_proc_frame:
                cv2.imwrite(os.path.join(out_path,str(counter) + '.jpg'), frame)
                print(f'{counter}.jpg saved...')
        else:
            cap.release()
        counter = counter + 1

def make_video(output_folder, video_filename):
    image_list = []
    image_path_list = os.listdir(output_folder)
    image_path_list = sorted(image_path_list, key = lambda x: int(x[:len(x)-4]))
    print(image_path_list)
    for image in image_path_list:
        img = cv2.imread(os.path.join(output_folder,image))
        img = cv2.resize(img, (1280,720))
        h,w,_ = img.shape
        size = (w,h)
        image_list.append(img)

    video = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'DIVX'),15,size)

    for i in range(len(image_list)):
        video.write(image_list[i])

    video.release()
    print('Video Saved')

def is_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print('Creating {}...'.format(path))


if __name__ == '__main__' :
    vid_name = 'first_run.mp4'
    in_path = os.path.join('video_data', vid_name)
    out_path = 'output'
    is_dir(out_path)
    childoutpath = vid_name.split('.')[0]
    finaloutpath = os.path.join(out_path,childoutpath)
    is_dir(finaloutpath)
    threshold = 0.6
    iou_threshold = 0.4
    workstations = [[138,671,923,1440],[481,790,556,1081],[562,801,379,750],[699,828,238,506],[779,904,165,386]]
    read_video(in_path, finaloutpath, threshold,iou_threshold)
    make_video(finaloutpath, 'final_result_'+ vid_name)