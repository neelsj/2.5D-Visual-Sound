
from pytube import YouTube
# misc
import os
import shutil
import math
import datetime

import numpy as np
import cv2

from random import sample 

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')
        
    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        #print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')
        
    def get_n_for_fps(self, target_fps):
        every_x_frame = round(self.fps/target_fps)
        self.get_n_images(every_x_frame)
        return every_x_frame

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
        
        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():
            
            success,image = self.vid_cap.read() 
            
            if not success:
                break
            
            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ('%06d' % img_cnt) + img_ext)
                cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 80])  
                img_cnt += 1
                
            frame_cnt += 1
        
        self.vid_cap.release()

    def extract_specific_frames(self, timestamps, img_name, dest_path=None, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
       
        frameNums = [int(round(t*self.fps)) for t in timestamps]

        for n in frameNums:

            self.vid_cap.set(1, n)
            success,image = self.vid_cap.read() 
            
            if not success:
                break
            
            img_path = os.path.join(dest_path, ''.join([img_name, '_', str(n), img_ext]))
            cv2.imwrite(img_path, image)  
                        
        self.vid_cap.release()

def Extract(outputPath, fileName):

    try:

        fileNameBase = os.path.splitext(os.path.split(fileName)[1])[0]
        fullOutputPath = outputPath + fileNameBase

        frameTimestamps = []
        fe = FrameExtractor(fileName)

        every_x_frame = fe.get_n_for_fps(10)
        fe.extract_frames(every_x_frame, "", dest_path=fullOutputPath, img_ext = '.jpg')
     
    except Exception as e:
        print(e)
        pass

if __name__ == "__main__":

    inputPath='E:/Research/Videos/FAIR-Play/videos/'
    outputPath='E:/Research/Videos/FAIR-Play/frames/'

    filenames = os.listdir(inputPath)
    files = [os.path.join(inputPath, f) for f in filenames]

    #file = sample(files, 200)

    files = tqdm(files)

    if (not os.path.isdir(outputPath)):
        os.mkdir(outputPath)

    #for f in files:
    #    Extract(outputPath, f)

    num_cores = int(multiprocessing.cpu_count()*3/4)
    Parallel(n_jobs=num_cores)(delayed(Extract)(outputPath, f) for f in files)

