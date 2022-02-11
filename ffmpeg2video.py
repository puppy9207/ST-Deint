"""
    -- ffmpeg_video_numpy --
    ffmpeg-python 라이브러리를 이용한 numpy와 video 교환 프로그램

    영상을 numpy array로 바꾸거나, numpy array를 영상으로 바꿀수 있습니다.
    cuda encoding도 적용되며, A100 GPU 에서는 cuda encoding이 지원하지 않아서 이 예제에서는 사용하지 않았습니다.

    requirements )
    1. apt install ffmpeg
    2. pip install -r requirements.txt
"""

import os
import time

import ffmpeg
import numpy as np
import cv2
import torch
from model import Net as STNet
import torchvision.transforms as transforms

from tqdm import tqdm
from glob import glob

def video2np(file_name: str):
    """
        Video를 Numpy array로 변환시켜줍니다.
        
        file_name :
            Numpy array로 바꿀 Video 파일 경로를 넣어줍니다.

            Example)
                file_name = '/workspace/junseo/DALI/video/output2_SR.mp4'
    """

    # 비디오의 확장자들을 선언합니다.
    video_ext = (
        "mp4", "m4v", "mkv", "webm",
        "mov", "avi", "wmv", "mpg", "flv","m2t", "mxf","MXF"
    )

    # 파일이 존재 여부를 체크합니다.
    if not os.path.isfile(file_name):
        raise FileNotFoundError("해당 경로에 파일이 존재하지 않습니다.")

    # 파일이 비디오인지 체크합니다.
    if not file_name.endswith(video_ext):
        raise Exception("파일의 확장자가 Video 파일이 아닙니다.")

    # 비디오의 정보 값을 읽어 오기 위해 opencv의 VideoCapture를 불러옵니다.
    cap = cv2.VideoCapture(file_name)

    # 비디오를 열지 못했을 때 에러 처리
    if not cap.isOpened():
        raise Exception("Video 파일을 인식하지 못했습니다.")

    # 비디오의 총 프레임을 가져옵니다.
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 비디오의 FPS를 가져옵니다.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 비디오의 width 값과 height 값을 가져옵니다.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FFmpeg 프로세스를 생성 후 따로 작업하기 위해 async 선언과 출력에 접근하기 위해 pipe_stdout에 True 값을 줍니다.
    if ffmpeg.probe(file_name, select_streams="a")["streams"]:
        audio = ffmpeg.input(file_name).audio
    else:
        audio = None

    process = (
        ffmpeg
            .input(file_name)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
            .run_async(pipe_stdout=True)
    )

    # bytes를 stack 하기 위해 선언해줍니다.
    frame = bytearray()


    # length // 10 
    # 여기서 비디오를 이미지로 만드는 작업을 합니다. process의 stdout에 read를 해서 frame bytes들을 쌓아줍니다.
    # for i in range(10):
    #     for _ in tqdm(range(length//10)):
    #         in_bytes = process.stdout.read(height * width * 3)

    #         # bytes를 frame bytearray 에 extend 시켜 stack 합니다.
    #         frame.extend(in_bytes)
    #     frames = np.frombuffer(frame, np.uint8).reshape([-1, height, width, 3])
    #     np2video(frames, 'outvideo/king_of_comedy_AI.mp4',audio=audio, framerate=fps)

    for _ in tqdm(range(length)):
        in_bytes = process.stdout.read(height * width * 3)

        # bytes를 frame bytearray 에 extend 시켜 stack 합니다.
        frame.extend(in_bytes)
    frames = np.frombuffer(frame, np.uint8).reshape([-1, height, width, 3])

    return frames, audio, fps

def np2video(images: np.ndarray, file_name: str, framerate: float, audio, vcodec: str = 'libx264'):
    """
        Numpy array를 Video로 변환시켜줍니다.
        
        images :
            Video로 바꿀 Numpy array를 넣어줍니다.
            Numpy array의 shape는 NWHC 를 지켜야 합니다.

            Example)
                array shape => [300, 1080, 1920, 3]

        file_name :
            Numpy array로 바꿀 Video 파일 경로를 넣어줍니다.

            Example)
                file_name = '/workspace/junseo/DALI/video/output2_SR2.mp4'

        framerate :
            output Video의 fps를 설정합니다.
            
            Example)
                fps = 29.97

        vcodec :
            인코딩 할때 코덱을 설정합니다. defaults는 libx264 입니다.
    """

    # 이미지가 Numpy array가 아니면 Numpy array로 바꿔줍니다.
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)

    # 이미지의 정보를 가져옵니다.
    n, height, width, channels = images.shape

    # FFmpeg 프로세스를 생성 후 따로 작업하기 위해 async 선언과 입력에 접근하기 위해 pipe_stdin에 True 값을 줍니다.
    if audio:
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate)
                .output(audio, file_name, pix_fmt='yuv422p',acodec="copy", vcodec=vcodec, crf=0 ,loglevel="quiet")
                .overwrite_output() 
                .run_async(pipe_stdin=True)
        )
    else:
        process = ( 
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate)
                .output(file_name, pix_fmt='yuv422p', vcodec=vcodec, crf=0,loglevel="quiet")
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = STNet().to(device)
    net = torch.load("model_w_atten.pt")
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    totensor = transforms.ToTensor()
    # 여기서 이미지에서 비디오를 만드는 작업을 합니다. process의 stdin에 write를 하여 인코딩합니다.
    cnt = 0
    for frame in tqdm(images):
        in_st = time.time()
        lr = totensor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            sr = net(lr)
        restoreA, restoreB = sr
        restoreA = restoreA.mul(255).clamp(0,255).cpu().numpy().squeeze()
        restoreA = restoreA.transpose([1,2,0])
        restoreB = restoreB.mul(255).clamp(0,255).cpu().numpy().squeeze()
        restoreB = restoreB.transpose([1,2,0])
        outputA = []
        outputB = []
        for i in range(height):
            if i%2==0:
                outputA.append(restoreB[i//2,:,:])
                outputB.append(frame[i,:,:])
            else:
                outputA.append(frame[i,:,:])
                outputB.append(restoreA[i//2,:,:])

        outputA = np.asarray(outputA)
        outputB = np.asarray(outputB)
        outputA = np.ascontiguousarray(outputA,dtype=np.uint8)
        outputB = np.ascontiguousarray(outputB,dtype=np.uint8)
        process.stdin.write(outputB)
        # process.stdin.write(outputA)
        # print(f"time : {time.time()-in_st}")
        cnt +=1

    process.stdin.close()
    process.wait()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    indir = "invideo"
    outdir = "outvideo"
    for x in glob(f"{indir}/*"):
        print(x)
        start = time.time()
        file_name = os.path.basename(x).split(".")[0]
        video, audio, fps = video2np(x)
        np2video(video, f'{outdir}/{file_name}_AI.mov',audio=audio, framerate=fps)
        print(f"total time : {time.time()-start}")