#Adapted from: https://github.com/ondyari/FaceForensics
import os
import argparse
import cv2

def extract_frames(data_path, output_path,frames_extracted):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
      
    frame_num=0
    reader = cv2.VideoCapture(data_path)
    #Number of frames in video
    total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_step = total_frames // frames_extracted
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    for i in range(frames_extracted):
            
        # Set the frame position
        reader.set(cv2.CAP_PROP_POS_FRAMES, i * frames_step)
        # Read the frame
        success, image = reader.read()
        if not success:
            break
        cv2.imwrite(os.path.join(output_path, '{:04d}.png'.format(frame_num)),image)
        frame_num += 1
    reader.release()


def extract_method_videos(args):
    """Extracts all videos of a specified method"""

    if args.dataset=="FF_original":
        videos_path = os.path.join(args.data_dir_path,"FF++","original_sequences",args.compression,"videos")
        images_path = os.path.join(args.data_dir_path,"FF++","original_sequences",args.compression,"images")
    elif args.dataset.split("-")[0] =="Celeb":
        videos_path = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"videos")
        images_path = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"images")
    elif args.dataset =="YouTube-real":
        videos_path = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"videos")
        images_path = os.path.join(args.data_dir_path,"Celeb-DF",args.dataset,"images")
    elif args.dataset=="DeeperForensics":
        videos_path = os.path.join(args.data_dir_path,args.dataset,"manipulated_sequences","videos")
        images_path= os.path.join(args.data_dir_path,args.dataset,"manipulated_sequences","images")
    elif args.dataset=="ForgeryNet":

        videos_path = os.path.join(args.data_dir_path,args.dataset,"videos")
        images_path = os.path.join(args.data_dir_path,args.dataset,"images")
    else:
        videos_path = os.path.join(args.data_dir_path,"FF++","manipulated_sequences",args.dataset,args.compression,"videos")
        images_path = os.path.join(args.data_dir_path,"FF++","manipulated_sequences",args.dataset,args.compression,"images")
    all_videos = len(os.listdir(videos_path))
    video_iter=0
    for video in os.listdir(videos_path):

        
        print("Progress: {}/{} Current video {}".format(video_iter,all_videos,video),flush=True)
        image_folder = video.split('.')[0]
        os.makedirs(os.path.join(images_path,image_folder))
        extract_frames(os.path.join(videos_path, video),
                       os.path.join(images_path, image_folder),args.frames_extracted)
        video_iter+=1
        
if __name__ == '__main__':
    
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_dir_path', type=str, default = "/Data")
    p.add_argument("--dataset",choices=["FF_original","Deepfakes","Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures",
                                        "Celeb-real","Celeb-synthesis","YouTube-real","DeeperForensics",
                                        "ForgeryNet"])
    p.add_argument("--compression",choices=["c0","c23","c40"],type=str,default="c0")
    p.add_argument('--frames_extracted', type=int,default=40)
    args = p.parse_args()
    extract_method_videos(args)
