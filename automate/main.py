# -*- coding: utf-8 -*- 
import os
import sys
import argparse
import pathlib
# os.system('export GLOG_logtostderr=1')
# os.system('export GLOG_logtostderr=1 && bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
#     mediapipe/examples/desktop/hello_world:hello_world')




def main(input_data_path,output_data_path):
    comp='bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu'
    #명령어 컴파일
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \--calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt'
    #미디어 파이프 명령어 저장listfile
    listfile=os.listdir(input_data_path)
    if not(os.path.isdir(output_data_path+"Relative/")):
        os.mkdir(output_data_path+"Relative/")
    if not(os.path.isdir(output_data_path+"Absolute/")):
        os.mkdir(output_data_path+"Absolute/")
    for file in listfile:
        #해당 디렉토리의 하위 디렉토리 폴더명을 찾음
        if not(os.path.isdir(input_data_path+file)): #ignore .DS_Store
            continue
        word = file+"/"
        fullfilename=os.listdir(input_data_path+word)
        # 하위디렉토리의 모든 비디오들의 이름을 저장
        if not(os.path.isdir(output_data_path+"_"+word)):
            os.mkdir(output_data_path+"_"+word)
        if not(os.path.isdir(output_data_path+"Relative/"+word)):
            os.mkdir(output_data_path+"Relative/"+word)
        if not(os.path.isdir(output_data_path+"Absolute/"+word)):
            os.mkdir(output_data_path+"Absolute/"+word)
        os.system(comp)
        outputfilelist=os.listdir(output_data_path+'_'+word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue         
            inputfilen='   --input_video_path='+input_data_path+word+mp4list
            outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
            cmdret=cmd+inputfilen+outputfilen
            os.system(cmdret)

if __name__ == "__main__":
    cpath = pathlib.Path(__file__).parent.absolute()
    # print(cpath)
    input_data_path=str(cpath)+'/../put-videos-here/'
    output_data_path=str(cpath)+'/../temporary-folder/'
    # #print(input_data_path)
    main(input_data_path,output_data_path)