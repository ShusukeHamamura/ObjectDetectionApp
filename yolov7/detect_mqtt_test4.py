import argparse
import time
from pathlib import Path

import numpy as np
import scipy
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from collections import deque

# motpy
sys.path.append('../yolov7/motpy/motpy')
from core import Detection
from tracker import MultiObjectTracker
from testing_viz import draw_track

# MQTTによる送信関係
from multiprocessing.connection import answer_challenge
import paho.mqtt.client as mqtt     # MQTTのライブラリをインポート
import os
import sys
import time
import json
import logging
import logging.handlers
import ssl
import common
import boto3

# ブローカーに接続できたときの処理
def on_connect(client, userdata, flag, rc):
    print("Connected with result code " + str(rc))

# ブローカーが切断したときの処理
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")

# publishが完了したときの処理
def on_publish(client, userdata, mid):
    display_flag = False
    if display_flag:
        print("publish: {0}".format(mid))

# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# MQTT送信クラス
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
class mqtt_iotcore():

    def __init__(self):
        return

    def mqtt_open(self):
        print("mqtt_open: ")

        self.mqttc = mqtt.Client()
        self.mqttc.on_connect = on_connect         # 接続時のコールバック関数を登録
        self.mqttc.on_disconnect = on_disconnect   # 切断時のコールバックを登録
        self.mqttc.on_publish = on_publish         # メッセージ送信時のコールバック
        self.mqttc.tls_set(
            ca_certs=common.MQTT_SERVER_ROOTCAPATH, 
            certfile=common.MQTT_SERVER_CERTIFICATEPATH, 
            keyfile=common.MQTT_SERVER_PRIVATEKEYPATH, 
            cert_reqs=ssl.CERT_REQUIRED, 
            tls_version=ssl.PROTOCOL_TLSv1_2, 
            ciphers=None)
        self.mqttc.connect(common.MQTT_SERVER_HOST, common.MQTT_SERVER_PORT, common.MQTT_KEEPALIVE)
        self.mqttc.loop_start()    # subはloop_forever()だが，pubはloop_start()で起動だけさせる
        
        # 画像ファイルのアップロードの準備
        self.s3_cl = boto3.client(
            's3',
            aws_access_key_id=common.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=common.AWS_SECRET_ACCESS_KEY,
            region_name=common.REGION_NAME
        )
        time.sleep(1)
        return

    def mqtt_publish(self, send_json):
        self.mqttc.publish(common.MQTT_TOPIC_RESULT, json.dumps(send_json), 1)
        
    # MQTT出力
    def mqtt_output(self, detection_No, calibration, jsx_No, img_cut=False, speed=100, mqIndex=False):
        # 処理時間の生成
        Current_time = int(time.time())
        
        # 移動方向キャリブレーション中
        if mqIndex: 
            output_filename = 'direction.jpg'
        
        # 速度キャリブレーション中
        elif (detection_No < calibration):
            output_filename = 'velocity.jpg'
            speed = 100
        
        # 速度計測中
        elif (detection_No >= calibration):
            output_filename = 'dest.jpg'
            # 切り出し画像の保存
            cv2.imwrite(output_filename, img_cut)
            
        # 画像ファイルのアップロード
        s3_filename = jsx_No + "_" + str(Current_time) + ".jpg"
        self.s3_cl.upload_file(output_filename, common.BUCKET_NAME, s3_filename)

        pict_url = self.s3_cl.generate_presigned_url(
            ClientMethod = 'get_object',
            Params = {'Bucket' : common.BUCKET_NAME, 'Key' : s3_filename},
            ExpiresIn = common.S3_URL_TIMEOUT,
            HttpMethod = 'GET'
        )

        # JSONデータ作成
        send_json = {"message_kind": 11, "send_time": Current_time, "imei": jsx_No, "pict_time": Current_time, "speed": speed, "pict_url": pict_url}

        # mqtt publish
        self.mqtt_publish(send_json)
# <---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---<---

class MOT:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt=0.1)

    def track(self, outputs, ratio):
        if outputs is not None:
            outputs = outputs.cpu().numpy()
            # outputs = [Detection(box=box[:4] / ratio, score=box[4] * box[5], class_id=box[6]) for box in outputs]
            outputs = [Detection(box=box[:4], score=box[4], class_id=box[5]) for box in outputs]
        else:
            outputs = []
        
        tracks = self.tracker.step(detections=outputs)
        return tracks

def detect(save_img=False):
    demo, source, weights, view_img, save_txt, imgsz, trace = opt.demo, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt') 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Motpyの呼び出し
    mot = MOT()
    
    # ビデオ出力の縮小パラメーター
    resize = 1
    reductionRatio = 0.75
    # 送信する際の縮小比率

    # デバッグするかどうか
    debug_index = True          # True:デバッグ出力する、False:しない

    # 速度算出パラメータ
    speed_index = -1            # 算出位置：下側--->0、左側--->1、右側--->2
                                # 判定前初期値は-1で、自動的に判定される。
    speed_coefficent = 1.0      # 速度算出補正パラメータ
                                # 初期値は1.0であるが、キャリブレーションにより自動的に計算される。
    sensory_coefficient = 1.0   # 誘導員の速度感覚係数（90km/h--->0.9）
    speed_ratio = 0.9           # 速度算出位置の画面に対する比率
    speed_limit = 50            # 最小速度（この速度以下は処理しない）
    calibration = 11            # キャリブレション実施回数（実際に実施回数は、-1したもの）
    max_calib_direction = 5     # 方向を算出する際の試行回数

    # 軌跡保存パラメータ
    MaxStep = 30        	    # 保存する同じIDに対する最大ステップ数
    Nbox = 20           	    # 保存する違うIDのBoxの数
    steplag = 10                # ステップlag

    # MQTT出力設定
    index_MQTT = False
    jsx_No = 'jsx11'
    
    # 画像表示設定
    index_Display = True        # 画面に表示する--->True、表示しない--->False
    
    # MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->MQTT--->_
    #ログディレクトリ作成
    try:
        os.makedirs(common.LOG_DIR)
    except FileExistsError:
        pass

    #ログ設定
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    handler = logging.handlers.RotatingFileHandler(filename=common.LOG_FILE_NAME, maxBytes=1000000, backupCount=9)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logging.info('=======================================')
    logging.info('Start send AI result')
    
    # MQTT Connect
    if index_MQTT:
        ai_guide_mqtt = mqtt_iotcore()
        try:
            ai_guide_mqtt.mqtt_open()
        except:
            logging.error('  Error end')
            del ai_guide_mqtt
            time.sleep(3)
            sys.exit()
    # MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---MQTT<---
    # Motpyの呼び出し
    mot = MOT()

    # 結果保存ディレクトリ
    if save_img:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 初期化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # CUDA でのみサポートされる半精度

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # 2 段階の分類器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
    # 動画,Webカメラの取得
    # Webカメラの場合
    if demo == "video" or demo == "webcam": 
        cap = cv2.VideoCapture(source if opt.demo == "video" else opt.camid)
    
    # ネットワークカメラの場合
    elif demo == "ipcam":
        # vivotekカメラの場合の設定
        cap = cv2.VideoCapture('rtsp://root:abit1900@192.168.1.120/live1s2.sdp')

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 動画の幅（float）
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 動画の高さ（float）
    fps = int(cap.get(cv2.CAP_PROP_FPS))            # フレームレート
    
    if save_img:
        save_folder = save_dir
        os.makedirs(save_folder, exist_ok=True)
        if demo == "video":
            save_path = os.path.join(save_folder, os.path.splitext(source.split("/")[-1])[0] + '.mp4')
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width/resize), int(height/resize))
        )

    # 名前と色を取得する
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    # 軌跡を保存する配列を定義
    pts = [deque(maxlen=MaxStep) for _ in range(Nbox)]
    pts_velocity = [deque(maxlen=MaxStep) for _ in range(Nbox)]
    IDStrings = [""]*Nbox       # IDの最初の8文字を格納した配列
    existCheck0 = [0]*Nbox      # IDが存在するかを保存しておく配列
    existCheck1 = [0]*Nbox      # フレームごとにIDが存在するかを保存しておく配列
    outCheck = [0]*Nbox         # 出力したかどうかのチェック
    directionCheck = [0]*3      # 移動方向の算定
    directionBox = [0]*Nbox     # 方向チェック済みかどうか
    ave_speed = np.zeros(calibration-1)

    detection_No = -3       # キャリブレーションで最初に破棄する枚数
    calib_count = 0         # キャリブレーションのときのカウント
    average_speed = 0.0     # 速度補正の際の平均速度

    t0 = time.time()
    
    # 車両の軌跡を描画するための処理
    _, first_img = cap.read()
    
    while True:
        ret_val, im0s = cap.read()
        if ret_val:
            width = WIDTH
            height = HEIGHT
            # 入力画像変換処理
            # Padded resize
            img = letterbox(im0s, 640, stride=32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            ratio = min(int(imgsz) / im0s.shape[0], int(imgsz) / im0s.shape[1])
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 準備
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # 推論
            t1 = time_synchronized()
            with torch.no_grad():   # 勾配を計算すると GPU メモリ リークが発生する
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # NMS を適用する
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # 分類子を適用
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            
            det = pred[0]
            im0  =im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # ボックスを img_size から im0 サイズにリスケール
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # 結果描画
                for *xyxy, conf, cls in reversed(det): #xyxy:bbox座標、conf:bbox信頼度、cls:bboxクラス
                    if int(cls) != 0:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                # motpy処理
                tracks = mot.track(det, ratio)
                # motpy結果描画
                for trc in tracks:
                    class_id = int(trc.class_id)
                    if class_id > 0:
                        # 上下左右の位置
                        top = int(trc.box[1])
                        bottom = int(trc.box[3])
                        left = int(trc.box[0])
                        right = int(trc.box[2])
                        boxsize = (abs(bottom-top),abs(left-right))
                        # IDの最初の８文字を切り出す
                        string = trc.id[:8]
                        draw_track(im0, trc, ratio, thickness=2)
                        # Boxの中心位置の計算
                        xx = int((left+right)/2.0)
                        yy = int((top+bottom)/2.0)
                        center = (xx,yy)

                        # 速度算出位置の計算（最初は中央で行う）
                        if speed_index==-1:     # 初期状態
                            vel_point = center
                        if speed_index==0:      # 速度算出を下側で行う場合
                            vel_point = (int((right+left)/2),int(bottom))
                        elif speed_index==1:    # 速度算出を左側で行う場合
                            vel_point = (int(left),int(bottom))
                        elif speed_index==2:    # 速度算出を右側で行う場合
                            vel_point = (int(right),int(bottom))

                        i=-1
                        existIndex = False
                        # 同じIDのものがあるかどうかを捜す
                        for id in IDStrings:
                            i += 1
                            if existCheck0[i]==1:
                                # 同じものがあったとき
                                if id == string:
                                    existCheck0[i]=1
                                    existCheck1[i]=1
                                    existIndex = True
                                    # 一方向だけを保存
                                    pts[i].append(center)
                                    pts_velocity[i].append(vel_point)
                                    break
                        # 同じIDのものが無かったとき
                        if not existIndex:
                            # 空いている番号を探す
                            for k in range(Nbox):
                                if existCheck0[k] ==0 :
                                    break
                            # 空いている番号のところにIDを記載する
                            IDStrings[k] = string
                            existCheck0[k]=1
                            existCheck1[k]=1
                            outCheck[k]=0
                            pts[k].append(center)
                            pts_velocity[k].append(vel_point)
                            directionBox[k]=0
                        # 軌跡を描画(MaxStepまでの範囲）
                        for j in range(1, len(pts[i])):
                            if pts[i][j - 1] is None or pts[i][j] is None:
                                continue
                            thick = int(np.sqrt(64 / float(j + 1)) )
                            cv2.line(im0,(pts[i][j-1]), (pts[i][j]),color=(255, 0, 0),thickness=thick)
                            
                            # 軌跡の保存
                            if calib_count >= max_calib_direction:
                                cv2.circle(first_img, center, 2, (0, 0, 255), thickness=2)

                        # *************************************************
                        # 移動方向の判定
                        # ************************************************* 
                        if calib_count <= max_calib_direction:
                            for k in range(Nbox):
                                mqIndex = False
                                if directionBox[k]==0:
                                    j=len(pts_velocity[k])-1
                                    if j>=0:
                                        # 移動方向
                                        xxx = pts_velocity[k][j][0]
                                        yyy = pts_velocity[k][j][1]

                                        # 下側判定
                                        if yyy > height :
                                            directionCheck[0] += 1
                                            calib_count += 1
                                            directionBox[i]=1
                                            mqIndex = True

                                        # 左側判定
                                        if xxx < 0 :
                                            directionCheck[1] += 1
                                            calib_count += 1
                                            directionBox[i]=1
                                            mqIndex = True

                                        # 右側判定
                                        if xxx > width :
                                            directionCheck[2] += 1
                                            calib_count += 1
                                            directionBox[i]=1
                                            mqIndex = True

                                        # 最終的な方向判定
                                        if calib_count == max_calib_direction:
                                            calib_count += 1
                                            max_value = max(directionCheck)
                                            speed_index = directionCheck.index(max_value)
                                        
                                        # MQTT出力
                                        if index_MQTT and mqIndex:
                                            ai_guide_mqtt.mqtt_output(detection_No, calibration, jsx_No, mqIndex=mqIndex)
                            
                        
                        # 速度、画像等の出力
                        if outCheck[i]==0:
                             # *************************************************
                            # 車両が上側から下側に移動する場合
                            # *************************************************
                            if speed_index==0:      # 速度算出を下側で行う場合
                                if bottom > (height * speed_ratio):
                                    # outCheck[i]=1
                                    # 速度の算出
                                    if len(pts[i]) == MaxStep:
                                        # 上から下に移動しているもののみを抽出
                                        if (pts[i][0][1] < pts[i][int(MaxStep/2)][1]) and (pts[i][int(MaxStep/2)][1] < pts[i][MaxStep-1][1]):
                                            detection_No += 1
                                            x1 = pts_velocity[i][MaxStep-steplag][0]
                                            x0 = pts_velocity[i][0][0]
                                            y1 = pts_velocity[i][MaxStep-steplag][1]
                                            y0 = pts_velocity[i][0][1]
                                            velocity = ((x1-x0)**2+(y1-y0)**2)**0.5
                                            speed = int(velocity/speed_coefficent)
                                            # 平均速度を算出し、補正係数を求める
                                            if (detection_No > 0) and (detection_No < calibration):
                                                ave_speed[detection_No - 1] = speed
                                            # トリム平均で平均速度を求める
                                            if detection_No == calibration-1:
                                                average_speed = scipy.stats.trim_mean(ave_speed, 0.1)
                                                speed_coefficent = average_speed/(sensory_coefficient*100)

                                            # 誤検出で規定の速度以下の場合は処理を行わない
                                            if speed > speed_limit:
                                                # 速度の画面表示
                                                if debug_index:
                                                    if (detection_No < 1):
                                                        print("対象外速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No < calibration):
                                                        print("No.{:>2d}".format(detection_No),end="   ")
                                                        print("速度キャリブレーション中",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No >= calibration):
                                                        print("補正速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))

                                                # 画像範囲はみ出し処理と画像の切り出し
                                                if top < 0:
                                                    top=0
                                                if bottom > height:
                                                    bottom=int(height)
                                                if left < 0:
                                                    left=0
                                                if right > width:
                                                    right = int(width)
                                                skip_index = False
                                                if (left < right) and (top < bottom):
                                                    skip_index = True
                                                    img_cut = im0[top:bottom,left:right]
                                                    fx = img_cut.shape[1]
                                                    fy = img_cut.shape[0]
                                                    img_cut = cv2.resize(img_cut, (int(fx*reductionRatio), int(fy*reductionRatio)))
                                                    if debug_index:
                                                        cv2.imshow('sample.jpg',img_cut)
                                                # MQTT出力
                                                if index_MQTT:
                                                    if skip_index:
                                                        ai_guide_mqtt.mqtt_output(detection_No, calibration, jsx_No, s3_cl, img_cut, speed)

                            # *************************************************
                            # 車両が右側から左側に移動する場合
                            # *************************************************
                            elif speed_index==1:      # 速度算出を左側で行う場合
                                if left < (width * (1.0-speed_ratio)):
                                    outCheck[i]=1
                                    # 速度の算出
                                    if len(pts[i]) == MaxStep:
                                        # 右から左に移動しているもののみを抽出
                                        if (pts[i][0][0] > pts[i][int(MaxStep/2)][0]) and (pts[i][int(MaxStep/2)][0] > pts[i][MaxStep-1][0]):
                                            detection_No += 1
                                            x1 = pts_velocity[i][MaxStep-steplag][0]
                                            x0 = pts_velocity[i][0][0]
                                            y1 = pts_velocity[i][MaxStep-steplag][1]
                                            y0 = pts_velocity[i][0][1]
                                            velocity = ((x1-x0)**2+(y1-y0)**2)**0.5
                                            speed = int(velocity/speed_coefficent)
                                            # 平均速度を算出し、補正係数を求める
                                            if (detection_No > 0) and (detection_No < calibration):
                                                ave_speed[detection_No - 1] = speed
                                            # トリム平均で平均速度を求める
                                            if detection_No == calibration-1:
                                                average_speed = scipy.stats.trim_mean(ave_speed, 0.1)
                                                speed_coefficent = average_speed/(sensory_coefficient*100)

                                            # 誤検出で規定の速度以下の場合は処理を行わない
                                            if speed > speed_limit:
                                                # 速度の画面表示
                                                if debug_index:
                                                    if (detection_No < 1):
                                                        print("対象外速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No < calibration):
                                                        print("No.{:>2d}".format(detection_No),end="   ")
                                                        print("速度キャリブレーション中",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No >= calibration):
                                                        print("補正速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))

                                                # 画像範囲はみ出し処理と画像の切り出し
                                                if top < 0:
                                                    top=0
                                                if bottom > height:
                                                    bottom=int(height)
                                                if left < 0:
                                                    left=0
                                                if right > width:
                                                    right = int(width)
                                                skip_index = False
                                                if (left < right) and (top < bottom):
                                                    skip_index = True
                                                    img_cut = im0[top:bottom,left:right]
                                                    fx = img_cut.shape[1]
                                                    fy = img_cut.shape[0]
                                                    img_cut = cv2.resize(img_cut, (int(fx*reductionRatio), int(fy*reductionRatio)))
                                                    if debug_index:
                                                        cv2.imshow('sample.jpg',img_cut)
                                                # MQTT出力
                                                if index_MQTT:
                                                    if skip_index:
                                                        ai_guide_mqtt.mqtt_output(detection_No, calibration, jsx_No, s3_cl, img_cut, speed)
                            
                            # *************************************************
                            # 車両が左側から右側に移動する場合
                            # *************************************************
                            elif speed_index==2:      # 速度算出を右側で行う場合
                                if (right > width * speed_ratio) : # motpyでのbboxの右側が画面のspeed_ratio割を超えた場合
                                    outCheck[i]=1
                                    # 速度の算出
                                    if len(pts[i]) == MaxStep:
                                        # 左から右に移動しているもののみを抽出
                                        if (pts[i][0][0] < pts[i][int(MaxStep/2)][0]) and (pts[i][int(MaxStep/2)][0] < pts[i][MaxStep-1][0]):
                                            detection_No += 1
                                            x1 = pts_velocity[i][MaxStep-steplag][0]
                                            x0 = pts_velocity[i][0][0]
                                            y1 = pts_velocity[i][MaxStep-steplag][1]
                                            y0 = pts_velocity[i][0][1]
                                            velocity = ((x1-x0)**2+(y1-y0)**2)**0.5
                                            # 平均速度を算出し、補正係数を求める
                                            speed = int(velocity/speed_coefficent)
                                            if (detection_No > 0) and (detection_No < calibration):
                                                ave_speed[detection_No - 1] = speed
                                            # トリム平均で平均速度を求める
                                            if detection_No == calibration-1:
                                                average_speed = scipy.stats.trim_mean(ave_speed, 0.1)
                                                speed_coefficent = average_speed/(sensory_coefficient*100)

                                            # 誤検出で規定の速度以下の場合は処理を行わない
                                            if speed > speed_limit:
                                                # 速度の画面表示
                                                if debug_index:
                                                    if (detection_No < 1):
                                                        print("対象外速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No < calibration):
                                                        print("No.{:>2d}".format(detection_No),end="   ")
                                                        print("速度キャリブレーション中",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                    elif (detection_No >= calibration):
                                                        print("補正速度",end="   ")
                                                        print("{:<10s}".format(names[int(cls)]),end = "  ")
                                                        print("速度＝{:>3d}".format(speed))
                                                # outCheck[i]=1
                                                # 画像範囲はみ出し処理
                                                if top < 0:
                                                    top=0
                                                if bottom > height:
                                                    bottom=int(height)
                                                if left < 0:
                                                    left=0
                                                if right > width:
                                                    right = int(width)
                                                skip_index = False
                                                if (left < right) and (top < bottom):
                                                    skip_index = True
                                                    # print(frame.shape)
                                                    # print(top, bottom, left, right)
                                                    img_cut = im0[top:bottom,left:right]
                                                    fx = img_cut.shape[1]
                                                    fy = img_cut.shape[0]
                                                    img_cut = cv2.resize(img_cut, (int(fx*reductionRatio), int(fy*reductionRatio)))
                                                    if debug_index:
                                                        cv2.imshow('sample.jpg',img_cut)

                                                # MQTT出力
                                                if index_MQTT:
                                                    if skip_index:
                                                        ai_guide_mqtt.mqtt_output(detection_No, calibration, jsx_No, img_cut, speed)

            # ビデオ出力の際のフレームの縮小
            output_frame = cv2.resize(im0,(int(width/resize), int(height/resize)))
            # ファイル出力する場合
            if save_img:
                vid_writer.write(output_frame)
                if index_Display:
                    cv2.imshow('frame',output_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            # ファイル出力しない場合
            else:
                if index_Display:
                    cv2.imshow('frame', output_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            
            cv2.imwrite('region_sample.jpg', first_img)
        
        else:
            break
        # ***************************************
        # フレームで、物体を検出していない場合
        # ***************************************
        for i in range(Nbox):
            if existCheck0[i] != existCheck1[i]:
                existCheck0[i] = 0
                IDStrings[i]=""
                for j in range(1, len(pts[i])):
                    pts[i] = deque(maxlen=MaxStep)
        existCheck1 = [0]*Nbox
        # --------------------------------------------------------------------------

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default="video", help="demo type, eg. image, video and webcam")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id") # WebカメラのカメラID
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
