import argparse
import numpy as np
import scipy
import sys
import cv2
import os
import time
import requests
import logging.handlers
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier
from collections import deque

# motpy関連
sys.path.append('./yolov7/motpy/motpy')
from core import Detection
from tracker import MultiObjectTracker
from testing_viz import draw_track

class MOT:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt=0.1)

    def track(self, outputs, ratio):
        if outputs is not None:
            outputs = outputs.cpu().numpy()
            outputs = [Detection(box=box[:4], score=box[4], class_id=box[5]) for box in outputs]
        else:
            outputs = []
        
        tracks = self.tracker.step(detections=outputs)
        return tracks

class LINE:
    def __init__(self):
        line_notify_token = 'nK7MB25wzCzft8klaefrJXrydnxfTnCJKFtjO4wUB9t'
        self.line_notify_api = 'https://notify-api.line.me/api/notify'
        self.headers = {'Authorization': f'Bearer {line_notify_token}'}
        
    def send_line(self, message, image=None):
        text = {'message': message}
        file = None
        if image != None:
            file = {'imageFile': open(str(image), "rb")}
        requests.post(self.line_notify_api, headers=self.headers, files=file, data=text)

def detect(opt, save_img=False):
    demo, source, weights, imgsz, view_img, save_img= opt.demo, opt.source, opt.weights, opt.img_size, opt.view_img, opt.save
    
    # ビデオ出力の縮小パラメーター
    resize = 1.5                #縮小倍率
    reductionRatio = 0.75       # 送信する際の縮小比率

    # デバッグするかどうか
    debug_index = True          # 出力結果を表示するかどうか

    # 速度算出パラメータ
    speed_index = -1            # 算出位置：下側→0、左側→1、右側→2
                                # 判定前初期値は-1で、自動的に判定される。
    speed_coefficent = 1.0      # 速度算出補正パラメータ
                                # 初期値は1.0であるが、キャリブレーションにより自動的に計算される。
    sensory_coefficient = 1.0   # 誘導員の速度感覚係数（90km/h→0.9）
    speed_ratio = 0.9           # 速度算出位置(画面比率)
    speed_limit = 50            # 最小速度
    calibration = 6             # キャリブレション実施回数
    max_calib_direction = 5     # 方向を算出する際の試行回数

    # 軌跡保存パラメータ
    MaxStep = 30        	    # 保存する同じトラッカーに対する最大ステップ数
    Nbox = 20           	    # 保存する違うトラッカーのBoxの数
    steplag = 10                # ステップlag
    
    # LINE Notify 出力設定
    index_LINE = False
    
    # Motpyの呼び出し
    mot = MOT()
    
    # LINE Notify
    if index_LINE:
        line = LINE()

    # 結果一時保存ディレクトリ
    save_dir = "./yolov7/result_box"

    # 初期化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half()  # to FP16

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 動画・Webカメラ
    if demo == "video" or demo == "webcam": 
        cap = cv2.VideoCapture(source if opt.demo == "video" else opt.camid)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 動画の幅（float）
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 動画の高さ（float）
    fps = int(cap.get(cv2.CAP_PROP_FPS))            # フレームレート
    
    if save_img:
        save_folder = save_dir
        os.makedirs(save_folder, exist_ok=True)
        if demo == "video":
            save_path = os.path.join(save_folder, os.path.splitext(source.split("/")[-1])[0] + '.mp4')
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width/resize), int(height/resize))
        )

    # 名前と色を取得する
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    # 軌跡を保存する配列を定義
    pts = [deque(maxlen=MaxStep) for _ in range(Nbox)]              #トラッカーの中心座標
    pts_velocity = [deque(maxlen=MaxStep) for _ in range(Nbox)]     #トラッカーの角座標
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
    
    while True:
        ret_val, im0s = cap.read()
        if ret_val:
            # 入力画像変換処理
            img = letterbox(im0s, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0 

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
            with torch.no_grad():   # 勾配を計算すると GPU メモリ リークが発生する
                pred = model(img, augment=opt.augment)[0]

            # NMS を適用する
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # 分類子を適用
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            
            det = pred[0]
            im0  =im0s

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
                for trc in tracks:
                    class_id = int(trc.class_id)
                    if class_id > 0:
                        # 上下左右の位置
                        top = int(trc.box[1])
                        bottom = int(trc.box[3])
                        left = int(trc.box[0])
                        right = int(trc.box[2])
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

                        # *************************************************
                        # 移動方向の判定
                        # motpyの性質上予測したボックスが動画のwidth>0,width<0
                        # height>0となるので、それを利用して移動方向の判定を行う
                        # ************************************************* 
                        if calib_count <= max_calib_direction:
                            for k in range(Nbox):
                                lIndex = False
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
                                            lIndex = True

                                        # 左側判定
                                        if xxx < 0 :
                                            directionCheck[1] += 1
                                            calib_count += 1
                                            directionBox[i]=1
                                            lIndex = True

                                        # 右側判定
                                        if xxx > width :
                                            directionCheck[2] += 1
                                            calib_count += 1
                                            directionBox[i]=1
                                            lIndex = True

                                        # 最終的な方向判定
                                        if calib_count == max_calib_direction:
                                            calib_count += 1
                                            max_value = max(directionCheck)
                                            speed_index = directionCheck.index(max_value)
                                        
                                        # LINENotify出力
                                        if index_LINE and lIndex:
                                            line.send_line("移動方向キャリブレーション中")
                                            lIndex = False
                                        
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
                                                        
                                                # LINENotify出力
                                                if index_LINE:
                                                    if skip_index:
                                                        cv2.imwrite('./yolov7/dest.jpg',img_cut)
                                                        if (detection_No < calibration):
                                                            line.send_line("速度キャリブレーション中", './yolov7/dest.jpg')
                                                        elif (detection_No >= calibration):
                                                            line.send_line(f'{str(speed)} km/h', './yolov7/dest.jpg')

                            # *************************************************
                            # 車両が右側から左側に移動する場合
                            # *************************************************
                            elif speed_index==1:      # 速度算出を左側で行う場合
                                if left < (width * (1.0-speed_ratio)):
                                    outCheck[i]=1
                                    # 速度の算出
                                    if (len(pts[i]) == MaxStep):
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
                                                
                                                # LINENotify出力
                                                if index_LINE:
                                                    if skip_index:
                                                        cv2.imwrite('./yolov7/dest.jpg',img_cut)
                                                        if (detection_No < calibration):
                                                            line.send_line("速度キャリブレーション中", './yolov7/dest.jpg')
                                                        elif (detection_No >= calibration):
                                                            line.send_line(f'{str(speed)} km/h', './yolov7/dest.jpg')
                            
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
                                                    img_cut = im0[top:bottom,left:right]
                                                    fx = img_cut.shape[1]
                                                    fy = img_cut.shape[0]
                                                    img_cut = cv2.resize(img_cut, (int(fx*reductionRatio), int(fy*reductionRatio)))
                                                    if debug_index:
                                                        cv2.imshow('sample.jpg',img_cut)
                                                
                                                # LINENotify出力
                                                if index_LINE:
                                                    if skip_index:
                                                        cv2.imwrite('./yolov7/dest.jpg',img_cut)
                                                        if (detection_No < calibration):
                                                            line.send_line("速度キャリブレーション中", './yolov7/dest.jpg')
                                                        elif (detection_No >= calibration):
                                                            line.send_line(f'{str(speed)} km/h', './yolov7/dest.jpg')

            # フレームの縮小
            output_frame = cv2.resize(im0,(int(width/resize), int(height/resize)))
            if save_img:
                vid_writer.write(output_frame)
                cv2.imshow('frame',output_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            if view_img:
                cv2.imshow('frame', output_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
        else:
            break

        # フレームで物体を検出していない場合
        for i in range(Nbox):
            if existCheck0[i] != existCheck1[i]:
                existCheck0[i] = 0
                IDStrings[i]=""
                for j in range(1, len(pts[i])):
                    pts[i] = deque(maxlen=MaxStep)
        existCheck1 = [0]*Nbox
        
def main(input, weight, view=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default="video")
    parser.add_argument('--weights', nargs='+', type=str, default=weight, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=input, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=view, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save', default=False, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', default=True, action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    detect(opt)
            
# main("../yolov7/inference/videos/sample01.mp4", "yolov7/weights/car_best.pt", view=True)