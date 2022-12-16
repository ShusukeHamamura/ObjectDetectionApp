# 物体検出アプリ
 
yolov7を用いた画像・動画・Webカメラでの物体検出が行えます。

また物体検出と物体追跡を用いて車両の速度検知を行えます。
 
# DEMO
 
 画像・動画での物体検出
 
https://user-images.githubusercontent.com/106325569/207815994-b5fa62d2-517f-4efc-9d7c-7e78bd8a51bf.mp4
 
 車両の速度検知
 
https://user-images.githubusercontent.com/106325569/207821132-f753fe84-2060-4869-8cf4-2c1137e60b07.mp4

# Features
 
 物体検出にはyolov7(https://github.com/WongKinYiu/yolov7.git)
 
 物体追跡にはmotpy(https://github.com/wmuron/motpy.git)
 
 を使用しました。
 
 主に私が書いたコードはApp.py、yolov7/detect_app.py、yolov7/detect_speed.pyとなっております。
 
 yolov7/detect_app.py、yolov7/detect_speed.pyに関しては既存のyolov7/detect.pyを元に作成いたしました。
 
注目してもらいたいプログラムはApp.pyとyolov7/detect_speed.pyです。

特にdetect_speed.pyでは106行からのdetect関数内の処理に注目してもらいたいです。
 
# Requirement
  ```bash
 pip install -r requirements.txt
```
# Usage
 ```bash
 python App.py
``` 
# Note
 
注意点などがあれば書く
 
# Author
 
作成情報を列挙する
 
* 濵村秀亮
* 所属
* E-mail
 
# License

