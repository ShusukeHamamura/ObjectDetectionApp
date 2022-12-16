# Object Detection App(For submission)
 
yolov7を用いた画像・動画・Webカメラでの物体検出が行えます。

また物体検出と物体追跡を用いて車両の速度検知を行えます。
 
# DEMO
 
 画像・動画での物体検出
 
https://user-images.githubusercontent.com/106325569/207815994-b5fa62d2-517f-4efc-9d7c-7e78bd8a51bf.mp4
 
 車両の速度検知
 
https://user-images.githubusercontent.com/106325569/207821132-f753fe84-2060-4869-8cf4-2c1137e60b07.mp4

# Note
 
 物体検出にはyolov7[1]、物体追跡にはmotpy[2]を使用しています。
 
 主に私が書いたコードはApp.py、yolov7/detect_app.py、yolov7/detect_speed.pyとなっております。
 
 yolov7/detect_app.py、yolov7/detect_speed.pyに関しては既存のyolov7/detect.pyを元に作成いたしました。
 
注目してもらいたいプログラムはApp.pyとyolov7/detect_speed.pyです。

長いので特にdetect_speed.pyでは106行からのdetect関数内の処理に注目してもらいたいです。

## aa
aaaaaa
 
# Requirement
  ```bash
 pip install -r requirements.txt
```
# Usage
 ```bash
 python App.py
``` 
 
# Author
 
* 濵村秀亮
* hama.i051ff@gmail.com

# Reference

[1] https://github.com/WongKinYiu/yolov7.git

[2] https://github.com/wmuron/motpy.git
