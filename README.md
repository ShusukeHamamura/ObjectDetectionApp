# 物体検出アプリ
 
yolov7を用いた画像・動画・Webカメラでの物体検出が行えます

また物体検出と物体追跡を用いて車両の速度検知を行えます
 
# DEMO
 
"hoge"の魅力が直感的に伝えわるデモ動画や図解を載せる
 
# Features
 
 物体検出にはyolov7(https://github.com/WongKinYiu/yolov7.git)
 
 物体追跡にはmotpy(https://github.com/wmuron/motpy.git)
 
 を使用しました
 
 主に私が書いたコードはApp.py、yolov7/detect_app.py、yolov7/detect_speed.pyとなっております
 
 yolov7/detect_app.py、yolov7/detect_speed.pyに関しては既存のdetect.pyを元に作成いたしました
 
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
ライセンスを明示する
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
社内向けなら社外秘であることを明示してる
 
"hoge" is Confidential.
