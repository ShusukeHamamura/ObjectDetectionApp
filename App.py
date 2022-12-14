from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import configparser, os, webbrowser, sys
import threading
import os
import shutil

sys.path.append('./yolov7')
import detect_app
import detect_speed

class Application:
    
    output_canvas = None
    
    def __init__(self, root):
        root.title('物体検出アプリケーション')
        self.file_name = StringVar()
        self.file_name.set('画像または動画を選択してください')
        label1 = Label(textvariable=self.file_name, font=('', 20))
        label1.pack(pady=10)

        #ウィンドウのサイズについて
        self.clientHeight = '700'
        self.clientWidth = '800'
        # ウィンドウサイズ可変
        # cp = configparser.ConfigParser()
        # try:
        #     cp.read(self.__class__.__name__ + '.ini')
        #     self.clientHeight = cp['Client']['Height']
        #     self.clientWidth = cp['Client']['Width']
        #     self.directory = cp['File']['Directory']
        # except:
        #     print(self.__class__.__name__ + ':Use default value(s)', file=sys.stderr)
        root.geometry(self.clientWidth + 'x' + self.clientHeight)
        root.protocol('WM_DELETE_WINDOW', self.menuFileExit)
        
        # 画像表示の場所指定とサイズ指定
        # 入力画像設定
        self.input_canvas = Canvas(root, width=int(self.clientWidth)-50, height=int(self.clientWidth)-50)
        self.input_canvas.place(x=25, y=60)

        #メニューバー表示
        root.option_add('*tearOff', FALSE)
        menu = Menu(root)
        self.menuFile = Menu(menu)
        menu.add_cascade(menu=self.menuFile, label='ファイル(F)', underline=5)
        self.menuFile.add_command(label='開く(O)', underline=3, command=self.menuFileOpen)
        self.menuFile.add_separator()
        self.menuFile.add_command(label='終了(X)', underline=3, command=self.menuFileExit)

        self.menuRun = Menu(menu)
        menu.add_cascade(menu=self.menuRun, label='実行(R)', underline=3)
        self.menuRun.add_command(label='画像認識(Y)', underline=10, command=self.yolo_image)
        self.menuRun.add_separator()
        self.menuRun.add_command(label='動画認識(Y)', underline=10, command=self.yolo_video1)
        self.menuRun.add_separator()
        self.menuRun.add_command(label='Webカメラ認識', underline=10, command=self.yolo_webcam1)
        self.menuRun.add_separator()
        self.menuRun.add_command(label='車両速度検知', underline=10, command=self.speed_detection1)

        self.menuHelp = Menu(menu)
        menu.add_cascade(menu=self.menuHelp, label='オプション')
        self.menuHelp.add_command(label='ファイルの確認', command=self.menuHelpConfirmation)
        self.menuHelp.add_separator()
        self.menuHelp.add_command(label='検出結果の保存', underline=8, command=self.save_result)
        self.menuHelp.add_separator()
        self.menuHelp.add_command(label='バージョン情報(V)', underline=8, command=self.menuHelpVersion)
        
        root['menu'] = menu
    
    def yolo_image(self):
        if self.extension==".jpg" or self.extension==".png":
            root.title('物体検出中')
            detect_app.main(self.select_file_name, "yolov7/weights/yolov7.pt")
            root.title('物体検出アプリケーション')
            img = Image.open("yolov7/result_box/result.jpg")
            w = img.width
            h = img.height
            can_size = int(self.clientWidth)-50
            ratio = min(can_size / w, can_size / h)
            self.input_image = img.resize((round(ratio * w), round(ratio * h)))
            self.input_image = ImageTk.PhotoImage(self.input_image)
            self.input_canvas.create_image(0, 0, image=self.input_image, anchor=NW)
        else:
            messagebox.showinfo('エラー', "画像ファイルを選択してください")
    
    def yolo_video1(self):
        thread = threading.Thread(target=self.yolo_video2)
        thread.start()
        
    def yolo_video2(self):
        if self.extension==".mp4" or self.extension==".mov":
            root.title('物体検出中')
            detect_app.main(self.select_file_name, "yolov7/weights/yolov7.pt", view=True)
            root.title('物体検出アプリケーション')
        else:
            messagebox.showinfo('エラー', "動画ファイルを選択してください")
    
    def yolo_webcam1(self):
        thread = threading.Thread(target=self.yolo_webcam2)
        thread.start()
    
    def yolo_webcam2(self):
        root.title('物体検出中')
        detect_app.main("0", "yolov7/weights/yolov7.pt", view=True)
        root.title('画像認識アプリケーション')
    
    def speed_detection1(self):
        thread = threading.Thread(target=self.speed_detection2)
        thread.start()
    
    def speed_detection2(self):
        if self.extension==".mp4" or self.extension==".mov":
            root.title('速度検知中')
            self.file_name.set('結果はLINEへ送信されています')
            detect_speed.main(self.select_file_name, "yolov7/weights/car_best.pt", view=True)
            root.title('物体検出アプリケーション')
            self.file_name.set('ファイルが選択されています')
        else:
            messagebox.showinfo('エラー', "動画ファイルを選択してください")
    
    def save_result(self):
        try:
            iDir = os.path.abspath(os.path.dirname(__file__))
            self.save_filename = filedialog.asksaveasfilename(initialdir=iDir,
                                                            title = "名前を付けて保存",
                                                            filetypes = [("JPEG", ".jpg"), ("MP4", ".mp4")],
                                                            defaultextension = " "
                                                            )
            filetype = os.path.splitext(self.save_filename)[1]
            shutil.copyfile("./yolov7/result_box/result" + filetype, self.save_filename)
            messagebox.showinfo('info', '出力結果を保存しました')
        except:
            messagebox.showinfo('info', '出力結果の保存をキャンセルしました')

    def menuFileOpen(self):
        iDir = os.path.abspath(os.path.dirname(__file__))
        self.select_file_name = filedialog.askopenfilename(initialdir=iDir)
        self.extension = os.path.splitext(self.select_file_name)[1]
        filetypes = [".jpg", ".png", ".mp4", ".mov"]
        if self.extension in filetypes:
            self.file_name.set("ファイルが選択されています")
            img = Image.open(self.select_file_name)
            w = img.width
            h = img.height
            can_size = int(self.clientWidth)-50
            ratio = min(can_size / w, can_size / h)
            self.input_image = img.resize((round(ratio * w), round(ratio * h)))
            self.input_image = ImageTk.PhotoImage(self.input_image)
            self.input_canvas.create_image(0, 0, image=self.input_image, anchor=NW)
        else :
            messagebox.showinfo('エラー', "画像または動画ファイルを選択してください")

    def menuFileExit(self):
        cp = configparser.ConfigParser()
        cp['Client'] = { 'Height': str(root.winfo_height()),
                        'Width': str(root.winfo_width())}
        with open(self.__class__.__name__ + '.ini', 'w') as f:
            cp.write(f)
        root.destroy()

    def menuHelpOpenWeb(self):
        messagebox.showinfo('エラー', "実装中")

    def menuHelpConfirmation(self):
        s = 'ファイル名：\n'
        try:
            if len(self.select_file_name) == 0:
                s += 'ファイルが選択されていません'
            else:
                s += self.select_file_name
        except:
            s += 'ファイルが選択されていません'
        messagebox.showinfo('ファイルの確認', s)

    def menuHelpVersion(self):
        messagebox.showinfo('バージョン情報', 'version 1')

if __name__ == '__main__':
    root = Tk()
    Application(root)
    root.mainloop()