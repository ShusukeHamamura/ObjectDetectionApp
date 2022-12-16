from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import configparser, os, webbrowser, sys
import threading
import os
import shutil
import cv2

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
        self.clientHeight = '750'
        self.clientWidth = '700'
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
        root.protocol('WM_DELETE_WINDOW', self.menuExit)
        
        # 画像表示の場所指定とサイズ指定
        # 入力画像設定
        self.input_canvas = Canvas(root, width=int(self.clientWidth)-50, height=int(self.clientWidth)-50)
        self.input_canvas.place(x=25, y=60)

        #メニューバー表示
        root.option_add('*tearOff', FALSE)
        menu = Menu(root)
        self.menuFile = Menu(menu)
        menu.add_cascade(menu=self.menuFile, label='ファイル', underline=5)
        self.menuFile.add_command(label='開く', underline=3, command=self.menuFileOpen)
        self.menuFile.add_separator()
        self.menuFile.add_command(label='アプリの終了', underline=3, command=self.menuExit)

        self.menuRun = Menu(menu)
        menu.add_cascade(menu=self.menuRun, label='実行', underline=3)
        self.menuRun.add_command(label='画像で検知', underline=10, command=lambda : self.Mythread(self.yolo_image))
        self.menuRun.add_separator()
        self.menuRun.add_command(label='動画で検知', underline=10, command=lambda: self.Mythread(self.yolo_video))
        self.menuRun.add_separator()
        self.menuRun.add_command(label='Webカメラで検知', underline=10, command=lambda: self.Mythread(self.yolo_webcam))
        self.menuRun.add_separator()
        self.menuRun.add_command(label='車両速度検知', underline=10, command=lambda: self.Mythread(self.speed_detection))

        self.menuHelp = Menu(menu)
        menu.add_cascade(menu=self.menuHelp, label='オプション')
        self.menuHelp.add_command(label='ファイルの確認', command=self.menuHelpConfirmation)
        self.menuHelp.add_separator()
        self.menuHelp.add_command(label='検出結果の保存', underline=8, command=self.save_result)
        self.menuHelp.add_separator()
        self.menuHelp.add_command(label='バージョン情報', underline=8, command=self.menuHelpVersion)
        self.menuHelp.add_separator()
        self.menuHelp.add_command(label='github', underline=8, command=self.menuHelpOpenWeb)
        
        root['menu'] = menu
        
    def Mythread(self, func):
        thread = threading.Thread(target=func)
        thread.start()
    
    def yolo_image(self):
        try:
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
                messagebox.showerror('error', "画像ファイルを選択してください")
        except:
            messagebox.showerror('error', "ファイルが選択されていません")
        
    def yolo_video(self):
        try:
            if self.extension==".mp4" or self.extension==".mov":
                root.title('物体検出中')
                detect_app.main(self.select_file_name, "yolov7/weights/yolov7.pt", view=True)
                root.title('物体検出アプリケーション')
            else:
                messagebox.showerror('error', "動画ファイルを選択してください")
        except:
            messagebox.showerror('error', "ファイルが選択されていません1111")
    
    def yolo_webcam(self):
        root.title('物体検出中')
        detect_app.main("0", "yolov7/weights/yolov7.pt", view=True)
        root.title('画像認識アプリケーション')
    
    def speed_detection(self):
        try:
            if self.extension==".mp4" or self.extension==".mov":
                root.title('速度検知中')
                self.file_name.set('結果はLINEへ送信されています')
                detect_speed.main(self.select_file_name, "yolov7/weights/car_best.pt", view=True)
                root.title('物体検出アプリケーション')
                self.file_name.set('ファイルが選択されています')
            else:
                messagebox.showerror('error', "動画ファイルを選択してください")
        except:
            messagebox.showerror('error', "ファイルが選択されていません")
    
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
            if self.extension in filetypes[0:2]:
                self.file_name.set("画像ファイルが選択されています")
                img = Image.open(self.select_file_name)
                w = img.width
                h = img.height
            else :
                self.file_name.set("動画ファイルが選択されています")
                cap = cv2.VideoCapture(self.select_file_name)
                __, img = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            can_size = int(self.clientWidth)-50
            ratio = min(can_size / w, can_size / h)
            self.input_image = img.resize((round(ratio * w), round(ratio * h)))
            self.input_image = ImageTk.PhotoImage(self.input_image)
            self.input_canvas.create_image(can_size/2, can_size/2, image=self.input_image)
        else :
            messagebox.showerror('error', "画像または動画ファイルを選択してください")
    
    def menuExit(self):
        cp = configparser.ConfigParser()
        cp['Client'] = { 'Height': str(root.winfo_height()),
                        'Width': str(root.winfo_width())}
        with open(self.__class__.__name__ + '.ini', 'w') as f:
            cp.write(f)
        root.destroy()

    def menuHelpOpenWeb(self):
        webbrowser.open('https://github.com/ShusukeHamamura/object_detection_app.git')

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