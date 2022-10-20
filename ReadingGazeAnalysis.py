# -*- coding: utf-8 -*-
"""
POLS 視野等表示 ver.9
Stand-alone

"""
# Import and setting ===========================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
import pickle
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.simpledialog as simpledialog
from mpl_toolkits.mplot3d import Axes3D
from chardet.universaldetector import UniversalDetector# 文字コード判定
import scipy.signal as sign

fp = FontProperties(fname="c:\\Windows\\Fonts\\YuGothM.ttc")#日本語フォント位置指定

deffont=('Yu Gothic', 20)
# file_name = ""


class ShowViewPointapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="k-lab_logo.ico")
        tk.Tk.wm_title(self, "EOG Gaze Track 3D")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (StartPage, GazeMap,GazeAnalysis):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        global sf
        style = ttk.Style()
        style.configure('TButton', font=deffont)
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="CSVファイルを選択してください", font=deffont)
        label.pack(pady=10)

        button0 = ttk.Button(self, text='ファイル選択', style='TButton', command=lambda: self.load_file())
        button0.pack(pady=10, ipadx=10)

        label2 = ttk.Label(self, text="サンプリング間隔", font=deffont)
        label2.pack(pady=10)

        sf = ttk.Spinbox(self, from_=1, to=20, width=5, increment=1, font=deffont)
        sf.pack(pady=10)

        label3 = ttk.Label(self, text="(msec)", font=deffont)
        label3.pack(pady=10)


        button2 = ttk.Button(self, text="視線追跡", command=lambda: controller.show_frame(GazeMap))
        button2.pack(pady=10, ipadx=10)
        
        button3 = ttk.Button(self, text="統計解析", command=lambda: controller.show_frame(GazeAnalysis))
        button3.pack(pady=10, ipadx=10)

    def load_file(self):
        global file_name,encode

        file_name = filedialog.askopenfilename(filetypes=[("CSV Files", ".csv")])

        # 文字コード判定------------------------------------
        detector = UniversalDetector()
        fen = open(file_name, mode='rb')
        for binary in fen:
            detector.feed(binary)
            if detector.done:
                break
        detector.close()
        encode=detector.result['encoding']

        self.text = tk.StringVar()#file nameの更新
        self.text.set("%s" % file_name)

        label_path = ttk.Label(self, textvariable=self.text,font=("Yu Gothic", 17))
        label_path.pack(pady=10)


    def sampling_frequency(self):
        sf_values = int(sf.get())
        freq=1000/sf_values
        return freq

    def Data_analysis(self):
        global file_name,pos, move_point,xylim,AnalysisData,signal

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)


        # Pols data analysis =======================================================================================

        signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding=encode)
        # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",")
        # 　csv ファイルを行列式として読み込む（utf-8形式）



class GazeMap(tk.Frame):

    def __init__(self, parent, controller):
        global  sf, b1, b2, bv1, bv2
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="位置か速度を選択してください", font=deffont)
        label.grid(pady=10, padx=10, row=0, column=0, columnspan=3)
        
        bv1=tk.StringVar()
        b1 = ttk.Entry(self,width=10, textvariable=bv1)
        b1.insert(tk.END,"0")
        b1.grid(pady=10, padx=10, row=1, column=0)
        
        label1 = ttk.Label(self, text="~", font=deffont)
        label1.grid(row=1, column=1)
        
        bv2=tk.StringVar()
        b2 = ttk.Entry(self,width=10, textvariable=bv2)
        b2.insert(tk.END,"20")
        b2.grid(pady=10, padx=10, row=1, column=2)

        button1 = ttk.Button(self, text='x', style='TButton', command=lambda: self.gaze_3d())
        button1.grid(pady=10, padx=10, row=3, column=0, columnspan=3)
        
        button3 = ttk.Button(self, text='y', style='TButton', command=lambda: self.gaze_v())
        button3.grid(pady=10, padx=10, row=4, column=0, columnspan=3)

        button2 = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button2.grid(pady=10, padx=10, row=5, column=0, columnspan=3)
    
    def spin_values(self):
        b1_values = float(bv1.get())
        b2_values = float(bv2.get())
        print(bv1.get)
        print(bv2.get)
        print(b1_values)
        print(b2_values)
        return b1_values, b2_values
        
    def maxwave_detector(self, x, stime, start, end):
    #サッケードピーク(最大)を見つけ出す関数
        idx=sign.argrelmax(x,order=50)
        print(idx)
        cutted_idx=np.array([])
        for i in idx[0]:
            if (i >= start and i <= end):
                cutted_idx=np.append(cutted_idx,i)
        
        max_time=([])
        max_pos=([])
        for i in range(len(cutted_idx)):
            max_time=np.append(max_time, stime[int(cutted_idx[i])])
            max_pos=np.append(max_pos, x[int(cutted_idx[i])])

        return max_time,max_pos
        
    def minwave_detector(self, x, stime, start, end):
    #サッケードピーク(最小)を見つけ出す関数
        idx=sign.argrelmin(x,order=50)
        
        #絶対値の標準偏差を抽出する処理(10/17追加
        abs_pos=np.abs(x)
        SD_pos=np.std(abs_pos)
        print(SD_pos)
        
        print(idx)
        cutted_idx=np.array([])
        for i in idx[0]:
            if (i >= start and i <= end and abs(x[i]) >= SD_pos):
                cutted_idx=np.append(cutted_idx,i)
        
        min_time=([])
        min_pos=([])
        for i in range(len(cutted_idx)):
            min_time=np.append(min_time, stime[int(cutted_idx[i])])
            min_pos=np.append(min_pos, x[int(cutted_idx[i])])

        return min_time, min_pos
        
        

    def gaze_3d(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)

            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(vsignal[:, 0:2], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t, d=(1 / samplingfreq))
            bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
            fft_sig[bandpassidx, :] = 0
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            pos = ifft_sig
            np.savetxt("vertical.csv", pos, delimiter=",")
            posx=pos[:, 0]
            posy=pos[:, 1]
            
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
            
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
                startidx=cbox1
            else:
                startidx=0
            
            if (box2 <= box1):
                endidx=t
            elif (cbox2>0) and (cbox2<=t):
                endidx=cbox2
            else:
                endidx=t
             
            #SCピークを関数から算出
            max_time, max_posx =self.maxwave_detector(posx, stime, startidx, endidx)
            min_time, min_posx=self.minwave_detector(posx, stime, startidx, endidx)
            
            


            #plot--------------------------------------------------------------------
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(stime[startidx:endidx], posx[startidx:endidx], label='Eye Movement')
            ax.plot(max_time, max_posx, 'ro', label='SCPeak')
            ax.plot(min_time, min_posx, 'ro')
            ax.set_xlabel("time[s]")
            ax.set_ylabel("Vertical Signal")
            plt.legend(loc="best")
            plt.show()
            

            
    def gaze_v(self):

        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)

            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(vsignal[:, 0:2], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t, d=(1 / samplingfreq))
            bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
            fft_sig[bandpassidx, :] = 0
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            pos = ifft_sig
            np.savetxt("vertical.csv", pos, delimiter=",")
            posx=pos[:, 0]
            posy=pos[:, 1]
            
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
            
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
                startidx=cbox1
            else:
                startidx=0
            
            if (box2 <= box1):
                endidx=t
            elif (cbox2>0) and (cbox2<=t):
                endidx=cbox2
            else:
                endidx=t
             
            
            max_time, max_posy =self.maxwave_detector(posy, stime, startidx, endidx)
            min_time, min_posy=self.minwave_detector(posy, stime, startidx, endidx)


            #plot--------------------------------------------------------------------
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(stime[startidx:endidx], posy[startidx:endidx], label='Eye Movement')
            ax.plot(max_time, max_posy, 'ro', label='SCPeak')
            ax.plot(min_time, min_posy, 'ro')
            ax.set_xlabel("time[s]")
            ax.set_ylabel("Vertical Signal")
            plt.legend(loc="best")
            plt.show()


class GazeAnalysis(tk.Frame):

    def __init__(self, parent, controller):
        global  sf, b1, b2, bv3, bv4
        ttk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="位相か速度を選択してください", font=deffont)
        label.grid(pady=10, padx=10, row=0, column=0, columnspan=3)
        
        bv3=tk.StringVar()
        b1=ttk.Entry(self, width=10, textvariable=bv3)
        b1.insert(tk.END,"0")
        b1.grid(pady=10, padx=10, row=1, column=0)

        label1 = ttk.Label(self, text="~", font=deffont)
        label1.grid(row=1, column=1)
        
        bv4=tk.StringVar()
        b2=ttk.Entry(self, width=10, textvariable=bv4)
        b2.insert(tk.END,"0")
        b2.grid(pady=10, padx=10, row=1, column=2)

        button1 = ttk.Button(self, text='位相', style='TButton', command=lambda: self.gaze_2dpha())
        button1.grid(pady=10, padx=10, row=3, column=0, columnspan=3)
        
        button3 = ttk.Button(self, text='速度', style='TButton', command=lambda: self.gaze_2dver())
        button3.grid(pady=10, padx=10, row=4, column=0, columnspan=3)

        button2 = ttk.Button(self, text="戻る", command=lambda: controller.show_frame(StartPage))
        button2.grid(pady=10, padx=10, row=5, column=0, columnspan=3)
     
    def spin_values(self):
        b1_values = float(bv3.get())
        b2_values = float(bv4.get())
        print(bv3.get)
        print(bv4.get)
        print(b1_values)
        print(b2_values)
        return b1_values, b2_values
        

    def gaze_2dpha(self):
        
        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)
            
            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            
            samplingfreq =StartPage.sampling_frequency(self)
            stime = np.arange(t) / samplingfreq  # 検査時間(sec)

            # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(vsignal[:, 0:2], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t, d=(1 / samplingfreq))
            bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
            fft_sig[bandpassidx, :] = 0
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            pos = np.cumsum(ifft_sig, axis=0)  # 位置座標一覧の作成(逆フーリエ変換した信号をX軸Y軸それぞれに対して積分)
            np.savetxt("integral.csv", pos, delimiter=",")
            posx=pos[:, 0]
            posy=pos[:, 1]
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
           
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
             startidx=cbox1
            else:
             startidx=0
            
            if (box2 <= box1):
             endidx=t
            elif (cbox2>0) and (cbox2<=t):
             endidx=cbox2
            else:
             endidx=t
            
            
        
            #plot--------------------------------------------------------------------
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(posx[startidx:endidx], posy[startidx:endidx], label='Eye Movement')
            ax.set_xlabel("Horizontal Signal")
            ax.set_ylabel("Vertical Signal")
            plt.legend(loc="best")
            plt.show()

            
    def gaze_2dver(self):
        
        if file_name == "":
            sub_win = tk.Toplevel()
            tk.Message(sub_win, aspect=1000, text="CSVファイルを選択してください",
                       font=deffont).pack(pady=10, padx=10)

        else:
            signal = np.loadtxt("%s" % file_name, skiprows=7, usecols=(0,1), delimiter=",", encoding=encode)
            # signal = np.loadtxt("%s" % file_name, skiprows=1, delimiter=",", encoding="utf-8")
            # signal = np.loadtxt("%s" %file_name, skiprows=1, delimiter=",")
            # 　csv ファイルを行列式として読み込む（utf-8形式）
            vsignal=signal*(10-(-10))/65536+(-10)
            
            (t, s) = signal.shape  # 行列数表示: t=行数(時間データ),s=列数(計測項目データ)
            samplingfreq =StartPage.sampling_frequency(self)
            
            
            box1, box2= self.spin_values()
            cbox1=int(box1*samplingfreq)
            cbox2=int(box2*samplingfreq)
           
            # zoom--------------------------------------------------------------------
            if (0<=cbox1) and (cbox1 < t):
             startidx=cbox1
            else:
             startidx=0
            
            if (box2 <= box1):
             endidx=t
            elif (cbox2>0) and (cbox2<=t):
             endidx=cbox2
            else:
             endidx=t
             
            ssignal=vsignal[startidx:endidx]
            (t2,s2)=ssignal.shape
            
            """
            Memo -------------------------------------------------------------------------------------------------
            signal:
            0=No,1=Idx_Raw,2=テーブル番号,3=絶対位置X,4=絶対位置Y,5=表示状態,6=状態連番,
            7=信号Ch0(水平信号(左眼反転)),8=信号(Ch1垂直信号(mV)),9=信号Ch2(水平眼位(左眼反転)),10=信号Ch3(垂直眼位)
            ------------------------------------------------------------------------------------------------------
            """
            
            
            
            stime = np.arange(t2) / samplingfreq  # 検査時間(sec)

            # FFT --------------------------------------------------------------------
            fft_sig = np.fft.fft(ssignal[:, 0:2], axis=0)  # X軸Y軸それぞれに対してFFT
            freq = np.fft.fftfreq(t2, d=(1 / samplingfreq))
            bandpassidx = np.where((np.abs(freq) < 0.02) | (np.abs(freq) > 9.8))[0]
            fft_sig[bandpassidx, :] = 0
            ifft_sig = np.real(np.fft.ifft(fft_sig, axis=0))
            pos = ifft_sig
            np.savetxt("verocity.csv", pos, delimiter=",")
            posx=pos[:, 0]
            posy=pos[:, 1]
            
             
            #　microsaccade detect-----------------------------------------------------
            
            #　posx,yを昇順にソートし、(値-中央値)²を行っている
            e1=np.sort(posx)
            e2=np.sort(posy)
            e3=np.square(e1-np.median(e1))
            e4=np.square(e1-np.median(e2))
            # 標準偏差から閾値を計算
            q=len(e1)
            print(q)
            thresholdx=4*(np.sqrt((np.sum(e3))/q))
            thresholdy=4*(np.sqrt((np.sum(e4))/q))
            
            
            
        
            # plot--------------------------------------------------------------------
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(posx, posy, label='Eye Movement')
            # 閾値をもとにした楕円の作成
            theta=np.linspace(0, 2*np.pi, 65)
            elx=thresholdx*np.cos(theta)
            ely=thresholdy*np.sin(theta)
            ax.plot(elx, ely, label='Threshold')
            #ココマデ
            ax.set_xlabel("Horizontal Signal")
            ax.set_ylabel("Vertical Signal")
            plt.legend(loc="best")
            plt.show()     

     
            
           



app = ShowViewPointapp()
app.mainloop()
