import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QWidget, QVBoxLayout, QTabWidget, QHBoxLayout, QSlider, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from collections import deque
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
cv2.setLogLevel(0)

CLEAR_ICON_PATH = r"X:\Projects\Personal\Hand Gesture\Assets\trash.png"
SAVE_ICON_PATH = r"X:\Projects\Personal\Hand Gesture\Assets\save.png"
CURSOR_PEN_ICON_PATH = r"X:\Projects\Personal\Hand Gesture\Assets\pencil.png"
CURSOR_ERASER_ICON_PATH = r"X:\Projects\Personal\Hand Gesture\Assets\eraser.png"
CURSOR_DEFAULT_ICON_PATH = r"X:\Projects\Personal\Hand Gesture\Assets\cursor.png"

MIN_THICK, MAX_THICK, ERASE_THICK, FPS_LOCK = 2, 30, 125, 150
CANVAS_W, CANVAS_H = 1920, 1080
BASE_COLOR_NAMES = ['Black','White','Red','Green','Blue','Yellow','Magenta','Cyan']
BASE_COLORS = {
    'Black':(0,0,0),'White':(255,255,255),
    'Red':(0,0,255),'Green':(0,255,0),
    'Blue':(255,0,0),'Yellow':(0,255,255),
    'Magenta':(255,0,255),'Cyan':(255,255,0),
    'Gray':(128,128,128)
}
CURSOR_SIZE = 40

def load_icon(path,size):
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if img is None: return None
    h,w=img.shape[:2]; s=size/max(h,w)
    return cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)

def overlay_icon(canvas,icon,x,y):
    ih,iw=icon.shape[:2]
    x1,y1=max(x,0),max(y,0)
    x2,y2=min(x+iw,canvas.shape[1]),min(y+ih,canvas.shape[0])
    if x1>=x2 or y1>=y2: return
    ix1,iy1=x1-x,y1-y
    ix2,iy2=ix1+(x2-x1),iy1+(y2-y1)
    region=canvas[y1:y2,x1:x2]; part=icon[iy1:iy2,ix1:ix2]
    if part.shape[2]==4:
        alpha=part[:,:,3:]/255.0
        canvas[y1:y2,x1:x2]=(alpha*part[:,:,:3]+(1-alpha)*region).astype(np.uint8)
    else:
        canvas[y1:y2,x1:x2]=part

def fingers_up(lm,hl):
    tips=[4,8,12,16,20]; pips=[3,6,10,14,18]
    thumb=int(lm.landmark[tips[0]].x<lm.landmark[pips[0]].x) if hl=='Right' else int(lm.landmark[tips[0]].x>lm.landmark[pips[0]].x)
    flags=[thumb]
    for t,p in zip(tips[1:],pips[1:]):
        flags.append(int(lm.landmark[t].y<lm.landmark[p].y))
    return flags

def recognize_and_draw_shape(pts,img,col,thick):
    if len(pts)<5: return
    arr=np.array(pts,np.int32)
    vx,vy,x0,y0=cv2.fitLine(arr,cv2.DIST_L2,0,0.01,0.01).flatten()
    d=np.abs(vy*(arr[:,0]-x0)-vx*(arr[:,1]-y0))/np.hypot(vx,vy)
    if d.max()<5:
        cv2.line(img,tuple(arr[0]),tuple(arr[-1]),col,thick); return
    x,y,w_,h_=cv2.boundingRect(arr); area=w_*h_
    if area>0 and abs(cv2.contourArea(arr)-area)/area<0.2:
        cv2.rectangle(img,(x,y),(x+w_,y+h_),col,thick); return
    (cx,cy),r=cv2.minEnclosingCircle(arr)
    err=np.max(np.abs(np.hypot(arr[:,0]-cx,arr[:,1]-cy)-r))
    if err<5:
        cv2.circle(img,(int(cx),int(cy)),int(r),col,thick)

mp_hands=mp.solutions.hands; mp_draw=mp.solutions.drawing_utils

class DrawApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand-Tracking Draw")
        self.cap=cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,CANVAS_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CANVAS_H)
        self.hands=mp_hands.Hands(static_image_mode=False,model_complexity=1,max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.kalman=cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix=np.eye(2,4,dtype=np.float32)
        self.kalman.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov=np.eye(4,dtype=np.float32)*1e-2
        self.kalman.measurementNoiseCov=np.eye(2,dtype=np.float32)*1e-1
        self.kalman.errorCovPost=np.eye(4,dtype=np.float32)
        self.kalman.statePost=np.array([[CANVAS_W/2],[CANVAS_H/2],[0],[0]],np.float32)
        self.clear_icon=load_icon(CLEAR_ICON_PATH,60)
        self.save_icon=load_icon(SAVE_ICON_PATH,60)
        self.pen_cursor=load_icon(CURSOR_PEN_ICON_PATH,CURSOR_SIZE)
        self.eraser_cursor=load_icon(CURSOR_ERASER_ICON_PATH,CURSOR_SIZE)
        self.default_cursor=load_icon(CURSOR_DEFAULT_ICON_PATH,CURSOR_SIZE)
        self.canvases=[self._make_canvas('White')]
        self.page_idx=0; self._just_swiped=False
        self.stroke_raw=[]; self.smooth_buf=deque(maxlen=5)
        self.PS,self.M=60,20
        self.init_ui()
        self.timer=QTimer(self); self.timer.timeout.connect(self.update_frame); self.timer.start(int(1000/FPS_LOCK))

    def _make_canvas(self,bg):
        return np.ones((CANVAS_H,CANVAS_W,3),np.uint8)*(255 if bg=='White' else 0)

    def init_ui(self):
        tabs=QTabWidget()
        cam=QWidget(); cl=QVBoxLayout(cam); cl.addStretch()
        row=QHBoxLayout(); row.addStretch()
        self.cam_lbl=QLabel(); self.cam_lbl.setFixedSize(CANVAS_W,CANVAS_H)
        row.addWidget(self.cam_lbl); row.addStretch(); cl.addLayout(row); cl.addStretch()
        tabs.addTab(cam,"Camera")
        cvs=QWidget(); vl=QVBoxLayout(cvs)
        top=QHBoxLayout(); top.addStretch()
        self.bg_cb=QComboBox(); self.bg_cb.addItems(["White","Black"]); self.bg_cb.currentTextChanged.connect(self._change_bg)
        top.addWidget(QLabel("Canvas:")); top.addWidget(self.bg_cb)
        self.hand_cb=QComboBox(); self.hand_cb.addItems(["Right","Left"])
        top.addWidget(QLabel("Hand:")); top.addWidget(self.hand_cb)
        self.color_cb=QComboBox(); self.color_cb.addItems(BASE_COLOR_NAMES)
        top.addWidget(QLabel("Brush:")); top.addWidget(self.color_cb)
        top.addStretch(); vl.addLayout(top); vl.addStretch()
        mid=QHBoxLayout(); mid.addStretch()
        self.slider=QSlider(Qt.Vertical); self.slider.setRange(MIN_THICK,MAX_THICK); self.slider.setValue((MIN_THICK+MAX_THICK)//2)
        mid.addWidget(self.slider); mid.addSpacing(20)
        self.canvas_lbl=QLabel(); self.canvas_lbl.setFixedSize(CANVAS_W,CANVAS_H)
        mid.addWidget(self.canvas_lbl); mid.addStretch(); vl.addLayout(mid); vl.addStretch()
        tabs.addTab(cvs,"Canvas"); self.setCentralWidget(tabs)

    def _change_bg(self,t): self.canvases=[self._make_canvas(t)]; self.page_idx=0

    @property
    def hand_pref(self): return self.hand_cb.currentText().lower()

    def goto_page(self,idx):
        if idx<0: return
        while idx>=len(self.canvases): self.canvases.append(self._make_canvas(self.bg_cb.currentText()))
        self.page_idx=idx; self.stroke_raw.clear(); self.smooth_buf.clear(); self._just_swiped=True

    def clear_current(self): self.canvases[self.page_idx][:]=self.canvases[self.page_idx][0,0]

    def save_all(self):
        name,ok=QInputDialog.getText(self,"Save As","Filename:")
        if not ok or not name: return
        fn=f"{name}.pdf"
        imgs=[Image.fromarray(cv2.cvtColor(cv,cv2.COLOR_BGR2RGB)) for cv in self.canvases]
        imgs[0].save(fn,"PDF",save_all=True,append_images=imgs[1:])
        QMessageBox.information(self,"Saved",f"Drawing saved as:\n{fn}")

    def update_frame(self):
        ret,frame=self.cap.read()
        if not ret: return
        frame=cv2.flip(frame,1); cvs=self.canvases[self.page_idx]
        total_w=len(BASE_COLOR_NAMES)*(self.PS+5)-5; sx=(CANVAS_W-total_w)//2
        for i,col in enumerate(BASE_COLOR_NAMES):
            x,y=sx+i*(self.PS+5),self.M
            cv2.rectangle(cvs,(x,y),(x+self.PS,y+self.PS),BASE_COLORS[col],-1)
            cv2.rectangle(cvs,(x,y),(x+self.PS,y+self.PS),(200,200,200),1)
        overlay_icon(cvs,self.clear_icon,sx-2*self.M-60,self.M)
        sw=self.save_icon.shape[1]; overlay_icon(cvs,self.save_icon,sx+total_w+2*self.M,self.M)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=self.hands.process(rgb)
        mode,ix,iy='none',None,None; thickness=self.slider.value()

        if res.multi_hand_landmarks and res.multi_handedness:
            for i,lm in enumerate(res.multi_hand_landmarks):
                hl=res.multi_handedness[i].classification[0].label
                if hl.lower()!=self.hand_pref: continue
                st=fingers_up(lm,hl)
                if st==[0,1,1,1,0] and not self._just_swiped:
                    tip_x=int(lm.landmark[8].x*CANVAS_W)
                    self.goto_page(self.page_idx+1 if tip_x>CANVAS_W//2 else self.page_idx-1)
                elif st!=[0,1,1,1,0]: self._just_swiped=False
                rx=int(lm.landmark[8].x*CANVAS_W); ry=int(lm.landmark[8].y*CANVAS_H)
                self.kalman.correct(np.array([[np.float32(rx)],[np.float32(ry)]]))
                p=self.kalman.predict(); ix,iy=int(p[0,0]),int(p[1,0])
                if st==[0,1,0,0,0]: mode='draw'
                elif st==[0,1,1,0,0]: mode='erase'
                elif st==[1,1,0,0,0]:
                    for j,nm in enumerate(BASE_COLOR_NAMES):
                        x0=sx+j*(self.PS+5)
                        if x0<ix<x0+self.PS and self.M<iy<self.M+self.PS: self.color_cb.setCurrentText(nm)
                    cx,cy=sx-2*self.M-60,self.M
                    if cx<ix<cx+60 and cy<iy<cy+60: self.clear_current()
                    sx2=sx+total_w+2*self.M
                    if sx2<ix<sx2+sw and cy<iy<cy+60: self.save_all()
                mp_draw.draw_landmarks(frame,lm,mp.solutions.hands.HAND_CONNECTIONS)
                break

        disp=cvs.copy()
        cv2.putText(disp, f"Page {self.page_idx+1}", (10, CANVAS_H-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if mode=='draw' and ix is not None:
            overlay_icon(disp,self.pen_cursor,ix-CURSOR_SIZE,iy-CURSOR_SIZE)
            self.stroke_raw.append((ix,iy)); self.smooth_buf.append((ix,iy))
            pts=np.array(self.smooth_buf); sx2,sy2=int(pts[:,0].mean()),int(pts[:,1].mean())
            if getattr(self,'prev_smooth',None): cv2.line(cvs,self.prev_smooth,(sx2,sy2),BASE_COLORS[self.color_cb.currentText()],thickness)
            self.prev_smooth=(sx2,sy2)

        elif mode=='erase' and ix is not None:
            overlay_icon(disp,self.eraser_cursor,ix-CURSOR_SIZE,iy-CURSOR_SIZE)
            if getattr(self,'prev_smooth',None):
                bgv=255 if self.bg_cb.currentText()=='White' else 0
                cv2.line(cvs,self.prev_smooth,(ix,iy),(bgv,bgv,bgv),ERASE_THICK)
            self.prev_smooth=(ix,iy)

        else:
            if ix is not None: overlay_icon(disp,self.default_cursor,ix-CURSOR_SIZE,iy-CURSOR_SIZE)
            if len(self.stroke_raw)>5:
                recognize_and_draw_shape(self.stroke_raw,cvs,BASE_COLORS[self.color_cb.currentText()],thickness)
            self.stroke_raw.clear(); self.smooth_buf.clear(); self.prev_smooth=None

        self._show_img(frame,self.cam_lbl)
        self._show_img(disp,self.canvas_lbl)

    def _show_img(self,img,label):
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,_=rgb.shape
        q=QImage(rgb.data,w,h,QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q))

if __name__=='__main__':
    app=QApplication(sys.argv)
    win=DrawApp()
    win.show()
    sys.exit(app.exec_())
