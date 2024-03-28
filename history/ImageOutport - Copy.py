import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QRectF, QObject, pyqtSignal
from PyQt5.QtWidgets import QMenu, QAction
import cv2




IMG_IN, IMG_OUT = "Img In", "Img Out"
RAW_IMG, CROP_IMG = "Raw Img", "Crop Img"
DATA_IMG_ORIG, DATA_IMG = "Img Data Orig", "Img Data"
DATA_IMG_FT, DATA_IMG_FT_SHIFTED = "Img Data FT", "Img Data FT Shifted"
FT_MAG, FT_PHASE, FT_REAL, FT_IMAG  = "FT Magnitude", "FT Phase", "FT Real", "FT Imaginary"

class SignalEmitter(QObject):
    change_ROI = pyqtSignal(pg.ROI)
    created_object = pyqtSignal()

class ImageOutport(QtWidgets.QWidget):
    def __init__(self, parent):
        super(ImageOutport, self).__init__(parent)
        uic.loadUi("./ImageOutport.ui", self)
        # self.setupUi(self)
        self.show()
        

        self.signal_emitter = SignalEmitter()
        self.ROI_boundaries = QRectF(0, 0, 100, 100)
        self.image = None
        self.original_image_attr = self.create_image_attributes()
        self.output_image_attr = self.create_image_attributes()

        
        self.ft_flag, self.loaded_flag = True, False
        self.brightness_constant, self.contrast_constant = 1, 1
        self.ROI_position = None


        self.FTCombobox.currentIndexChanged.connect(self.plot_ft)
        
        self.image_view = self.ImageWidget.addViewBox()
        self.ft_view = self.FTWidget.addViewBox()
        self.configure_view(self.image_view)
        self.configure_view(self.ft_view)

        self.image_item = pg.ImageItem()
        self.ft_item = pg.ImageItem()
        self.image_view.addItem(self.image_item)
        self.ft_view.addItem(self.ft_item)

        self.ft_ROI = pg.ROI(pos = self.ft_view.viewRect().center(), size = (50, 50), hoverPen='b', resizable= True, invertible= True, rotatable= False, maxBounds= self.ROI_boundaries)
        self.ft_view.addItem(self.ft_ROI)
        self.add_scale_handles_ROI(self.ft_ROI)

        self.ft_ROI.sigRegionChangeFinished.connect(lambda: self.update_region(flag = True))
        self.ImageWidget.scene().sigMouseClicked.connect(self.browse_handler)
        self.ImageWidget.scene().sigMouseClicked.connect(lambda event: self.brightness_contrast_handler() if event.button() == 2 else None)
        self.image_item.mouseDragEvent = self.brightness_change





    def brightness_change(self, event):
        drag_distance = event.pos() - event.lastPos()

        brightness_delta = drag_distance.x() * 3
        contrast_delta = drag_distance.y() * 3

        self.brightness_constant = max(-100, min(100, self.brightness_constant + brightness_delta))
        self.contrast_constant = max(-100, min(100, self.contrast_constant + contrast_delta))

        self.update_brightness_contrast()



    def update_brightness_contrast(self):
        modified_image = cv2.convertScaleAbs(self.original_image_attr[DATA_IMG_ORIG], alpha=(self.contrast_constant / 100.0), beta=self.brightness_constant)

        self.original_image_attr[DATA_IMG] = modified_image
        self.output_image_attr[DATA_IMG] = modified_image
        if self.ft_flag:
            self.update_region(flag = True)

        self.calc_imag_ft(self.original_image_attr)
        self.calc_imag_ft(self.output_image_attr)

        self.display_img(modified_image)



    
    def browse_handler(self, event):
        if not self.loaded_flag:
            if event.double():
                file_dialog = QFileDialog(self)
                file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif)")
                file_dialog.setWindowTitle("Open Image File")
                file_dialog.setFileMode(QFileDialog.ExistingFile)

                if file_dialog.exec_() == QFileDialog.Accepted:
                    selected_file = file_dialog.selectedFiles()[0]
                    self.load_image(selected_file)

    

    def load_image(self, path):
        self.original_image_attr[RAW_IMG] = cv2.imread(path)
        self.image = cv2.rotate(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), cv2.ROTATE_90_CLOCKWISE)

        self.original_image_attr[DATA_IMG] = self.image
        self.original_image_attr[DATA_IMG_ORIG] = self.image
        self.output_image_attr[DATA_IMG] = self.image
        self.calc_imag_ft(self.original_image_attr)
        self.calc_imag_ft(self.output_image_attr)
        self.display_img(self.original_image_attr[DATA_IMG])
        self.ROI_boundaries.adjust(0, 0, self.ft_item.width(), self.ft_item.height())
        self.set_ROI_size()
        self.center_ROI()





   
    def update_region(self, flag = False):
        if flag:
            self.signal_emitter.change_ROI.emit(self.sender())
        self.original_image_attr[IMG_IN], self.original_image_attr[IMG_OUT] = self.part_region()
        new_image = self.ft_ROI.getArrayRegion(self.original_image_attr[DATA_IMG_FT_SHIFTED], self.ft_item)
        new_image = np.fft.ifft2(np.fft.ifftshift(new_image))
        self.output_image_attr[DATA_IMG] = new_image
        self.calc_imag_ft(self.output_image_attr)

    
    def display_img(self, data):
        self.image_item.setImage(data)



    def calc_imag_ft(self, image_attr):
        if image_attr[DATA_IMG] is None or image_attr[DATA_IMG].size == 0:
            return
        image_attr[DATA_IMG_FT] = np.fft.fft2(image_attr[DATA_IMG])
        image_attr[DATA_IMG_FT_SHIFTED] = np.fft.fftshift(image_attr[DATA_IMG_FT])

        image_attr[FT_MAG] = np.abs(image_attr[DATA_IMG_FT_SHIFTED])
        image_attr[FT_PHASE] = np.angle(image_attr[DATA_IMG_FT_SHIFTED])
        image_attr[FT_REAL] = np.real(image_attr[DATA_IMG_FT_SHIFTED])
        image_attr[FT_IMAG] = np.imag(image_attr[DATA_IMG_FT_SHIFTED])

        self.plot_ft()

    
    def create_image_attributes(self):
        return {
            IMG_IN: None, IMG_OUT: None, RAW_IMG: None, CROP_IMG: None,
            DATA_IMG_ORIG: None, DATA_IMG: None, DATA_IMG_FT: None, DATA_IMG_FT_SHIFTED: None,
            FT_MAG: None, FT_PHASE: None, FT_REAL: None, FT_IMAG: None}
    


    def brightness_contrast_handler(self):
        self.original_image_attr[DATA_IMG] = self.original_image_attr[DATA_IMG_ORIG]
        self.output_image_attr[DATA_IMG] = self.original_image_attr[DATA_IMG_ORIG]

        if self.ft_flag:
            self.update_region(flag = True)

        self.calc_imag_ft(self.original_image_attr)
        self.calc_imag_ft(self.output_image_attr)

        self.display_img(self.original_image_attr[DATA_IMG_ORIG])



    def part_region(self):
        indices, _ = self.ft_ROI.getArraySlice(self.original_image_attr[DATA_IMG_FT_SHIFTED], self.ft_item, returnSlice=True)
        
        region = np.full(self.original_image_attr[DATA_IMG_FT_SHIFTED].shape, False)
        region[indices] = True
        
        data_in_region = self.original_image_attr[DATA_IMG_FT_SHIFTED] * region
        data_out_region = self.original_image_attr[DATA_IMG_FT_SHIFTED].copy()
        data_out_region[region] = 0

        return (data_in_region, data_out_region)


    def configure_view(self, view):
            view.setAspectLocked(True)
            view.setMouseEnabled(x=False, y=False)
            view.setMenuEnabled(False)

    
    def plot_ft(self):
        if self.FTCombobox.currentText() == 'FT Magnitude':
            self.ft_item.setImage(np.log(1+self.original_image_attr[self.FTCombobox.currentText()]))
        else:
            self.ft_item.setImage(self.original_image_attr[self.FTCombobox.currentText()])
    



    def add_scale_handles_ROI(self, roi : pg.ROI):
        positions = np.array([[0,0], [1,0], [1,1], [0,1]])
        for pos in positions:        
            roi.addScaleHandle(pos = pos, center = 1 - pos)
            
    
    def center_ROI(self):
        roi_rect = self.ft_ROI.size()
        half_width = roi_rect[0] / 2
        half_height = roi_rect[1] / 2
        center = self.ft_item.boundingRect().center()
        adjusted_center = [center.x() - half_width, center.y()- half_height]
        self.ft_ROI.setPos(adjusted_center)
        
        
    def set_ROI_size(self):
        self.ft_ROI.setSize(size = (self.ft_item.boundingRect().width(), self.ft_item.boundingRect().height()))
        


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ImageOutport(None)
    window.show()
    app.exec_()
