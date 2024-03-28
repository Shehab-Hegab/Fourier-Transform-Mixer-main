from PyQt5.QtWidgets import QWidget, QFileDialog
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5 import QtWidgets, uic
from Image import Image
from PyQt5.QtCore import QRectF, QObject, pyqtSignal
import logging

logging.basicConfig(filename = 'application.log', level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s', filemode='w')



class SignalEmitter(QObject):
    sig_ROI_changed = pyqtSignal(pg.ROI)
    objectCreated = pyqtSignal()

class ImageOutport(QtWidgets.QWidget):
    def __init__(self, parent):
        super(ImageOutport, self).__init__(parent)
        uic.loadUi("./ImageOutport.ui", self)
        self.show()
        
        self.sig_emitter = SignalEmitter()
        self.ft_enabled, self.can_browse = True, True
        self.brightness_val, self.contrast_val = 1, 1
        self.input_image_data, self.modified_image_data = Image(), Image()
        self.ROI_Maxbounds = QRectF(0, 0, 100, 100)
        self.loaded = False
        self.img_data = None
        self.initial_roi_position = None
        self.image_view = self.ImageWidget.addViewBox()
        self.ft_view = self.FTWidget.addViewBox()
        self.img_item, self.ft_item = pg.ImageItem(), pg.ImageItem()
        
        
        self.set_view_settings(self.ft_view)
        self.set_view_settings(self.image_view)
        self.image_view.addItem(self.img_item)
        self.ft_view.addItem(self.ft_item)
        self.ft_roi = pg.ROI(pos = self.ft_view.viewRect().center(), size = (50, 50), hoverPen='b', resizable= True, invertible= True, rotatable= False, maxBounds= self.ROI_Maxbounds)
        
        self.ft_view.addItem(self.ft_roi)
        for pos in np.array([[0,0], [1,0], [1,1], [0,1]]):        
            self.ft_roi.addScaleHandle(pos = pos, center = 1 - pos)



        self.FTCombobox.currentIndexChanged.connect(lambda :self.display_ft(self.FTCombobox.currentText()))
        self.ft_roi.sigRegionChangeFinished.connect(lambda: self.update_region(finish = True))
        self.ImageWidget.scene().sigMouseClicked.connect(self.browse_handler)
        self.ImageWidget.scene().sigMouseClicked.connect(lambda event: self.reset_brightness_contrast() if event.button() == 2 else None)
        self.FTWidget.scene().sigMouseClicked.connect(lambda event: self.reset_ROI() if event.button() == 2 else None)
        self.img_item.mouseDragEvent = self.set_brightness_contrast
        
        
        
    
    def update_region(self, finish = False):
        if finish:
            self.sig_emitter.sig_ROI_changed.emit(self.ft_roi)
        self.modified_image_data.image_out_roi, self.modified_image_data.image_in_roi =  np.fft.ifft2(np.fft.ifftshift(self.apart_region()))


    def set_view_settings(self, img_view):
        img_view.setAspectLocked(True)
        img_view.setMouseEnabled(x=False, y=False)
        img_view.setMenuEnabled(False)




    def apart_region(self):
        data = self.input_image_data.shifted_ft_data
        region_indices, _ = self.ft_roi.getArraySlice(data, self.ft_item, returnSlice=True)
        mask = np.full(data.shape, False)
        mask[region_indices] = True
        data_in = data * mask
        data_out = data.copy()
        data_out[mask] = 0
        return (data_in, data_out)




    def browse_handler(self, event):
        if self.can_browse:
            if event.double():
                file_dialog = QFileDialog(self)
                file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif)")
                file_dialog.setWindowTitle("Open Image File")
                file_dialog.setFileMode(QFileDialog.ExistingFile)
                if file_dialog.exec_() == QFileDialog.Accepted:
                    selected_file = file_dialog.selectedFiles()[0]
                    self.load_image(selected_file)
                    





    def load_image(self, img_path):
        self.input_image_data.original_image = cv2.imread(img_path)
        self.img_data = cv2.rotate(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), cv2.ROTATE_90_CLOCKWISE)
        self.input_image_data.image_data = self.img_data
        self.input_image_data.original_image_data = self.img_data
        self.modified_image_data.image_data = self.img_data
        self.get_image_attributes(self.input_image_data)
        self.get_image_attributes(self.modified_image_data)
        self.display_img(self.input_image_data.image_data)
        self.ROI_Maxbounds.adjust(0, 0, self.ft_item.width(), self.ft_item.height())
        self.reset_ROI()
        self.loaded = True
        logging.info('Image Loaded')






    def get_image_attributes(self, img):
        if img.image_data is None or img.image_data.size == 0:
            return
        
        img.ft_data = np.fft.fft2(img.image_data)
        img.shifted_ft_data = np.fft.fftshift(img.ft_data)
        img.ft_real, img.ft_imaginary = np.real(img.shifted_ft_data), np.imag(img.shifted_ft_data)
        img.ft_magnitude, img.ft_phase = np.abs(img.shifted_ft_data), np.angle(img.shifted_ft_data)

        self.display_ft(self.FTCombobox.currentText())





    def display_ft(self, mode):
        FT_image  = self.input_image_data.attr(mode)
        if mode != "FT Phase":
            FT_image = np.log(1 + FT_image)

        self.ft_item.setImage(FT_image)





    def display_img(self, data):
        self.img_item.setImage(data)







    def set_brightness_contrast(self, event):
        drag_distance = event.pos() - event.lastPos()
        brightness_delta = drag_distance.x() * 3
        contrast_delta = drag_distance.y() * 3
        self.brightness_val = max(0, min(100, self.brightness_val + brightness_delta))
        self.contrast_val = max(0, min(100, self.contrast_val + contrast_delta))
        adjusted_image = cv2.convertScaleAbs(self.input_image_data.original_image_data, alpha=1 + (self.contrast_val / 100.0),
                                             beta=self.brightness_val)
        
        self.input_image_data.image_data, self.modified_image_data.image_data = adjusted_image, adjusted_image
        if self.ft_enabled:
            self.update_region(finish=True)
        self.get_image_attributes(self.input_image_data)
        self.get_image_attributes(self.modified_image_data)
        self.display_img(adjusted_image)
        logging.debug(f"Brightness: {brightness_delta}, Contrast: {contrast_delta}")








    def reset_brightness_contrast(self):
        self.input_image_data.image_data = self.input_image_data.original_image_data
        self.modified_image_data.image_data = self.input_image_data.original_image_data
        if self.ft_enabled:
            self.update_region(finish=True)
        self.get_image_attributes(self.input_image_data)
        self.get_image_attributes(self.modified_image_data)
        self.display_img(self.input_image_data.original_image_data)

    

    






    
    def center_ROI_to_image(self):
        roi_rect = self.ft_roi.size()
        half_width = roi_rect[0] / 2
        half_height = roi_rect[1] / 2
        center = self.ft_item.boundingRect().center()
        adjusted_center = [center.x() - half_width, center.y()- half_height]
        self.ft_roi.setPos(adjusted_center)
        
        



        
    def reset_ROI(self):
        self.ft_roi.setSize(size=(self.ft_item.boundingRect().width(), self.ft_item.boundingRect().height()))
        self.center_ROI_to_image()
        





if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ImageOutport(None)
    window.show()
    app.exec_()
