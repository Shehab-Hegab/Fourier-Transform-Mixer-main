# Main Program Libraries
# ----------------------
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog
import numpy as np
import cv2



IMG_IN, IMG_OUT = "Img In", "Img Out"
RAW_IMG, CROP_IMG = "Raw Img", "Crop Img"
DATA_IMG_ORIG, DATA_IMG = "Img Data Orig", "Img Data"
DATA_IMG_FT, DATA_IMG_FT_SHIFTED = "Img Data FT", "Img Data FT Shifted"
FT_MAG, FT_PHASE, FT_REAL, FT_IMAG  = "FT Magnitude", "FT Phase", "FT Real", "FT Imaginary"

class mainWindow(QMainWindow):
    # ---------------------------
    def __init__(self):
        # Ui importing and setting window title
        super(mainWindow, self).__init__()
        uic.loadUi("./mainWindow.ui", self)
        self.setWindowTitle("mainWindow")
        self.show()



        self.image_plots = [getattr(self, f"Image{i+1}") for i in range(4)]
        self.sliders = [getattr(self, f"SliderImage{i+1}") for i in range(4)]
        self.slider_labels = [getattr(self, f"LabelImage{i+1}") for i in range(4)]
        self.output_plots = [getattr(self, f"Output{i+1}") for i in range(2)]

        for i in range(4):
            self.sliders[i].valueChanged.connect(self.update_sliders_weights)
            # self.image_plots[i].FTCombobox.currentIndexChanged.connect(lambda: self.fixIndex(self.image_plots[i].FTCombobox.currentIndex()))
            self.image_plots[i].ImageWidget.scene().sigMouseClicked.connect(self.resize_images)
            self.image_plots[i].signal_emitter.change_ROI.connect(lambda ROI: self.modify_all_regions(ROI))
        
        self.ApplyButton.clicked.connect(self.apply_handler)
        self.Output1Check.setChecked(True)


        self.image_weights = [1, 1, 1, 1]

        for plot in self.output_plots:
            plot.FTWidget.setVisible(False)
            plot.FTCombobox.setVisible(False)

        self.mean_val = 0
        self.invert_ROI = False

        self.update_sliders_weights()


        self.other_mode = {
            FT_MAG: FT_PHASE,
            FT_REAL: FT_IMAG,
        }


    def modify_all_regions(self, ROI):
        for plot in self.image_plots:
            if plot.ft_ROI is not ROI:
                plot.ft_ROI.setState(ROI.getState(), update = False)
                plot.ft_ROI.stateChanged(finish = False)
                plot.update_region(flag = False)

    def fixIndex(self, index):
        for port in self.image_plots:
            port.FTCombobox.setCurrentIndex(index)

    
    def update_sliders_weights(self):
        for i, slider in enumerate(self.sliders):
            self.image_weights[i] = slider.value() / 10
            self.slider_labels[i].setText(f"Image{i+1}: {int(self.image_weights[i] * 100)}%")


    def get_mean_val(self):
        self.mean_val = 0
        for i in self.image_weights:
            if i != 0:
                self.mean_val += 1



    def apply_handler(self):
        mode = 'FT Real'
        modified_data = self.apply_weights(mode)

        if mode == 'FT Magnitude':
            modified_data = np.log(modified_data + 1)   

        if self.Output1Check.isChecked():
            self.display_image(self.output_plots[0], modified_data)

        elif self.Output2Check.isChecked():
            self.display_image(self.output_plots[1], modified_data)


    def display_image(self, port, modified_data):
        port.original_image_attr[DATA_IMG] = modified_data
        port.original_image_attr[DATA_IMG_ORIG] = modified_data
        port.calc_imag_ft(port.original_image_attr)
        port.display_img(port.original_image_attr[DATA_IMG])



    def apply_weights(self, mode):
        mode_dat = 0
        other_dat = 0
        other = self.other_mode[mode]

        self.get_mean_val()

        progress_bar = QtWidgets.QProgressBar(self)
        progress_bar.setGeometry(30, 40, 200, 25)
        progress_bar.setMaximum(100)
        self.statusBar().addWidget(progress_bar)

        for index, (port, weight) in enumerate(zip(self.image_plots, self.image_weights)):
            if port.output_image_attr[mode] is not None and port.output_image_attr[mode].any():
                mode_dat += port.output_image_attr[mode] * (weight / self.mean_val)
                other_dat += port.output_image_attr[other] * (weight / self.mean_val)

                progress_value = int((index + 1) / len(self.image_plots) * 100)
                progress_bar.setValue(progress_value)
                QtWidgets.QApplication.processEvents()

        progress_bar.deleteLater() 

        if mode == 'FT Magnitude':
            output = np.clip(np.abs(np.fft.ifft2(mode_dat * np.exp(1j * other_dat))), 0, 255)
        else:
            output = np.clip(np.abs(np.fft.ifft2(mode_dat + (1j * other_dat))), 0, 255)

        return output
    

    def resize_images(self):
        min_height, min_width = self.image_plots[0].original_image_attr[RAW_IMG].shape[:2]
        for port in self.image_plots[1:]:
            img = port.original_image_attr[RAW_IMG]
            height, width = img.shape[:2]
            min_height = min(min_height, height)
            min_width = min(min_width, width)

        for port in self.image_plots:
            port.original_image_attr[RAW_IMG] = cv2.resize(
                port.original_image_attr[RAW_IMG], (min_width, min_height)
            )

            new_img = cv2.rotate(cv2.cvtColor(port.original_image_attr[RAW_IMG], cv2.COLOR_BGR2GRAY), cv2.ROTATE_90_CLOCKWISE)
            port.original_image_attr[DATA_IMG] = new_img
            port.original_image_attr[DATA_IMG_ORIG] = new_img

            port.output_image_attr[DATA_IMG] = new_img
            port.output_image_attr[DATA_IMG_ORIG] = new_img


            port.calc_imag_ft(port.original_image_attr)
            port.calc_imag_ft(port.output_image_attr)

            port.display_img(port.original_image_attr[DATA_IMG_ORIG])



# Main Code
#----------
def main():
    app = QApplication([])
    window = mainWindow()
    app.exec_()

if __name__ == "__main__":
    main()