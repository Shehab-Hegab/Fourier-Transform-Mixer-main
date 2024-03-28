import cv2
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5 import QtCore, QtWidgets, uic
import numpy as np
import logging


logging.basicConfig(filename = 'application.log', level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s', filemode='w')



class MixerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MixerWindow, self).__init__()
        uic.loadUi("./mainWindow.ui", self)
        self.setWindowTitle("mainWindow")
        self.show()

        self.mean_val = 0

        for plot in [self.Output1, self.Output2]:
            plot.FTWidget.setVisible(False)
            plot.FTCombobox.setVisible(False)


        self.image_plots = [getattr(self, f"Image{i+1}") for i in range(4)]
        self.sliders = [getattr(self, f"SliderImage{i+1}") for i in range(4)]
        self.slider_labels = [getattr(self, f"LabelImage{i+1}") for i in range(4)]

        self.weights = [1, 1, 1, 1]            
            
        self.comp_modes = { "FT Magnitude": "FT Phase", "FT Real": "FT Imaginary", "FT Phase": "FT Magnitude", "FT Imaginary": "FT Real"}

        [slider.valueChanged.connect(self.update_weights) for slider in self.sliders]

        for port in self.image_plots:
            port.ImageWidget.scene().sigMouseClicked.connect(self.resize_images)
            port.sig_emitter.sig_ROI_changed.connect(lambda roi: self.set_all_ROI(roi))

        
        self.ApplyButton.clicked.connect(self.apply_handler)
        self.Output1Check.setChecked(True)







    def set_all_ROI(self, roi):
        new_state = roi.getState()
        for port in self.image_plots:
            if (port.ft_roi is not roi) and (port.loaded):
                port.ft_roi.setState(new_state, update=False)
                port.ft_roi.stateChanged(finish=False)
                port.update_region(finish=False)







    def update_weights(self):
        for i in range(4):
            self.weights[i] = self.sliders[i].value() / 10
            self.slider_labels[i].setText(f"Image{i+1}: {int(self.weights[i] * 100)}%")





    def apply_handler(self):
        modified_data = self.apply_weights()
        modified_data = np.log(modified_data + 1)
        if self.Output1Check.isChecked():
            self.display_output(self.Output1, modified_data)

        elif self.Output2Check.isChecked():
            self.display_output(self.Output2, modified_data)







    def display_output(self, output, data):
        output.input_image_data.image_data, output.input_image_data.original_image_data = data, data
        output.get_image_attributes(output.input_image_data)
        output.display_img(output.input_image_data.image_data)







    def show_error_message(self, message):
        QMessageBox.critical(self, 'Invalid Pairs', message, QMessageBox.Ok)





    def update_progress_bar(self, progress_bar, index, total):
        progress_value = int((index + 1) / total * 100)
        progress_bar.setValue(progress_value)
        QApplication.processEvents()





    def process_port(self, port, weight, mode, comp_mode, other_weight):
        if self.ROICheckbox.isChecked():
            port.modified_image_data.image_data = port.modified_image_data.image_out_roi
        else:
            port.modified_image_data.image_data = port.modified_image_data.image_in_roi
        port.get_image_attributes(port.modified_image_data)

        mode_data = port.modified_image_data.attr(mode) * (weight / self.mean_val)
        comp_mode_data = port.modified_image_data.attr(comp_mode) * (other_weight / self.mean_val)

        return mode_data, comp_mode_data





    def apply_weights(self):
        pairs = []
        for i, port in enumerate(self.image_plots):
            if port.loaded:
                pairs.append(port.FTCombobox.currentText())

        if not all(pair in {'FT Magnitude', 'FT Phase'} for pair in pairs) and not all(pair in {'FT Real', 'FT Imaginary'} for pair in pairs):
            self.show_error_message('Choose valid pairs: Mag/Phase or Real/Imaginary')
            return

        mode_data, comp_mode_data = 0, 0
        self.mean_val += sum(1 for i in self.weights if i != 0)
        logging.debug(f"Mean Value: {self.mean_val}")

        progress_bar = QtWidgets.QProgressBar(self)
        progress_bar.setGeometry(30, 40, 200, 25)
        progress_bar.setMaximum(100)
        self.statusBar().addWidget(progress_bar)

        for i in range(4):
            if self.image_plots[i].loaded:
                mode = self.image_plots[i].FTCombobox.currentText()
                comp_mode = self.comp_modes[mode]

                if mode in ["FT Phase", "FT Imaginary"]:
                    mode, comp_mode = comp_mode, mode

                other_weight = 0 if self.weights[i] == 0 else 1

                mode_data_add, comp_mode_data_add = self.process_port(self.image_plots[i], self.weights[i], mode, comp_mode, other_weight)
                mode_data += mode_data_add
                comp_mode_data += comp_mode_data_add

                self.update_progress_bar(progress_bar, i, len(self.image_plots))


        progress_bar.deleteLater()

        if mode == "FT Magnitude":
            output = np.clip(np.abs(np.fft.ifft2(mode_data * np.exp(1j * comp_mode_data))), 0, 255)
        else:
            output = np.clip(np.abs(np.fft.ifft2(mode_data + (1j * comp_mode_data))), 0, 255)

        return output








    def resize_images(self):
        min_height, min_width = float('inf'), float('inf')
        for port in self.image_plots:
            if port.input_image_data.original_image is not None:
                height, width = port.input_image_data.original_image.shape[:2]
                min_height = min(min_height, height)
                min_width = min(min_width, width)

        for port in self.image_plots:
            if port.input_image_data.original_image is not None:
                port.input_image_data.original_image = cv2.resize(
                    port.input_image_data.original_image, (min_width, min_height)
                )

                new_img = cv2.rotate(cv2.cvtColor(port.input_image_data.original_image, cv2.COLOR_BGR2GRAY), cv2.ROTATE_90_CLOCKWISE)
                port.input_image_data.image_data = new_img
                port.input_image_data.original_image_data = new_img

                port.modified_image_data.image_data = new_img
                port.modified_image_data.original_image_data = new_img

                port.get_image_attributes(port.input_image_data)
                port.get_image_attributes(port.modified_image_data)

                port.display_img(port.input_image_data.original_image_data)

                port.sig_emitter.sig_ROI_changed.emit(port.ft_roi)
                



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MixerWindow()
    window.show()
    app.exec_()
