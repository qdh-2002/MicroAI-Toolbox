import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QWidget, QVBoxLayout,QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QGroupBox, QLabel, QComboBox,QLineEdit
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import mean_squared_error
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar



class NeuralNetworkVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = plt.figure(figsize=(1,12))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(100, 100)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)


    def draw_neural_network(self, input_neurons, hidden_layers, hidden_neurons, output_neurons):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        layer_sizes = [input_neurons] + hidden_neurons + [output_neurons]
        layers_count = len(layer_sizes)

        # 计算神经元位置
        layer_spacing = 1.0 / (layers_count + 1)
        neuron_radius = 0.005 

        positions = [[(layer_index + 1) * layer_spacing, neuron_index / (layer_size + 1)]
                     for layer_index, layer_size in enumerate(layer_sizes)
                     for neuron_index in range(layer_size)]

        # 绘制神经元
        for layer_index in range(layers_count):
            color = 'red' if layer_index == 0 or layer_index == layers_count - 1 else 'blue'
            for neuron_index in range(layer_sizes[layer_index]):
                position = positions[sum(layer_sizes[:layer_index]) + neuron_index]
                ax.add_patch(plt.Circle(position, neuron_radius, color=color, zorder=4))

        # 连接
        for layer_index in range(layers_count - 1):
            for neuron_index in range(layer_sizes[layer_index]):
                for next_neuron_index in range(layer_sizes[layer_index + 1]):
                    ax.plot([positions[sum(layer_sizes[:layer_index]) + neuron_index][0],
                             positions[sum(layer_sizes[:layer_index + 1]) + next_neuron_index][0]],
                            [positions[sum(layer_sizes[:layer_index]) + neuron_index][1],
                             positions[sum(layer_sizes[:layer_index + 1]) + next_neuron_index][1]],
                            color='black', zorder=3, linewidth = 0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 0.9)
        ax.axis('off')
        self.canvas.draw()

class Window2(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Load an existing model')
        self.setGeometry(100, 100, 580,500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        button_layout = QVBoxLayout()

        self.button_groupbox = QGroupBox("Model Information")
        button_groupbox_layout = QVBoxLayout()
        self.button_groupbox.setLayout(button_groupbox_layout)

        self.iteration_label = QLabel("Summary: N/A")
        self.iteration_label.setAlignment(Qt.AlignTop) 
        button_groupbox_layout.addWidget(self.iteration_label)

        button_layout.addWidget(self.button_groupbox)

        load_button = QPushButton('Load the Existing Model', self)
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)

        testing_button = QPushButton('Test the Model', self)
        testing_button.clicked.connect(self.open_test_model)
        button_layout.addWidget(testing_button)

        predict_button = QPushButton('Predict Output(s)', self)
        predict_button.clicked.connect(self.open_pre_output)
        button_layout.addWidget(predict_button)


        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)


    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '', 'Model files (*.h5)')
        if file_path:
            try:
                self.model = tf.keras.models.load_model(file_path)
                input_shape = self.model.layers[0].input_shape
                self.num_input = input_shape[1] 
        
                output_shape = self.model.layers[-1].output_shape
                self.num_output = output_shape[1] 
                self.show_model_summary()
                print("Model loaded successfully.")
            except Exception as e:
                print("Error loading model:", str(e))
            

    def show_model_summary(self):
        if self.model is None:
            print("Please load a model first.")
            return
        
        # 使用 QPlainTextEdit 显示模型摘要
        summary = self.get_model_summary()
        self.iteration_label.setText(f"Summary: {summary}")

    
        
    def get_model_summary(self):
        try:
            # 使用 TensorFlow 的模型摘要
            summary = []
            self.model.summary(print_fn=lambda x: summary.append(x))
            return '\n'.join(summary)
        except Exception as e:
            return "Error getting model summary: " + str(e)


    def open_test_model(self):
        self.sub_window = Window_Test2(self.model, self.num_input, self.num_output)
        self.sub_window.show()

        # Show the new window
        #self.sub_window.show()
        #self.close()


    def open_pre_output(self):
        # Instantiate another window
        self.pre_output = Pre_Output(self.model, self.num_input, self.num_output)
        self.pre_output.show()

        # Show the new window
        #self.sub_window.show()
        #self.close()


class Pre_Output(QtWidgets.QMainWindow):
    def __init__(self,model, num_input, num_output, parent = None):
        super().__init__(parent)
        self.model = model
        self.num_input = num_input
        self.num_output = num_output
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Predict Output')
        self.setGeometry(300, 150, 680, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # canvas
        self.figure1, self.ax1 = plt.subplots(figsize=(12, 6))
        self.canvas_test1 = FigureCanvas(self.figure1)
        self.ax1.plot([], [], 'b', label='Actual Values')
        self.ax1.plot([], [], 'r', label='Prediction')
        self.ax1.legend()
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output1')
        self.ax1.set_title('Test Results')


        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.canvas_test1)
        #canvas_layout.addWidget(self.canvas_test2)

        #self.input_combo = QComboBox()
        self.output_combo = QComboBox()

        for i in range(1, self.num_output + 1):
            self.output_combo.addItem(f"Output {i}")

        combobox_layout = QHBoxLayout()
        #combobox_layout.addWidget(self.input_combo)
        combobox_layout.addWidget(self.output_combo)

        button_layout = QHBoxLayout()

        upload_button = QPushButton('Upload Input File', self)
        predict_button = QPushButton('Start Predicting', self)
        save_button = QPushButton('Save Predicting Results ', self)

        button_layout.addWidget(upload_button)
        button_layout.addWidget(predict_button)
        button_layout.addWidget(save_button)


        # Connect button clicks to functions
        upload_button.clicked.connect(self.upload_data)
        predict_button.clicked.connect(self.start_predicting)
        save_button.clicked.connect(self.save_prediction_results)
        self.output_combo.currentIndexChanged.connect(self.update_canvas)

        # Main layout combining canvas layout and button layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.addLayout(canvas_layout)
        main_layout.addLayout(combobox_layout)
        main_layout.addLayout(button_layout)

    def update_canvas(self):
            output_index = self.output_combo.currentIndex()  # 获取当前选中的输出索引
            if hasattr(self, 'y_pred') and self.y_pred is not None:
                self.ax1.clear()
                self.ax1.plot(self.sample_indices, self.y_pred[:, output_index], 'r', label='Prediction')
                self.ax1.legend()
                self.ax1.set_xlabel('Samples')
                self.ax1.set_ylabel(f'Output {output_index + 1}') 
                self.ax1.set_title('Test Results')
                self.canvas_test1.draw()


    def load_data(self, file_name):
        data = np.genfromtxt(file_name, delimiter=',')
        self.x_test = data[:, :self.num_input]
        return self.x_test
    

    def upload_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Input File', '', 'CSV Files (*.csv);;All Files (*)', options=options)
        if file_name:
            self.input = self.load_data(file_name)


    #def upload_input_file(self):
        # Open file dialog to select CSV file
    #   file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV File', '', 'CSV Files (*.csv)')
    #    if file_path:
    #        # Perform actions with the selected file (e.g., load data)
    #        print("Selected file:", file_path)

    def start_predicting(self):
        self.y_pred = self.model.predict(self.x_test)
        
        QMessageBox.information(self, 'Prediction Success', 'Predictions have been generated successfully.')


    def start_predicting(self):
        self.y_pred = self.model.predict(self.x_test)
    
        # 检查预测是否成功
        if self.y_pred is not None:
            # Plot actual and predicted values
            self.sample_indices = range(len(self.x_test)) 
            output_index = 0
            self.ax1.clear()
            self.ax1.plot(self.sample_indices, self.y_pred[:,output_index], 'r', label='Prediction')
            self.ax1.legend()
            self.ax1.set_xlabel('Samples')
            self.ax1.set_ylabel('Output1')
            self.ax1.set_title('Test Results')
            self.canvas_test1.draw()
            QMessageBox.information(self, 'Prediction Success', 'Predictions have been generated successfully.')
        else:
            QMessageBox.information(self, 'Prediction Failed', 'An error has occurred, please check if you have uploaded the right file.')
            pass

    def save_prediction_results(self):
        # 选择保存路径
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')
        if save_path:

            column_titles = ','.join([f'output{i+1}' for i in range(self.num_output)])
            prediction_results = self.y_pred[:, :self.num_output]

            # Save prediction results to the selected path
            np.savetxt(save_path, prediction_results, delimiter=',', header=column_titles, comments='')

            # 显示保存成功
            QMessageBox.information(self, 'Save Success', f'Prediction results have been saved to {save_path}')


class Window_Test2(QtWidgets.QMainWindow):
    def __init__(self,model, num_input, num_output, parent = None):
        super().__init__(parent)
        self.model = model
        self.num_input = num_input
        self.num_output = num_output
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Test Model')
        self.setGeometry(300, 150, 680, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # First canvas
        self.figure1, self.ax1 = plt.subplots(figsize=(12, 6))
        self.canvas_test1 = FigureCanvas(self.figure1)
        self.ax1.plot([], [], 'b', label='Actual Values')
        self.ax1.plot([], [], 'r', label='Prediction')
        self.ax1.legend()
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output1')
        self.ax1.set_title('Test Results')


        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.canvas_test1)

        
        self.output_combo = QComboBox()


        for i in range(1, self.num_output + 1):
            self.output_combo.addItem(f"Output {i}")

        combobox_layout = QHBoxLayout()
        combobox_layout.addWidget(self.output_combo)

        # Vertical layout for buttons and labels
        self.button_groupbox = QGroupBox("Testing Information")
        button_layout = QVBoxLayout(self.button_groupbox)

        self.testing_error = QLabel("Testing Error: N/A")
        button_layout.addWidget(self.testing_error)

        self.error2_label = QLabel("Training Error(RMSE): N/A")
        button_layout.addWidget(self.error2_label)

        upload_button = QPushButton('Upload Testing Data', self)
        upload_button.clicked.connect(self.upload_data)
        button_layout.addWidget(upload_button)

        test_button = QPushButton('Test Model', self)
        test_button.clicked.connect(self.test_model)
        button_layout.addWidget(test_button)

        plot_button = QPushButton('Plot', self)
        plot_button.clicked.connect(self.test_plot)
        button_layout.addWidget(plot_button)

        # Main layout combining canvas layout and button layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.addLayout(canvas_layout)
        main_layout.addLayout(combobox_layout)
        main_layout.addWidget(self.button_groupbox)
        self.output_combo.currentIndexChanged.connect(self.update_canvas)

        # Add stretch to main layout
        #main_layout.addStretch(1)

    def update_canvas(self):
            output_index = self.output_combo.currentIndex()  # 获取当前选中的输出索引
            if hasattr(self, 'y_pred_test') and self.y_pred_test is not None:
                self.ax1.clear()
                self.ax1.plot(self.sample_indices, self.y_test[:, output_index], 'b', label='Actual Value')
                self.ax1.plot(self.sample_indices, self.y_pred_test[:, output_index], 'r', label='Prediction')
                self.ax1.legend()
                self.ax1.set_xlabel('Samples')
                self.ax1.set_ylabel(f'Output {output_index + 1}') 
                self.ax1.set_title('Test Results')
                self.canvas_test1.draw()


    def load_data(self, file_name):
        data = np.genfromtxt(file_name, delimiter=',')
        x_train = data[:, :self.num_input]
        y_train = data[:, self.num_input:]
        return x_train, y_train
    


    def upload_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Training Data File', '', 'CSV Files (*.csv);;All Files (*)', options=options)
        if file_name:
            self.x_test, self.y_test = self.load_data(file_name)
    


    def test_model(self):
        if not hasattr(self, 'x_test') or not hasattr(self, 'y_test'):
            QMessageBox.warning(self, 'Warning', 'Please upload testing data first.')
            return
        
        # Make predictions on the testing set
        self.y_pred_test = self.model.predict(self.x_test)

        # Calculate testing error
        testing_error = mean_squared_error(self.y_test, self.y_pred_test)


        self.sample_indices = range(len(self.x_test)) 
        output_index = 0
        

        # Plot actual and predicted values
        self.ax1.clear()
        self.ax1.plot(self.sample_indices, self.y_test[:,output_index], 'b', label='Actual Values')
        self.ax1.plot(self.sample_indices, self.y_pred_test[:,output_index], 'r', label='Prediction')
        self.ax1.legend()
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output1')
        self.ax1.set_title('Test Results')
        self.canvas_test1.draw()



        output_index = 0
        sample_indices = range(len(self.x_test)) 
        Ny = self.num_output

        Etrain = 0
        sum = 0

        array_max = np.zeros(Ny)

        for j in range(Ny):
            for i in range(len(self.x_test)):
                if self.y_test[i,j]>self.y_test[int(array_max[j]),j]:
                    array_max[j]=i
                    
                

        for i in range(len(self.x_test)):
            for j in range(Ny):
                temp = (self.y_test[i,j] - self.y_pred_test[i,j])*(self.y_test[i,j] - self.y_pred_test[i,j])/(self.y_test[int(array_max[j]),j]*self.y_test[int(array_max[j]),j])
                sum = sum + temp


        Etrain = math.sqrt(sum/(len(self.x_test)*Ny))

        ### update RMSE
        self.error2_label.setText(f"Training Error(RMSE): {Etrain:.4f}")


        # Plot actual and predicted values
        #self.ax2.clear()
        #self.ax2.plot(sample_indices, self.y_test[:,output_index+1], 'b', label='Actual Values')
        #self.ax2.plot(sample_indices, y_pred_test[:,output_index+1], 'r', label='Prediction')
        #self.ax2.legend()
        #self.ax2.set_xlabel('Samples')
        #self.ax2.set_ylabel('Output2')
        #self.ax2.set_title('Test Results')
        #self.canvas_test.draw()

        # 更新标签文本
        self.testing_error.setText(f"Testing_error: {testing_error}")
            
        # Update the GUI or show a message box with the testing error
        QMessageBox.information(self, 'Testing Information', f'Testing Error: {testing_error:.4f}')


    def test_plot(self):

        #input_index2 = self.input_combo.currentIndex()
        output_index2 = self.output_combo.currentIndex()
        
        # Plot actual and predicted values
        self.ax1.clear()
        self.ax1.plot(self.sample_indices, self.y_test[:,output_index2], 'b', label='Actual Values')
        self.ax1.plot(self.sample_indices, self.y_pred_test[:,output_index2], 'r', label='Prediction')
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output_{}'.format(output_index2 + 1))
        
        self.canvas_test1.draw()

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    def on_epoch_end(self, epoch, logs=None):
        self.dialog.update_progress(epoch + 1)

    def on_train_end(self, logs=None):
        self.dialog.accept()  # 关闭对话框


class TrainingProgressDialog(QDialog):
    def __init__(self, total_epochs):
        super().__init__()
        self.setWindowTitle("Training Progress")
        self.setGeometry(300, 150, 400, 100)
        self.total_epochs = total_epochs

        self.layout = QVBoxLayout()
        self.label = QLabel(f"Epoch: 0/{total_epochs}")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(total_epochs)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def update_progress(self, epoch):
        self.label.setText(f"Epoch: {epoch}/{self.total_epochs}")
        self.progress_bar.setValue(epoch)
        QApplication.processEvents()  # 更新界面


class Window1(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None
        self.para_activation = 'relu'
        self.para_solve = 'adam'
        self.para_iter = 10000
        self.para_batch = 32
        self.input_neurons = None
        self.hidden_layers = None
        self.hidden_neurons = None
        self.output_neurons = None


    def init_ui(self):
        self.setWindowTitle('Train Model(MLP)')
        self.setGeometry(100, 100, 750, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # First canvas for training loss
        self.figure1, self.ax1 = plt.subplots(figsize=(12, 6))
        self.canvas_trainingloss = FigureCanvas(self.figure1)
        self.ax1.plot([], [], 'b')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_title('Training Loss') 

        # Second canvas for test results
        self.figure2, self.ax2 = plt.subplots(figsize=(12, 6))
        self.canvas_test = FigureCanvas(self.figure2)
        self.ax2.plot([], [], 'b', label='Actual Values')
        self.ax2.plot([], [], 'r', label='Prediction')
        self.ax2.legend()
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_title('Test Results Using Training Data') 

        # Horizontal layout for the canvases
        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.canvas_trainingloss)
        canvas_layout.addWidget(self.canvas_test)

        # Vertical layout for parameter settings
        para_layout = QVBoxLayout()
        self.para_groupbox = QGroupBox("Training Parameters")
        para_groupbox_layout = QVBoxLayout()
        self.para_groupbox.setLayout(para_groupbox_layout)

        # Horizontal layout for each parameter row
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Neurons:"))
        self.input_neurons_edit = QLineEdit()
        input_layout.addWidget(self.input_neurons_edit)
        para_groupbox_layout.addLayout(input_layout)

        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Layers:"))
        self.hidden_layers_edit = QLineEdit()
        hidden_layout.addWidget(self.hidden_layers_edit)
        para_groupbox_layout.addLayout(hidden_layout)

        hidden_neurons_layout = QHBoxLayout()
        hidden_neurons_layout.addWidget(QLabel("Hidden Neurons per Layer:"))
        self.hidden_neurons_edit = QLineEdit()
        hidden_neurons_layout.addWidget(self.hidden_neurons_edit)
        para_groupbox_layout.addLayout(hidden_neurons_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Neurons:"))
        self.output_neurons_edit = QLineEdit()
        output_layout.addWidget(self.output_neurons_edit)
        para_groupbox_layout.addLayout(output_layout)

        self.visualization = NeuralNetworkVisualization()
        para_groupbox_layout.addWidget(self.visualization)

        vis_button = QPushButton("Apply")
        vis_button.clicked.connect(self.vis_model)
        para_groupbox_layout.addWidget(vis_button)

        self.input_neurons_edit.setFixedWidth(300)  
        self.hidden_layers_edit.setFixedWidth(300)
        self.hidden_neurons_edit.setFixedWidth(250)
        self.output_neurons_edit.setFixedWidth(300)
        vis_button.setFixedWidth(400)
        self.para_groupbox.setFixedWidth(450)
        self.para_groupbox.setFixedHeight(400)
    

        para_layout.addWidget(self.para_groupbox)


        # Vertical layout for buttons
        button_layout = QVBoxLayout()
        

        self.button_groupbox = QGroupBox("Training Information")
        button_groupbox_layout = QVBoxLayout()
        self.button_groupbox.setLayout(button_groupbox_layout)

        self.iteration_label = QLabel("Iteration: N/A")
        button_groupbox_layout.addWidget(self.iteration_label)

        self.error_label = QLabel("Training Error(MSE): N/A")
        button_groupbox_layout.addWidget(self.error_label)

        self.error2_label = QLabel("Training Error(RMSE): N/A")
        button_groupbox_layout.addWidget(self.error2_label)

        # 添加 QGroupBox 到 button_layout
        button_layout.addWidget(self.button_groupbox)


        upload_button = QPushButton('Upload Training Data', self)
        upload_button.clicked.connect(self.upload_data)
        button_layout.addWidget(upload_button)

        setting_button = QPushButton('Hyperparameters Settings', self)
        setting_button.clicked.connect(self.open_para)
        button_layout.addWidget(setting_button)

        train_button = QPushButton('Train Model', self)
        train_button.clicked.connect(self.train_model)
        button_layout.addWidget(train_button)

        test_button = QPushButton('Test Model', self)
        test_button.clicked.connect(self.open_test_model)
        button_layout.addWidget(test_button)

        save_button = QPushButton('Save Model', self)
        save_button.clicked.connect(self.save_model)
        button_layout.addWidget(save_button)



        # Add parameter layout and button layout to other layout
        other_layout = QHBoxLayout()
        other_layout.addLayout(para_layout)
        other_layout.addLayout(button_layout)

        # Main layout combining canvas layout and other layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.addLayout(canvas_layout)
        main_layout.addLayout(other_layout)


    def vis_model(self):
        try:
            self.input_neurons = int(self.input_neurons_edit.text())
            self.hidden_layers = int(self.hidden_layers_edit.text())
            
            # Attempt to parse hidden neurons input
            self.hidden_neurons = [int(neurons) for neurons in self.hidden_neurons_edit.text().split(',')]
            
            self.output_neurons = int(self.output_neurons_edit.text())
            
            # Proceed with visualization
            self.visualization.draw_neural_network(self.input_neurons, self.hidden_layers, self.hidden_neurons, self.output_neurons)
        
        except ValueError:
            # Display an error message if there is a value error
            QMessageBox.critical(self, "Input Error", "请检查隐藏层神经元个数输入格式（半角输入法）！\n"
                                 "Please enter valid numbers for hidden neurons. Use comma-separated values (e.g., 10,20,30).")
        except Exception as e:
            # Handle any other exceptions
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")

    

    def upload_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Training Data File', '', 'CSV Files (*.csv);;All Files (*)', options=options)
        if file_name:
            self.x_train, self.y_train = self.load_data(file_name)

        
        #画训练前的图像
        #self.ax.clear()
        #self.ax.plot(self.x_train, self.y_train, 'b', label='Actual Values (Before Training)')
        #self.ax.plot(self.x_train, y_pred_before, 'r', label='Prediction (Before Training)')
        #self.ax.legend()
        #self.ax.set_xlabel('x')
        #self.ax.set_ylabel('y')
        #self.canvas.draw()

    def load_data(self, file_name):
        data = np.genfromtxt(file_name, delimiter=',')
        x_train = data[:, :self.input_neurons]
        y_train = data[:, self.input_neurons:]
        return x_train, y_train
    

    def train_model(self):
        if not hasattr(self, 'x_train') or not hasattr(self, 'y_train'):
            QMessageBox.warning(self, 'Warning', 'Please upload training data first.')
            return

        # Create and train MLP model
        #self.model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
        #self.model.fit(self.x_train, self.y_train)

        #self.iterations = self.model.n_iter_
        #self.training_error = self.model.loss_

        total_epochs = self.para_iter
        progress_dialog = TrainingProgressDialog(total_epochs)
        progress_callback = TrainingCallback(progress_dialog)


        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(self.hidden_neurons[0], activation=self.para_activation, input_shape=(self.input_neurons,)))
        # 添加隐藏层
        for i in range(1, self.hidden_layers):
            self.model.add(tf.keras.layers.Dense(self.hidden_neurons[i], activation=self.para_activation))
        # 输出层
        self.model.add(tf.keras.layers.Dense(self.output_neurons))
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',      # 监控的指标
            patience=10,             # 容忍的训练轮数
            min_delta=0.001,         
            mode='min',              
            restore_best_weights=True 
)
        # 编译模型
        self.model.compile(optimizer=self.para_solve, loss='mean_squared_error')
        # 训练模型
        progress_dialog.show()
        history = self.model.fit(self.x_train, self.y_train, epochs=self.para_iter, verbose=0, batch_size = self.para_batch, callbacks=progress_callback)



        self.iterations = len(history.history['loss'])  # number of training iterations
        self.training_error = history.history['loss'][-1]  # training loss after all iterations

        # 计算 training loss
        training_loss_values = history.history['loss']


        ### 用sklearn包
        ####  计算 training loss
        #training_loss_values = []

        #for iteration in range(1, max_iter+1): 
        #    self.model.partial_fit(self.x_train, self.y_train)  
        #    training_loss = self.model.loss_
        #    if iteration == self.iterations:
        #        self.training_error = self.model.loss_
        #    training_loss_values.append(training_loss)

        # Plot training loss curve
        self.ax1.clear()
        self.ax1.plot(range(1, self.para_iter+1), training_loss_values, 'b', label='Training Loss')
        self.ax1.legend()
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Training Loss')
        self.canvas_trainingloss.draw()
        self.ax1.set_title('Training Loss')
        

        #QMessageBox.information(self, 'Training Information', f'Iterations: {self.iterations}\nTraining Error: {self.training_error:.4f}')
        # 更新标签文本
        self.iteration_label.setText(f"Iteration: {self.iterations}")
        self.error_label.setText(f"Training Error(MSE): {self.training_error:.4f}")
    

        # 画training value以及真实值

        # Make predictions on the testing set
        y_pred_train = self.model.predict(self.x_train)

        output_index = 0
        sample_indices = range(len(self.x_train)) 
        Ny = self.output_neurons

        Etrain = 0
        sum = 0

        array_max = np.zeros(Ny)

        for j in range(Ny):
            for i in range(len(self.x_train)):
                if self.y_train[i,j]>self.y_train[int(array_max[j]),j]:
                    array_max[j]=i
                    
                

        for i in range(len(self.x_train)):
            for j in range(Ny):
                temp = (self.y_train[i,j] - y_pred_train[i,j])*(self.y_train[i,j] - y_pred_train[i,j])/(self.y_train[int(array_max[j]),j]*self.y_train[int(array_max[j]),j])
                sum = sum + temp


        Etrain = math.sqrt(sum/(len(self.x_train)*Ny))

        ### update RMSE
        self.error2_label.setText(f"Training Error(RMSE): {Etrain:.4f}")



        # Plot actual and predicted values
        self.ax2.clear()
        self.ax2.plot(sample_indices, self.y_train[:,output_index], 'b', label='Actual Values')
        self.ax2.plot(sample_indices, y_pred_train[:,output_index], 'r', label='Prediction')
        self.ax2.legend()
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_title('Test Results Using Training Data')
        self.canvas_test.draw()

        self.open_TrainedEnd_window()

    def test_model(self):
        if not hasattr(self, 'x_train') or not hasattr(self, 'y_train'):
            QMessageBox.warning(self, 'Warning', 'Please upload training data first.')
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Testing Data File', '', 'CSV Files (*.csv);;All Files (*)', options=options)

        if file_name:
            x_test, y_test = self.load_data(file_name)

            # Make predictions on the testing set
            y_pred_test = self.model.predict(x_test)

            # Calculate testing error
            testing_error = mean_squared_error(y_test, y_pred_test)


            input_index = 0
            output_index = 0

            # Plot actual and predicted values
            self.ax2.clear()
            self.ax2.plot(x_test[:,input_index], y_test[:,output_index], 'b', label='Actual Values')
            self.ax2.plot(x_test[:,input_index], y_pred_test[:,output_index], 'r', label='Prediction')
            self.ax2.legend()
            self.ax2.set_xlabel('x')
            self.ax2.set_ylabel('y')
            self.ax2.set_title('Test Results')
            self.canvas_test.draw()


            # 更新标签文本
            self.testing_error.setText(f"Testing_error: {testing_error}")
            

            # Update the GUI or show a message box with the testing error
            QMessageBox.information(self, 'Testing Information', f'Testing Error: {testing_error:.4f}')


    #########################
    #####参数设置窗口#########

    def open_para(self):
        # Instantiate another window
        window2 = Window_Para(self)
        if window2.exec_():  # 显示子窗口
            pass  # 可添加对话框关闭后的处理

    def pass_para(self, activ, optm, iter, batch_size):
        self.para_activation = activ
        self.para_solve = optm
        self.para_iter = int(iter)
        self.para_batch = int(batch_size)




    def open_TrainedEnd_window(self):
        # 实例化一个对话框类
        self.dlg = Window_TrainedEnd()        
        # 显示对话框，代码阻塞在这里，
        # 等待对话框关闭后，才能继续往后执行
        self.dlg.exec_()
    
    def handle_act_changed(self):
        print("New parameter received:", self.window_para.para_act)
        self.para_activation = self.window_para.para_act



    def open_test_model(self):
        # Instantiate another window
        self.num_input = int(self.input_neurons_edit.text())
        self.num_output = int(self.output_neurons_edit.text())
        self.sub_window = Window_Test(self.model, self.num_input, self.num_output)
        self.sub_window.show()

        # Show the new window
        #self.sub_window.show()
        #self.close()

    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please train the model first.')
            return

        # Get the file path to save the model
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Model', '', 'H5 Files (*.h5);;All Files (*)', options=options)

        if file_name:
            # Append '.h5' extension if not provided
            if not file_name.endswith('.h5'):
                file_name += '.h5'

            # Save the model
            self.model.save(file_name)

            # Show success message
            QMessageBox.information(self, 'Model Saved', f'Model saved successfully to:\n{file_name}')

        

class Window_TrainedEnd(QtWidgets.QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Trained Result')
        self.setText('Model Trained Successfully!')


class Window_Test(QtWidgets.QMainWindow):
    def __init__(self,model, num_input, num_output, parent = None):
        super().__init__(parent)
        self.model = model
        self.num_input = num_input
        self.num_output = num_output
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Test Model')
        self.setGeometry(300, 150, 680, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # First canvas
        self.figure1, self.ax1 = plt.subplots(figsize=(12, 6))
        self.canvas_test1 = FigureCanvas(self.figure1)
        self.ax1.plot([], [], 'b', label='Actual Values')
        self.ax1.plot([], [], 'r', label='Prediction')
        self.ax1.legend()
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output1')
        self.ax1.set_title('Test Results')

        # Horizontal layout for the canvases
        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.canvas_test1)
        #canvas_layout.addWidget(self.canvas_test2)

        
        #self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        #for i in range(1, self.num_input + 1):
        #    self.input_combo.addItem(f"Input {i}")

        for i in range(1, self.num_output + 1):
            self.output_combo.addItem(f"Output {i}")

        combobox_layout = QHBoxLayout()
        #combobox_layout.addWidget(self.input_combo)
        combobox_layout.addWidget(self.output_combo)

        # Vertical layout for buttons and labels
        self.button_groupbox = QGroupBox("Testing Information")
        button_layout = QVBoxLayout(self.button_groupbox)

        self.testing_error = QLabel("Testing Error: N/A")
        button_layout.addWidget(self.testing_error)

        self.error2_label = QLabel("Training Error(RMSE): N/A")
        button_layout.addWidget(self.error2_label)

        upload_button = QPushButton('Upload Testing Data', self)
        upload_button.clicked.connect(self.upload_data)
        button_layout.addWidget(upload_button)

        test_button = QPushButton('Test Model', self)
        test_button.clicked.connect(self.test_model)
        button_layout.addWidget(test_button)

        plot_button = QPushButton('Plot', self)
        plot_button.clicked.connect(self.test_plot)
        button_layout.addWidget(plot_button)

        # Main layout combining canvas layout and button layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.addLayout(canvas_layout)
        main_layout.addLayout(combobox_layout)
        main_layout.addWidget(self.button_groupbox)

    
    def load_data(self, file_name):
        data = np.genfromtxt(file_name, delimiter=',')
        x_train = data[:, :self.num_input]
        y_train = data[:, self.num_input:]
        return x_train, y_train
    


    def upload_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Training Data File', '', 'CSV Files (*.csv);;All Files (*)', options=options)
        if file_name:
            self.x_test, self.y_test = self.load_data(file_name)
    


    def test_model(self):
        if not hasattr(self, 'x_test') or not hasattr(self, 'y_test'):
            QMessageBox.warning(self, 'Warning', 'Please upload testing data first.')
            return
        
        # Make predictions on the testing set
        self.y_pred_test = self.model.predict(self.x_test)

        # Calculate testing error
        testing_error = mean_squared_error(self.y_test, self.y_pred_test)


        self.sample_indices = range(len(self.x_test)) 
        output_index = 0
        

        # Plot actual and predicted values
        self.ax1.clear()
        self.ax1.plot(self.sample_indices, self.y_test[:,output_index], 'b', label='Actual Values')
        self.ax1.plot(self.sample_indices, self.y_pred_test[:,output_index], 'r', label='Prediction')
        self.ax1.legend()
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output1')
        self.ax1.set_title('Test Results')
        self.canvas_test1.draw()


        # 更新标签文本
        self.testing_error.setText(f"Testing_error: {testing_error}")
            
        # Update the GUI or show a message box with the testing error
        QMessageBox.information(self, 'Testing Information', f'Testing Error: {testing_error:.4f}')

        output_index = 0
        sample_indices = range(len(self.x_test)) 
        Ny = self.num_output

        Etrain = 0
        sum = 0

        array_max = np.zeros(Ny)

        for j in range(Ny):
            for i in range(len(self.x_test)):
                if self.y_test[i,j]>self.y_test[int(array_max[j]),j]:
                    array_max[j]=i
                    
                

        for i in range(len(self.x_test)):
            for j in range(Ny):
                temp = (self.y_test[i,j] - self.y_pred_test[i,j])*(self.y_test[i,j] - self.y_pred_test[i,j])/(self.y_test[int(array_max[j]),j]*self.y_test[int(array_max[j]),j])
                sum = sum + temp


        Etrain = math.sqrt(sum/(len(self.x_test)*Ny))

        ### update RMSE
        self.error2_label.setText(f"Training Error(RMSE): {Etrain:.4f}")


    def test_plot(self):

        #input_index2 = self.input_combo.currentIndex()
        output_index2 = self.output_combo.currentIndex()
        
        # Plot actual and predicted values
        self.ax1.clear()
        self.ax1.plot(self.sample_indices, self.y_test[:,output_index2], 'b', label='Actual Values')
        self.ax1.plot(self.sample_indices, self.y_pred_test[:,output_index2], 'r', label='Prediction')
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Output_{}'.format(output_index2 + 1))
        
        self.canvas_test1.draw()

        


class Window_Para(QtWidgets.QDialog):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Parameters Settings')
        
        layout = QtWidgets.QVBoxLayout()


        ########激活函数############
        activation = QLabel('Activation')
        layout.addWidget(activation)

        self.combo_box_act = QComboBox(self)
        self.combo_box_act.addItem('relu')
        self.combo_box_act.addItem('sigmoid')
        self.combo_box_act.addItem('tanh')
        #self.combo_box_act.setCurrentText(para_activation)
        #self.combo_box_act.currentIndexChanged.connect(self.on_act_change)
        layout.addWidget(self.combo_box_act)
        

       
        ########优化算法############
        solver = QLabel('Solver')
        layout.addWidget(solver)

        self.combo_box_sol = QComboBox(self)
        self.combo_box_sol.addItem('Adam')
        self.combo_box_sol.addItem('SGD')
        self.combo_box_sol.addItem('LBFGS')
        self.combo_box_sol.addItem('Levenberg-Marquardt')
        #combo_box_sol.currentIndexChanged.connect(self.on_sol_change)
        layout.addWidget(self.combo_box_sol)

        ##########最大迭代次数##########
        max_iter = QLabel('Maximum Number of Iterations:')
        layout.addWidget(max_iter)
        self.num_max_iter = QLineEdit(self)
        layout.addWidget(self.num_max_iter)


        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size = QLineEdit()
        batch_size_layout.addWidget(self.batch_size)
        layout.addLayout(batch_size_layout)

        self.setLayout(layout)
        self.setWindowTitle('Configurator')
        self.setGeometry(100, 100, 200, 300)
        self.show()

        confirm_button = QPushButton('Confirm')
        confirm_button.clicked.connect(self.confirm)
        layout.addWidget(confirm_button)

        
        self.setLayout(layout)


        
    def confirm(self):
        activ = self.combo_box_act.currentText()
        optm = self.combo_box_sol.currentText()
        iter = self.num_max_iter.text()
        batch_size = self.batch_size.text()

        self.accept()  # 关闭子窗口
        self.parent().pass_para(activ, optm, iter, batch_size)




class Window_LoadModel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Train Model')
        self.setGeometry(100, 100, 800, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)





class MainWindow_sub(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('MicroAI')
        self.setGeometry(100, 100, 600, 280)
        

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)



        text_label_en = QtWidgets.QLabel("Welcome to MicroAI Toolbox! This is a toolbox for training deep learning models, where you can upload your own training dataset for model training. Additionally, we offer an interface for adjusting certain hyperparameters, allowing you to modify aspects such as the number of hidden layers in an MLP, the number of neurons per layer, activation functions, and more according to your training needs. After training is complete, you can upload a testing dataset to evaluate your model. You also have the option to upload a trained model file (.h5 file) for testing or making predictions on inputs.", self)
        text_label_en.setWordWrap(True)  # 开启文字自动换行
        layout.addWidget(text_label_en)


        text_label = QtWidgets.QLabel("欢迎使用MicroAI Toolbox! 这是一个深度学习模型训练的工具箱，"
                                      "您可以选择上传自己的训练数据集来进行模型的训练。同时，"
                                      "我们提供了调整部分超参数的接口，您可以根据训练需求调整MLP的隐藏层层数、每层的神经元个数、激活函数等。训练完成后您可以上传测试数据集进行对模型的测试。"
                                      "您也可以选择上传训练好的模型文件（.h5文件），从而进行测试或对输入进行预测。", self)
        text_label.setWordWrap(True)  # 开启文字自动换行

        layout.addWidget(text_label)
        
        self.button_train = QtWidgets.QPushButton('Train A New Model')
        self.button_train.clicked.connect(self.open_new_window1)
        layout.addWidget(self.button_train)

        self.button_train = QtWidgets.QPushButton('Load An Existing Model')
        self.button_train.clicked.connect(self.open_new_window2)
        layout.addWidget(self.button_train)


    def open_new_window1(self):
        # Instantiate another window
        self.window_model_selection = Window_Model_Selection()
        # Show the new window
        self.window_model_selection.show()
        #self.close()

    def open_new_window2(self):
        # Instantiate another window
        #self.window2 = Window_LoadModel()
        # Show the new window
        self.window2 = Window2()
        self.window2.show()
        #self.close()

    
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('MicroAI')
        self.setGeometry(100, 100, 400, 300)
        

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # 添加图片
        image_label = QLabel(self)
        pixmap = QtGui.QPixmap('/img/img/img.jpg') 
        pixmap = pixmap.scaledToWidth(400)  
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)



        #self.setStyleSheet("QMainWindow {background-image: url('/Users/heqiudi/Documents/COGS118/screen');}")

        self.button_enter = QtWidgets.QPushButton('Enter')
        layout.addWidget(self.button_enter)
        self.button_enter.clicked.connect(self.open_sub_main_window)


        self.button_exit = QtWidgets.QPushButton('Exit')
        layout.addWidget(self.button_exit)
        self.button_exit.clicked.connect(self.close_window)



    def open_sub_main_window(self):
        # Instantiate another window
        self.window1 = MainWindow_sub()
        # Show the new window
        self.window1.show()
        self.close()

    def close_window(self):
        sys.exit(0)

class Window_Model_Selection(QtWidgets.QDialog):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Model Selection')
        
        layout = QtWidgets.QVBoxLayout()


        model_selector = QLabel('Please Select the Neural Network Architecture!')
        layout.addWidget(model_selector)

        self.combo_box_sele = QComboBox(self)
        self.combo_box_sele.addItem('MLP')
        self.combo_box_sele.addItem('CNN')



        self.setLayout(layout)
        self.setGeometry(100, 100, 360, 180)
        self.show()

        self.button_layout = QHBoxLayout()

        confirm_button = QPushButton('Confirm')
        confirm_button.clicked.connect(self.confirm)
        self.button_layout.addWidget(confirm_button)

        back_button = QPushButton('Back')
        back_button.clicked.connect(self.back)
        self.button_layout.addWidget(back_button)

        layout.addWidget(self.combo_box_sele)
        layout.addLayout(self.button_layout)

        
    def confirm(self):
        self.train_window_mlp = Window1()
        self.train_window_mlp.show()
        self.close()
    
    def back(self):
        self.close()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
