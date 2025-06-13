#!/usr/bin/env python3
# coding: utf-8


from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog
from function import function, function2
from LogRegressionDetailed_24Bus_24Period00 import *
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

class TrainThread(QThread):
    # 添加信号用于线程与主线程通信
    log_signal = pyqtSignal(str)
    
    def __init__(self, train_file, test_file, num_epochs, batch_size):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_path = "models/power_system_transformer_model.pth"
        self.input_dim = 576
        self.output_dim = 792
    
    def run(self):
        try:
            self.model_path, self.input_dim, self.output_dim = train_main(self.train_file, self.test_file, self.num_epochs, self.batch_size, log_callback=self.log_signal.emit)
            # self.log_signal.emit("训练完成！")
        except Exception as e:
            print(f"训练错误: {str(e)}")
            self.log_signal.emit(f"训练错误: {str(e)}")

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # 文本显示框
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        layout.addWidget(self.text_display)
        
        # 打开文件按钮
        btn_open_file = QPushButton('选择训练数据', self)
        btn_open_file.clicked.connect(self.open_train_file)
        layout.addWidget(btn_open_file)

         # 打开文件按钮
        btn_open_file2 = QPushButton('选择验证数据', self)
        btn_open_file2.clicked.connect(self.open_test_file)
        layout.addWidget(btn_open_file2)

        # 参数输入框
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setPlaceholderText('输入batch size (默认:64)')
        layout.addWidget(self.batch_size_input)
        
        self.epoch_input = QLineEdit()
        self.epoch_input.setPlaceholderText('输入epoch数 (默认:50)')
        layout.addWidget(self.epoch_input)
        
        # 开始训练按钮
        btn_train = QPushButton('开始训练', self)
        btn_train.clicked.connect(self.train)
        layout.addWidget(btn_train)

        # 加载模型按钮
        btn_load = QPushButton('加载模型', self)
        btn_load.clicked.connect(self.load_model)
        layout.addWidget(btn_load)

        btn_open_data = QPushButton('选择输入数据', self)
        btn_open_data.clicked.connect(self.open_data_file)
        layout.addWidget(btn_open_data)

        btn_predict = QPushButton('运行预测', self)
        btn_predict.clicked.connect(self.predict)
        layout.addWidget(btn_predict)
        
        
        self.setLayout(layout)
        self.setWindowTitle('SCUC')
        self.setGeometry(300, 300, 800, 500)
    
    
    def open_train_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '打开训练数据文件', '', '文本文件 (*.txt);;所有文件 (*)')
        if file_path:
            try:
                self.train_file = file_path
                self.text_display.append(f"成功打开训练数据文件: {file_path}")
            except Exception as e:
                self.text_display.append(f"打开文件失败: {str(e)}")

    def open_test_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '打开测试数据文件', '', '文本文件 (*.txt);;所有文件 (*)')
        if file_path:
            try:
                self.test_file = file_path
                self.text_display.append(f"成功打开测试数据文件: {file_path}")
            except Exception as e:
                self.text_display.append(f"打开文件失败: {str(e)}")

    def train(self):
        if hasattr(self, 'train_file') and hasattr(self, 'test_file'):
            # 获取输入参数
            try:
                batch_size = int(self.batch_size_input.text()) if self.batch_size_input.text() else 64
                epoch = int(self.epoch_input.text()) if self.epoch_input.text() else 50
            except ValueError:
                self.text_display.append("请输入有效的数字参数")
                return
                
            self.text_display.append(f"正在训练，batch size: {batch_size}, epoch: {epoch}")
            self.train_thread = TrainThread(self.train_file, self.test_file, epoch, batch_size)
            self.train_thread.finished.connect(self.on_train_finished)
            self.train_thread.log_signal.connect(self.log_message)  # 连接日志信号
            
            self.train_thread.start()  # 在后台线程运行训练
        else:
            self.text_display.setText("请先选择训练数据文件和测试数据文件")

    def load_model(self):
            # 默认模型参数
        default_model_path = "models/power_system_transformer_model.pth"
        default_input_dim = 576
        default_output_dim = 792
        
        try:
            if hasattr(self, 'train_thread') and hasattr(self.train_thread, 'model_path'):
                # 加载训练后的模型
                self.text_display.append("正在加载训练后的模型...")
                self.model = load_pretrained_model(self.train_thread.model_path,
                                                self.train_thread.input_dim,
                                                self.train_thread.output_dim)
                self.text_display.append("训练后的模型加载完成")
            else:
                # 加载默认模型
                self.text_display.append("正在加载默认模型...")
                self.model = load_pretrained_model(default_model_path,
                                                default_input_dim,
                                                default_output_dim)
                self.text_display.append("默认模型加载完成")
        except Exception as e:
            self.text_display.append(f"模型加载失败: {str(e)}")

    def open_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '打开输入数据文件', '', '文本文件 (*.txt);;所有文件 (*)')
        if file_path:
            try:
                self.data_file = file_path
                self.text_display.append(f"成功打开输入数据文件: {file_path}")
            except Exception as e:
                self.text_display.append(f"打开文件失败: {str(e)}")

    def predict(self):
        if hasattr(self, 'model') and hasattr(self, 'data_file'):
            try:
                # 读取输入数据
                start_time = time.time()
                predictions = predict_from_file(self.model, self.data_file)
                end_time = time.time()
                time_consumed = end_time - start_time
                self.text_display.append(f"预测结果: {predictions}")
                self.text_display.append(f"预测耗时: {time_consumed:.4f} 秒")
            except Exception as e:
                self.text_display.append(f"预测过程出现错误: {str(e)}")
        else:
            self.text_display.append("请先加载模型和选择输入数据文件")
                
    def on_train_finished(self):
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("训练完成！")
        msg.setWindowTitle("提示")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def log_message(self, message):
        self.text_display.append(message)
        QApplication.processEvents()  # 实时更新UI

if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()