# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\fastdaq\Documents\Github\Microscope_Thorlabs_Program\UI\UI\GuiDesigner_TWO_CARDS.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1700, 859)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_5.setMaximumSize(QtCore.QSize(398, 742))
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btn_start_stream = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_start_stream.setObjectName("btn_start_stream")
        self.horizontalLayout_4.addWidget(self.btn_start_stream)
        self.btn_stop_stream = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_stop_stream.setObjectName("btn_stop_stream")
        self.horizontalLayout_4.addWidget(self.btn_stop_stream)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.btn_start_stream_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_start_stream_2.setObjectName("btn_start_stream_2")
        self.horizontalLayout_16.addWidget(self.btn_start_stream_2)
        self.btn_stop_stream_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_stop_stream_2.setObjectName("btn_stop_stream_2")
        self.horizontalLayout_16.addWidget(self.btn_stop_stream_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.btn_start_stream_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_start_stream_3.setObjectName("btn_start_stream_3")
        self.horizontalLayout_17.addWidget(self.btn_start_stream_3)
        self.btn_stop_stream_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_stop_stream_3.setObjectName("btn_stop_stream_3")
        self.horizontalLayout_17.addWidget(self.btn_stop_stream_3)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_17)
        self.txtbws_stream_rate = QtWidgets.QTextBrowser(self.groupBox_3)
        self.txtbws_stream_rate.setMinimumSize(QtCore.QSize(351, 41))
        self.txtbws_stream_rate.setMaximumSize(QtCore.QSize(351, 41))
        self.txtbws_stream_rate.setObjectName("txtbws_stream_rate")
        self.verticalLayout_3.addWidget(self.txtbws_stream_rate)
        self.txtbws_stream_rate_2 = QtWidgets.QTextBrowser(self.groupBox_3)
        self.txtbws_stream_rate_2.setMinimumSize(QtCore.QSize(351, 41))
        self.txtbws_stream_rate_2.setMaximumSize(QtCore.QSize(351, 41))
        self.txtbws_stream_rate_2.setObjectName("txtbws_stream_rate_2")
        self.verticalLayout_3.addWidget(self.txtbws_stream_rate_2)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.le_npts_to_plot = QtWidgets.QLineEdit(self.groupBox_3)
        self.le_npts_to_plot.setObjectName("le_npts_to_plot")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_npts_to_plot)
        self.horizontalLayout_7.addLayout(self.formLayout)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.le_buffer_size_MB = QtWidgets.QLineEdit(self.groupBox_3)
        self.le_buffer_size_MB.setObjectName("le_buffer_size_MB")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_buffer_size_MB)
        self.horizontalLayout_5.addLayout(self.formLayout_2)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.gridLayout.addWidget(self.groupBox_3, 0, 0, 1, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.btn_single_acquire = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_single_acquire.setObjectName("btn_single_acquire")
        self.horizontalLayout_2.addWidget(self.btn_single_acquire)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem7)
        self.btn_single_acquire_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_single_acquire_2.setObjectName("btn_single_acquire_2")
        self.horizontalLayout_18.addWidget(self.btn_single_acquire_2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem9)
        self.btn_plot = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_plot.setObjectName("btn_plot")
        self.horizontalLayout_14.addWidget(self.btn_plot)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem10)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem11)
        self.btn_plot_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_plot_2.setObjectName("btn_plot_2")
        self.horizontalLayout_19.addWidget(self.btn_plot_2)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem12)
        self.verticalLayout.addLayout(self.horizontalLayout_19)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.le_npts_post_trigger = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_npts_post_trigger.setObjectName("le_npts_post_trigger")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_npts_post_trigger)
        self.verticalLayout.addLayout(self.formLayout_4)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.le_ppifg = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_ppifg.setObjectName("le_ppifg")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_ppifg)
        self.verticalLayout.addLayout(self.formLayout_3)
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setObjectName("label_14")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.le_ppifg_2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_ppifg_2.setObjectName("le_ppifg_2")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_ppifg_2)
        self.verticalLayout.addLayout(self.formLayout_7)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem13)
        self.btn_apply_ppifg = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_apply_ppifg.setObjectName("btn_apply_ppifg")
        self.horizontalLayout_10.addWidget(self.btn_apply_ppifg)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem14)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_21.addItem(spacerItem15)
        self.btn_apply_ppifg_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_apply_ppifg_2.setObjectName("btn_apply_ppifg_2")
        self.horizontalLayout_21.addWidget(self.btn_apply_ppifg_2)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_21.addItem(spacerItem16)
        self.verticalLayout.addLayout(self.horizontalLayout_21)
        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 2, 1)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.rbtn_walkon_1 = QtWidgets.QRadioButton(self.groupBox_7)
        self.rbtn_walkon_1.setObjectName("rbtn_walkon_1")
        self.verticalLayout_11.addWidget(self.rbtn_walkon_1)
        self.rbtn_walkon_2 = QtWidgets.QRadioButton(self.groupBox_7)
        self.rbtn_walkon_2.setChecked(False)
        self.rbtn_walkon_2.setObjectName("rbtn_walkon_2")
        self.verticalLayout_11.addWidget(self.rbtn_walkon_2)
        self.rbtn_walk_independently = QtWidgets.QRadioButton(self.groupBox_7)
        self.rbtn_walk_independently.setChecked(True)
        self.rbtn_walk_independently.setObjectName("rbtn_walk_independently")
        self.verticalLayout_11.addWidget(self.rbtn_walk_independently)
        self.rbtn_dont_correct = QtWidgets.QRadioButton(self.groupBox_7)
        self.rbtn_dont_correct.setObjectName("rbtn_dont_correct")
        self.verticalLayout_11.addWidget(self.rbtn_dont_correct)
        self.gridLayout.addWidget(self.groupBox_7, 1, 1, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(130, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem17, 2, 1, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.chkbx_save_data = QtWidgets.QCheckBox(self.groupBox_4)
        self.chkbx_save_data.setObjectName("chkbx_save_data")
        self.horizontalLayout_3.addWidget(self.chkbx_save_data)
        spacerItem18 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem18)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.chkbx_save_data_2 = QtWidgets.QCheckBox(self.groupBox_4)
        self.chkbx_save_data_2.setObjectName("chkbx_save_data_2")
        self.horizontalLayout_20.addWidget(self.chkbx_save_data_2)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem19)
        self.verticalLayout_6.addLayout(self.horizontalLayout_20)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_6.addWidget(self.progressBar)
        self.progressBar_2 = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.verticalLayout_6.addWidget(self.progressBar_2)
        self.gridLayout.addWidget(self.groupBox_4, 3, 0, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(20, 134, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem20, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_5, 0, 0, 1, 1)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_9.addWidget(self.label_5)
        self.le_ifgplot_ymax = QtWidgets.QLineEdit(self.groupBox)
        self.le_ifgplot_ymax.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_ymax.setObjectName("le_ifgplot_ymax")
        self.verticalLayout_9.addWidget(self.le_ifgplot_ymax)
        spacerItem21 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem21)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.rbtn_frequency = QtWidgets.QRadioButton(self.groupBox)
        self.rbtn_frequency.setObjectName("rbtn_frequency")
        self.verticalLayout_7.addWidget(self.rbtn_frequency)
        self.rbtn_time = QtWidgets.QRadioButton(self.groupBox)
        self.rbtn_time.setObjectName("rbtn_time")
        self.verticalLayout_7.addWidget(self.rbtn_time)
        self.verticalLayout_9.addLayout(self.verticalLayout_7)
        spacerItem22 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem22)
        self.le_ifgplot_ymin = QtWidgets.QLineEdit(self.groupBox)
        self.le_ifgplot_ymin.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_ymin.setObjectName("le_ifgplot_ymin")
        self.verticalLayout_9.addWidget(self.le_ifgplot_ymin)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_9.addWidget(self.label_6)
        self.horizontalLayout_6.addLayout(self.verticalLayout_9)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gv_ifgplot = PlotWidget(self.groupBox)
        self.gv_ifgplot.setMinimumSize(QtCore.QSize(500, 500))
        self.gv_ifgplot.setObjectName("gv_ifgplot")
        self.verticalLayout_2.addWidget(self.gv_ifgplot)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.le_ifgplot_xmin = QtWidgets.QLineEdit(self.groupBox)
        self.le_ifgplot_xmin.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_xmin.setObjectName("le_ifgplot_xmin")
        self.horizontalLayout.addWidget(self.le_ifgplot_xmin)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        spacerItem23 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem23)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.le_ifgplot_xmax = QtWidgets.QLineEdit(self.groupBox)
        self.le_ifgplot_xmax.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_xmax.setObjectName("le_ifgplot_xmax")
        self.horizontalLayout.addWidget(self.le_ifgplot_xmax)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_13.addWidget(self.groupBox)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_9 = QtWidgets.QLabel(self.groupBox_6)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_10.addWidget(self.label_9)
        self.le_ifgplot_ymax_2 = QtWidgets.QLineEdit(self.groupBox_6)
        self.le_ifgplot_ymax_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_ymax_2.setObjectName("le_ifgplot_ymax_2")
        self.verticalLayout_10.addWidget(self.le_ifgplot_ymax_2)
        spacerItem24 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem24)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.rbtn_frequency_2 = QtWidgets.QRadioButton(self.groupBox_6)
        self.rbtn_frequency_2.setObjectName("rbtn_frequency_2")
        self.verticalLayout_8.addWidget(self.rbtn_frequency_2)
        self.rbtn_time_2 = QtWidgets.QRadioButton(self.groupBox_6)
        self.rbtn_time_2.setObjectName("rbtn_time_2")
        self.verticalLayout_8.addWidget(self.rbtn_time_2)
        self.verticalLayout_10.addLayout(self.verticalLayout_8)
        spacerItem25 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem25)
        self.le_ifgplot_ymin_2 = QtWidgets.QLineEdit(self.groupBox_6)
        self.le_ifgplot_ymin_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_ymin_2.setObjectName("le_ifgplot_ymin_2")
        self.verticalLayout_10.addWidget(self.le_ifgplot_ymin_2)
        self.label_10 = QtWidgets.QLabel(self.groupBox_6)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_10.addWidget(self.label_10)
        self.horizontalLayout_11.addLayout(self.verticalLayout_10)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gv_ifgplot_2 = PlotWidget(self.groupBox_6)
        self.gv_ifgplot_2.setMinimumSize(QtCore.QSize(500, 500))
        self.gv_ifgplot_2.setObjectName("gv_ifgplot_2")
        self.verticalLayout_5.addWidget(self.gv_ifgplot_2)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.le_ifgplot_xmin_2 = QtWidgets.QLineEdit(self.groupBox_6)
        self.le_ifgplot_xmin_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_xmin_2.setObjectName("le_ifgplot_xmin_2")
        self.horizontalLayout_12.addWidget(self.le_ifgplot_xmin_2)
        self.label_11 = QtWidgets.QLabel(self.groupBox_6)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_12.addWidget(self.label_11)
        spacerItem26 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem26)
        self.label_12 = QtWidgets.QLabel(self.groupBox_6)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_12.addWidget(self.label_12)
        self.le_ifgplot_xmax_2 = QtWidgets.QLineEdit(self.groupBox_6)
        self.le_ifgplot_xmax_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ifgplot_xmax_2.setObjectName("le_ifgplot_xmax_2")
        self.horizontalLayout_12.addWidget(self.le_ifgplot_xmax_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_11.addLayout(self.verticalLayout_5)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_13.addWidget(self.groupBox_6)
        self.gridLayout_2.addLayout(self.horizontalLayout_13, 0, 1, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_8)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(4, 0, item)
        self.gridLayout_4.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.horizontalLayout_15.addWidget(self.groupBox_8)
        self.groupBox_9 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.groupBox_9)
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(1)
        self.tableWidget_2.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(4, 0, item)
        self.gridLayout_5.addWidget(self.tableWidget_2, 0, 0, 1, 1)
        self.horizontalLayout_15.addWidget(self.groupBox_9)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:\\Users\\fastdaq\\Documents\\Github\\Microscope_Thorlabs_Program\\UI\\UI\\save_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon)
        self.actionSave.setObjectName("actionSave")
        self.actionSave2 = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Save2/save_icon_2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave2.setIcon(icon1)
        self.actionSave2.setObjectName("actionSave2")
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionSave2)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Data Stream"))
        self.btn_start_stream.setText(_translate("MainWindow", "start stream 1"))
        self.btn_stop_stream.setText(_translate("MainWindow", "stop stream 1"))
        self.btn_start_stream_2.setText(_translate("MainWindow", "start stream 2"))
        self.btn_stop_stream_2.setText(_translate("MainWindow", "stop stream 2"))
        self.btn_start_stream_3.setText(_translate("MainWindow", "start stream 1 + 2"))
        self.btn_stop_stream_3.setText(_translate("MainWindow", "stop stream 1 + 2"))
        self.label_2.setText(_translate("MainWindow", "npts to show in plot"))
        self.label_4.setText(_translate("MainWindow", "storage buffer size (MB)"))
        self.le_buffer_size_MB.setText(_translate("MainWindow", "20"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Single Acquisition"))
        self.btn_single_acquire.setText(_translate("MainWindow", "Acquire + Calc PPIFG 1"))
        self.btn_single_acquire_2.setText(_translate("MainWindow", "Acquire + Calc PPIFG 2"))
        self.btn_plot.setText(_translate("MainWindow", "plot 1"))
        self.btn_plot_2.setText(_translate("MainWindow", "plot 2"))
        self.label_3.setText(_translate("MainWindow", "post-trigger"))
        self.label.setText(_translate("MainWindow", "ppifg 1"))
        self.label_14.setText(_translate("MainWindow", "ppifg 2"))
        self.btn_apply_ppifg.setText(_translate("MainWindow", "apply ppifg 1"))
        self.btn_apply_ppifg_2.setText(_translate("MainWindow", "apply ppifg 2"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Walking IFG\'s"))
        self.rbtn_walkon_1.setText(_translate("MainWindow", "walk on 1"))
        self.rbtn_walkon_2.setText(_translate("MainWindow", "walk on 2"))
        self.rbtn_walk_independently.setText(_translate("MainWindow", "walk independenty"))
        self.rbtn_dont_correct.setText(_translate("MainWindow", "don\'t correct"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Save Streaming Data"))
        self.chkbx_save_data.setText(_translate("MainWindow", "Save Data To Storage Buffer 1"))
        self.chkbx_save_data_2.setText(_translate("MainWindow", "Save Data To Storage Buffer 2"))
        self.groupBox.setTitle(_translate("MainWindow", "Oscilloscope 1"))
        self.label_5.setText(_translate("MainWindow", "y max"))
        self.rbtn_frequency.setText(_translate("MainWindow", "frequency"))
        self.rbtn_time.setText(_translate("MainWindow", "time"))
        self.label_6.setText(_translate("MainWindow", "y min"))
        self.label_8.setText(_translate("MainWindow", "x min"))
        self.label_7.setText(_translate("MainWindow", "x max"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Oscilloscope 2"))
        self.label_9.setText(_translate("MainWindow", "y max"))
        self.rbtn_frequency_2.setText(_translate("MainWindow", "frequency"))
        self.rbtn_time_2.setText(_translate("MainWindow", "time"))
        self.label_10.setText(_translate("MainWindow", "y min"))
        self.label_11.setText(_translate("MainWindow", "x min"))
        self.label_12.setText(_translate("MainWindow", "x max"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Oscilloscope"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Settings for Card 1"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Trigger Level (%)"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Plot Check Level (%)"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Segment Size"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Ext Clk"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Ext Trigger"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget.item(1, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget.item(2, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget.item(3, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget.item(4, 0)
        item.setText(_translate("MainWindow", "hello world"))
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.groupBox_9.setTitle(_translate("MainWindow", "Settings for Card 2"))
        item = self.tableWidget_2.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Trigger Level (%)"))
        item = self.tableWidget_2.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Plot Check Level (%)"))
        item = self.tableWidget_2.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Segment Size"))
        item = self.tableWidget_2.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Ext Clk"))
        item = self.tableWidget_2.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Ext Trigger"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        __sortingEnabled = self.tableWidget_2.isSortingEnabled()
        self.tableWidget_2.setSortingEnabled(False)
        item = self.tableWidget_2.item(0, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget_2.item(1, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget_2.item(2, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget_2.item(3, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tableWidget_2.item(4, 0)
        item.setText(_translate("MainWindow", "hello world"))
        self.tableWidget_2.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Settings"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave2.setText(_translate("MainWindow", "Save2"))
from PlotWidgets import PlotWidget
from . import QRC_file_rc
