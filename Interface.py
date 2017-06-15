# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '11.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D,Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack, row_stack, asarray
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from pandas import read_csv
from pymystem3 import Mystem
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from collections import Counter
import Twarc
import json
import codecs

import tweepy
auth = tweepy.OAuthHandler('DkNRJdzhUnThKJ7G5C9IftdUp', 'C14fr0ews91xJI8AH1I3BNhZrZ2gdlyz9KqnXFPQOnmZqJUmc7')
auth.set_access_token('866132837082296320-GRx4gxwbRVffxwXgMhjZhXbxgn4RaM0', 'rhtMycE2gFiJchJVIJtlEIf7qgkvqtCbmue9rPDoXEpkt')
api = tweepy.API(auth)

from PyQt5 import QtCore, QtGui, QtWidgets

# Create a summary of a tweet, only showing relevant fields.
def summarize(tweet, extra_fields = None):
    new_tweet = {}
    for field, value in tweet.items():
        if field in ["text", "id_str", "screen_name", "retweet_count", "favorite_count", "in_reply_to_status_id_str", "in_reply_to_screen_name", "in_reply_to_user_id_str"] and value is not None:
            new_tweet[field] = value
        elif extra_fields and field in extra_fields:
            new_tweet[field] = value
        elif field in ["retweeted_status", "quoted_status", "user"]:
            new_tweet[field] = summarize(value)
    return new_tweet

# Print out a tweet, with optional colorizing of selected fields.
def dump(tweet, colorize_fields=None, summarize_tweet=True):
    colorize_field_strings = []
    for line in json.dumps(summarize(tweet) if summarize_tweet else tweet, indent=4, sort_keys=True).splitlines():
        colorize = False
        for colorize_field in colorize_fields or []:
            if "\"{}\":".format(colorize_field) in line:
                print("\x1b" + line + "\x1b")
                break
        else:
            print(line)


tweet = list(t.hydrate(['']))[0]
dump(summarize(tweet, extra_fields=['in_reply_to_status_id_str', 'in_reply_to_user_id']), colorize_fields=['in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_screen_name', 'in_reply_to_user_id', 'in_reply_to_user_id_str'], summarize_tweet=False)



def stemconvtext(text):
    return(''.join(Mystem().lemmatize(text)))

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='linear',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('linear'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

class Ui_MainWindow(object):
    def load_tweet(self):
        tweet = api.get_status(self.plainTextEdit_2.toPlainText())
        self.textBrowser_2.setPlainText(tweet.text)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(911, 597)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.layoutWidget = QtWidgets.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(510, 10, 371, 411))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.checkBox = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBox.setObjectName("checkBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.checkBox)
        self.dateEdit_2 = QtWidgets.QDateEdit(self.layoutWidget)
        self.dateEdit_2.setMinimumDateTime(QtCore.QDateTime(QtCore.QDate(2000, 1, 1), QtCore.QTime(0, 0, 0)))
        self.dateEdit_2.setMaximumDate(QtCore.QDate(2017, 6, 30))
        self.dateEdit_2.setObjectName("dateEdit_2")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.dateEdit_2)
        self.dateEdit = QtWidgets.QDateEdit(self.layoutWidget)
        self.dateEdit.setDateTime(QtCore.QDateTime(QtCore.QDate(2017, 6, 15), QtCore.QTime(0, 0, 0)))
        self.dateEdit.setObjectName("dateEdit")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.dateEdit)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_2)
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox.setMaximum(3)
        self.spinBox.setObjectName("spinBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.spinBox)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_3)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.layoutWidget)
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.verticalLayout_2.addWidget(self.plainTextEdit_2)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.textBrowser = QtWidgets.QTextBrowser(self.layoutWidget)
        self.textBrowser.setEnabled(True)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_2.addWidget(self.textBrowser)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.lcdNumber_5 = QtWidgets.QLCDNumber(self.layoutWidget)
        self.lcdNumber_5.setProperty("intValue", 0)
        self.lcdNumber_5.setObjectName("lcdNumber_5")
        self.horizontalLayout_2.addWidget(self.lcdNumber_5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.layoutWidget1 = QtWidgets.QWidget(self.tab)
        self.layoutWidget1.setGeometry(QtCore.QRect(0, 0, 481, 451))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setContentsMargins(1, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.layoutWidget1)
        self.textBrowser_2.setEnabled(True)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.verticalLayout_3.addWidget(self.textBrowser_2)
        self.lcdNumber_4 = QtWidgets.QLCDNumber(self.layoutWidget1)
        self.lcdNumber_4.setProperty("intValue", 0)
        self.lcdNumber_4.setObjectName("lcdNumber_4")
        self.verticalLayout_3.addWidget(self.lcdNumber_4)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser_3.setGeometry(QtCore.QRect(0, 0, 411, 431))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.lcdNumber = QtWidgets.QLCDNumber(self.tab_2)
        self.lcdNumber.setEnabled(True)
        self.lcdNumber.setGeometry(QtCore.QRect(414, 14, 421, 31))
        self.lcdNumber.setSmallDecimalPoint(False)
        self.lcdNumber.setProperty("intValue", 0)
        self.lcdNumber.setObjectName("lcdNumber")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_4.setGeometry(QtCore.QRect(0, 0, 411, 431))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.lcdNumber_2 = QtWidgets.QLCDNumber(self.tab_3)
        self.lcdNumber_2.setEnabled(True)
        self.lcdNumber_2.setGeometry(QtCore.QRect(414, 14, 421, 31))
        self.lcdNumber_2.setProperty("intValue", 0)
        self.lcdNumber_2.setObjectName("lcdNumber_2")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.tab_4)
        self.textBrowser_5.setGeometry(QtCore.QRect(0, 0, 411, 431))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.lcdNumber_3 = QtWidgets.QLCDNumber(self.tab_4)
        self.lcdNumber_3.setEnabled(True)
        self.lcdNumber_3.setGeometry(QtCore.QRect(414, 14, 421, 31))
        self.lcdNumber_3.setProperty("intValue", 0)
        self.lcdNumber_3.setObjectName("lcdNumber_3")
        self.tabWidget.addTab(self.tab_4, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 911, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.load_tweet)



        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox.setText(_translate("MainWindow", "Анализировать комментарии"))
        self.label_2.setText(_translate("MainWindow", "Количество комментариев"))
        self.label.setText(_translate("MainWindow", "верхняя граница даты"))
        self.label_3.setText(_translate("MainWindow", "нижняя граница даты"))
        self.label_4.setText(_translate("MainWindow", "Id на пост"))
        self.plainTextEdit_2.setPlainText(_translate("MainWindow", ""))
        self.label_5.setText(_translate("MainWindow", "Список первых трех комментариев выбранных по дате"))
        self.textBrowser.setHtml(_translate("MainWindow", ""))
        self.label_7.setText(_translate("MainWindow", "Средняя тональность всех комментариев "))
        self.textBrowser_2.setHtml(_translate("MainWindow", ""))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Пост"))
        self.textBrowser_3.setHtml(_translate("MainWindow", ""))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Первый комментарий"))
        self.textBrowser_4.setHtml(_translate("MainWindow", ""))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Второй комментарий"))
        self.textBrowser_5.setHtml(_translate("MainWindow", ""))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Третий комментарий"))
        self.label_6.setText(_translate("MainWindow", "Эмоциональная тональность от 0 - абсолютный негатив, до 100 - абсолютный позитив"))
        self.pushButton.setText(_translate("MainWindow", "Анализ"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

