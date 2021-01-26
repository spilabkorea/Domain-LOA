# Log Anomaly Detection

## Introduction

Log Anomaly Detection은 Anomaly Detection 의 한 분야로 서버의 기록된 log 데이터에서 특이 상황을 감지한다. 실시간으로 수많은 기록이 지나가는 로그 특정 상, 수 많은 기록 중 확인이 필요한 범위를 축소해주는 Log Anomaly Detection은 서버 관리자의 시간과 노력을 아낄수 있을 것으로 기대 된다.

## Running on Your Own Machine

Keras==2.4.3 , tensorflow==2.4.0 기반

```
$ pip install -r requirements.txt
```

### 01 Autoencoder for HTTP

HTTP의 로그 기록 데이터를 이용하여 비지도 학습인 Autoencoder를 적용.

### 02 Autoencoder for HDFS

기존의 HTTP에서 진행된 방식에서 HDFS의 로그 기록 데이터로 변경하여 비지도 학습인 Autoencoder를 적용.

### 03 Autoencoder for syslog

실제 데이터인 syslog 데이터에 autoencoder를 적용하여 결과 확인.

#### NN_Autoencoder_Architecture

![NN_Autoencoder_Architecture](./data/model_plot.png)

#### LSTM_Autoencoder_Architecture
![LSTM_Autoencoder_Architecture](./data/LSTM_model_plot.png)


### 04 LDA for Dataset

토픽 모델링 방법인 LDA 를 로그에 적용하였을때 나오는 결과를 통해 데이터에 대한 이해.

### 05 TF-IDF for Dataset

정보 검색과 텍스트 마이닝에서 이용하는 가중치인 TF-IDF 를 로그에 적용하였을때 나오는 결과를 통해 데이터에 대한 이해.

### 06 Autoencoder for HDFS time

blk를 기준으로 라벨링된 HDFS를 1분 단위로 변경한 데이터에 NN, LSTM Autoencoder를 적용.

### 07 Autoencoder for Syslog time

PID를 기준으로 처리했던 Syslog 데이터를 1분 단위로 변경한 데이터에 NN, LSTM Autoencoder를 적용.

### 08 iso auto for syslog

분단위 Syslog 데이터에 Isolation Forest 를 적용하여 예상 normal data를 생성 해당 데이터에 Autoencoder 를 적용.

## Reference

* Autoencoder : https://github.com/zpettry/AI-Autoencoder-for-HTTP-Log-Anomaly-Detection
* HTTP Data : https://www.kaggle.com/shawon10/web-log-dataset#webLog.csv
* HDFS Data & Lenma preprocess : https://github.com/logpai/loglizer 
* Deeplog : https://github.com/nailo2c/deeplog