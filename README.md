# Log Anomaly Detection

## Introduction

Log Anomaly Detection은 Anomaly Detection 의 한 분야로 서버의 기록된 log 데이터에서 특이 상황을 감지한다. 실시간으로 수많은 기록이 지나가는 로그 특정 상, 많은 기록의 범위를 축소해주는 Log Anomaly Detection은 서버 관리자의 시간과 노력을 아낄수 있을 것으로 기대 된다.

## Running on Your Own Machine

Keras==2.4.3 , tensorflow==2.4.0 기반

```
$ pip install -r requirements.txt
```

## Autoencoder for HTTP

HTTP의 로그 기록 데이터를 이용하여 비지도 학습인 Autoencoder를 적용.

## Autoencoder for HDFS

기존의 HTTP에서 진행된 방식에서 HDFS의 로그 기록 데이터로 변경하여 비지도 학습인 Autoencoder를 적용.

## Reference

* Autoencoder : https://github.com/zpettry/AI-Autoencoder-for-HTTP-Log-Anomaly-Detection
* HTTP Data : https://www.kaggle.com/shawon10/web-log-dataset#webLog.csv
* HDFS Data & preprocess : https://github.com/logpai/loglizer 