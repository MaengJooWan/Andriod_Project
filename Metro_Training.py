import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
import time
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from socketIO_client_nexus import SocketIO, LoggingNamespace
import requests

global name, state

result = 0
#칼럼들
now = datetime.now()
days = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
time_list = ['05~06','06~07','07~08','08~09','09~10','10~11','11~12','12~13','13~14','14~15','15~16','16~17','17~18','18~19','19~20','20~21','21~22','22~23','23~24','24~05']
train_dict = {}

#프로그램 동작 경과시간 체크하기 위해 사용
start_time = time.time()

#요일
days_index = time.localtime().tm_wday
days_result = days[days_index]

#월
month = now.month
#시간
times = now.hour

train = []

def time_value(times):
    if times >=5:
        value = times - 5
    else:
        value = 19

    return value


time_index = int(time_value(times))
time_result = time_list[time_index]


def socket_result():
    # Listen
    global state
    socketIO = SocketIO('210.119.107.159', 9928, LoggingNamespace)

    state = ""

    print('시작!')

    while True:
        print("값을 받는 중...")
        socketIO.emit('chat', "hello")
        socketIO.on('train', on_response)
        socketIO.wait(2)
        print("받은 값은 ", state, "입니다.")

        if state != "":
            params = training()
            URL = 'http://210.119.107.159:9928/set_page'
            requests.post(URL, params).text
            print("전송 성공")
            state = ""


def on_response(*args):
    global name, state
    temp = list(args)
    name,state = str(temp[0]).split(",")
    #print(state)


def create_dataset(signal_data, look_back=1):
    x_arr, y_arr = [], []
    for i in range(len(signal_data) - look_back):
        x_arr.append(signal_data[i:(i + look_back), 0])
        y_arr.append(signal_data[i + look_back, 0])

    x_arr = np.array(x_arr)
    x_arr = np.reshape(x_arr, (x_arr.shape[0], x_arr.shape[1], 1))

    return x_arr, np.array(y_arr)


def training():
    # 전철역 데이터 가지고 오기
    with open("E:\\학교\\4학년\\1학기\\안드로이드\\Dataset\\train_name.csv", "r") as f:
        s = f.read() + '\n'  # 데이터 읽을 시 문자로 읽게 설정 안하면 리스트 형식으로 읽어옴.... 이것때매 몇 시간 삽질한겨 -_-
        train = s.split('\n')

    data = pd.read_csv('E:\\학교\\4학년\\1학기\\안드로이드\\Dataset\\All_Data.csv', encoding='CP949')

    print("역 :", state)
    print("월 : ", month)
    print("요일 : ", days_result)
    data_value = (data['역명'] == state) & (data['월'] == month) & (data['요일'] == days_result)
    data_result = data[data_value]

    print(data_result)

    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 행렬 다시 설정
    time_value = str(time_result)
    temp = data_result[time_value].values.reshape(-1, 1)
    trade_count = scaler.fit_transform(temp)

    # 데이터 분리
    # 훈련
    train = trade_count[0:int(len(trade_count) * 0.5)]
    # 검증
    val = trade_count[int(len(trade_count) * 0.5): int(len(trade_count) * 0.75)]
    # 시험
    test = trade_count[int(len(trade_count) * 0.75): -1]

    x_train, y_train = create_dataset(train, 1)
    x_val, y_val = create_dataset(val, 1)
    x_test, y_test = create_dataset(test, 1)

    # #학습 모델 구성
    batch_size = 22
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))

    model.add(LSTM(64, return_sequences=False))

    # output
    model.add(Dense(1, activation='linear'))

    # 손실
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=10)
    predictions = model.predict(x_test, batch_size)
    real_predictions = scaler.inverse_transform(predictions)  # 0~1의 값으로 정규화된 값을 원래의 크기로 되돌린다.
    print(state, "의", month, "월", days_result, "의", time_result, "시간의 전철 복잡도는 : ", real_predictions[-1],
          "입니다.")  # 예측한 건수를 출력한다.
    result = real_predictions[-1]

    params = {'user_name': name, "train_names": state, "train_result": result}

    return params



#소켓통신
socket_result()

