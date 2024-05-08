import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import matplotlib.dates as mdates
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm 
import csv
import warnings
warnings.filterwarnings('ignore')

def my_fft(x):
    """
    计算信号的傅里叶变化，返回其能量谱
    :param x: 信号
    :return: 能量谱
    """
    half_ppv = (np.max(x) - np.min(x))/2
    fourier_transform = np.fft.rfft(x)
    power_specturm =np.abs(fourier_transform)/len(x)/half_ppv  # 注意要除以信号长度以及幅值
    id = np.argmax(power_specturm)
    print(f"最大频谱能量位置:{id}, 最大频谱能量为：{power_specturm[id]}")
    plt.plot(power_specturm)
    plt.show()
    return power_specturm

def wave_coeff(x) -> float:
    return 2*np.max(my_fft(x))



def plot_ACF(en_type, load_data):
    # 绘制自相关函数（ACF）图表
    plot_acf(load_data)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('Autocorrelation Function (ACF)')
    plt.savefig(f'./new_energy/{en_type}_ACF.png')

    # 绘制偏自相关函数（PACF）图表
    plot_pacf(load_data)
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f'./new_energy/{en_type}_PACF.png')
    




# 移动平均图
def draw_trend(id, day, timeseries, size):
    fig, ax = plt.subplots()
    # f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()
 
    ax.plot(day, timeseries, color='blue', label='Original')
    ax.plot(day, rol_mean, color='red', label='Rolling Mean')
    ax.plot(day, rol_std, color='black', label='Rolling standard deviation')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig(f'./new_energy/{id}.png')
    # plt.show()

 
#Dickey-Fuller test:
def teststationarity(ts,max_lag = None):
    dftest = statsmodels.tsa.stattools.adfuller(ts,maxlag= max_lag)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput





# ####读数据
# data = pd.read_csv('./dataset/new.csv', parse_dates=["date"])

# data = data.loc[(data["date"].dt.date >= date(2020, 2, 5)) & (data["date"].dt.date <= date(2020, 3, 5))]

en_type = "solar"
power = "POWER"

df = pd.read_csv(f'./dataset/{en_type}_solution.csv', parse_dates=["TIMESTAMP"])
df = df.fillna(0)

all_data = pd.DataFrame()
all_data["TIMESTAMP"] = df[df["ZONEID"] == 1]["TIMESTAMP"]

zone_id = df["ZONEID"].unique()
for id in zone_id:
    print("ZONEID:", id)
    if id == 1:
        all_data[id] = df[df["ZONEID"] == id][power]
    else:
        temp_data = df[df["ZONEID"] == id][["TIMESTAMP", power]]
        temp_data = temp_data.rename(columns={power: id})
        temp_data = temp_data.fillna(id)  # 使用ZONEID作为填充值
        all_data = all_data.merge(temp_data, on="TIMESTAMP", how="left")

# 将所有ID列求和为一列
all_data["Sum"] = all_data.iloc[:, 1:].sum(axis=1)

# print(all_data)



# ###季节性分析
# rd = sm.tsa.seasonal_decompose(data['TARGETVAR'].values[::24], freq=30)
# resplot = rd.plot()
# plt.show()

# ###傅里叶波动
# wc = wave_coeff(data['load'])
# print(f"wave coeff: {wc:.2f}")

# ###查看原始数据的均值和方差
# draw_trend("sum", all_data['TIMESTAMP'], all_data['Sum'], 12)

# p_value = teststationarity(all_data['Sum'])["p-value"]
# print(p_value)



'''
# #####移动平均和p_value
fieldnames = ['id', 'p_value']
with open('./solar_p_value.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # 写入CSV文件的列名

    for column, values in all_data.iloc[:, 1:].iteritems():
        print("values", len(values))
        draw_trend(column, all_data['TIMESTAMP'], values, 12)

        p_value = teststationarity(values)["p-value"]
        temp_row = {'id':column, 'p_value': p_value}
        writer.writerow(temp_row)
'''

plot_ACF(en_type, all_data['Sum'])


'''
# ###可视化
fig, ax = plt.subplots()
ax.plot(all_data['TIMESTAMP'], all_data['Sum'])
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=10))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()
plt.xlabel("day")
plt.ylabel("POWER")
# plt.savefig('E:\\报告\\figure\\新能源图\\wind_3.png')
plt.show()
'''