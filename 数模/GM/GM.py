import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

#更换字体
plt.rcParams["font.sans-serif"] = "Source Han Serif CN"  # 使用中文字体（如：宋体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 导入数据
df=pd.read_excel("对长江水质污染的预测.xlsx",index_col=[0])#将该xlsx文件中的数据导入
df.plot(figsize=(12,6),marker="o",fontsize=16,legend=None)
plt.xlabel("年份",fontsize=20)
plt.ylabel("排污总量",fontsize=20)
plt.savefig("/home/zhuheqin/clone/python/数模/GM/对长江水质污染的预测.png")

df

#检验数据是否适合用GM模型
def level_check(data):
    n=len(data)
    lambda_k=np.zeros(n-1)
    for i in range(n-1):
        lambda_k[i]=data.iloc[i]/data.iloc[i+1]
        if lambda_k[i]<np.exp(-2/(n+1)) or lambda_k[i]>np.exp(2/(n+2)):
            flag=False
    else:
        flag=True

    if not flag:
        print(
            "*"*10,
            "级比检验失败,请做平移变换后重试",
            "*"*10,
        )
    else:
        print(
            "*"*10,
            "级比检验成功,可以使用GM模型",
            "*"*10,
        )

level_check(df["Pollution"])


#分割训练集与测试集
test_num=3
train=df["Pollution"][:-test_num]
test=df["Pollution"][-test_num:]

#建立GM模型
from statsmodels.api import add_constant
from IPython.display import Latex

data=train
def GM11(data,forcast_len=5,show=False):
    X_0=data
    X_1=data.cumsum()
    Z_1=-X_1.rolling(2).mean().dropna()
    B=add_constant(Z_1,prepend=False)
    Y=X_0[1:]
    U=(np.linalg.inv(B.T @ B)) @ B.T @ Y
    a=U.iloc[0]
    u=U.iloc[1]
    #拟合值
    fitted_values=[(X_0.iloc[0]-u/a)*(1-np.exp(a))*np.exp(-a*t)for t in range(1,len(data)+1)]
    #预测值
    forcast=[(X_0.iloc[0]-u/a)*(1-np.exp(a))*np.exp(-a*t)for t in range(len(data),len(data)+forcast_len)]
    #相对残差
    relative_residuals=(abs(data-fitted_values)/data)
    #级比
    class_ratio=data[1:]/data[:-1]
    #级比偏差
    eta=abs(1-(1-0.5*a)/(1+0.5*a)*(1/class_ratio))
    if show:
        print("发展系数:",-a,"\n灰作用量:",u)
        print("预测值:",forcast)
        print("拟合值:\n",pd.DataFrame(fitted_values,index=data.index))
    return fitted_values,forcast,relative_residuals,eta,u,a

fitted_values,forcast,relative_residuals,eta,u,a=GM11(data,show=True)

plt.figure(figsize=(12,6),dpi=80)
plt.plot(df["Pollution"],label="排污总量真实值")
plt.plot(range(1995,2007),fitted_values+forcast,label="GM预测值",linestyle="--")
plt.legend()
Latex(
    f"拟合结果:$\hat x^{{(1)}}(t)={train.iloc[0]-round(u/a,4)}e^{{{-round(a,4)}(t-1)}}{round(u/a,4)}$"
)
test_num=3
train=df["Pollution"][:-test_num]
test=df["Pollution"][-test_num:]

#建立GM模型
from statsmodels.api import add_constant
from IPython.display import Latex

data=train
def GM11(data,forcast_len=5,show=False):
    X_0=data
    X_1=data.cumsum()
    Z_1=-X_1.rolling(2).mean().dropna()
    B=add_constant(Z_1,prepend=False)
    Y=X_0[1:]
    U=(np.linalg.inv(B.T @ B)) @ B.T @ Y
    a=U.iloc[0]
    u=U.iloc[1]
    #拟合值
    fitted_values=[(X_0.iloc[0]-u/a)*(1-np.exp(a))*np.exp(-a*t)for t in range(1,len(data)+1)]
    #预测值
    forcast=[(X_0.iloc[0]-u/a)*(1-np.exp(a))*np.exp(-a*t)for t in range(len(data),len(data)+forcast_len)]
    #相对残差
    relative_residuals=(abs(data-fitted_values)/data)
    #级比
    class_ratio=data[1:]/data[:-1]
    #级比偏差
    eta=abs(1-(1-0.5*a)/(1+0.5*a)*(1/class_ratio))
    if show:
        print("发展系数:",-a,"\n灰作用量:",u)
        print("预测值:",forcast)
        print("拟合值:\n",pd.DataFrame(fitted_values,index=data.index))
    return fitted_values,forcast,relative_residuals,eta,u,a

fitted_values,forcast,relative_residuals,eta,u,a=GM11(data,show=True)

plt.figure(figsize=(12,6),dpi=80)
plt.plot(df["Pollution"],label="排污总量真实值")
plt.plot(range(1995,2007),fitted_values+forcast,label="GM预测值",linestyle="--")
plt.legend()
Latex(
    f"拟合结果:$\hat x^{{(1)}}(t)={train.iloc[0]-round(u/a,4)}e^{{{-round(a,4)}(t-1)}}{round(u/a,4)}$"
)

plt.savefig("/home/zhuheqin/clone/python/数模/GM/拟合结果.png")

