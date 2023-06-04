import pandas as pd

#创建数据
data={
    'Year':[1995,1996,1997,1998,1999,2000,2001,2002,2003,2004],
    'Pollution':[174.0,179.0,183.0,189.0,207.0,234.0,220.5,256.0,270.0,285.0]
}

#创建DataFrame
df=pd.DataFrame(data)

#保存为xlsx文件
df.to_excel('对长江水质污染的预测.xlsx',index=False)