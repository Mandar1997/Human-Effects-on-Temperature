# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 19:28:24 2020

@author: Mandar
"""

#import all the modules
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

#load the dataset
df=pd.read_excel("E:\\Python\\ChirantanPythonSpyder\\MidTerm1\\AirQualityUCI.xlsx")
df.describe()
df.info()

#replace -200 with nan values
data=df.replace(-200,np.nan)
data_decr=data.describe()
data.mean()
data.info()

#sns.heatmap(data.isnull(),square=True,annot=True,linewidths=4,linecolor='k')

#replace nan values with the mean of the column
data=data.fillna(data.mean())
#df['CO(GT)']=df['CO(GT)'].replace(0,data['CO(GT)'].mean())
#data['CO(GT)']=data['CO(GT)'].fillna(0)

#convert date and time column into string
data['Date']=data['Date'].astype(str)
data['Time']=data['Time'].astype(str)

#change the column names
data.columns= data.columns.str.replace('[(]','_')
data.columns= data.columns.str.replace('[)]','')
data.columns= data.columns.str.replace('.','_')

#combine date and time column
#data['Hour']=data['Time'].apply(lambda x: int(x.split(':')[0]))
data["Date_Time"] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
data.info()

#split date and time
data['month']= pd.to_datetime(data['Date_Time']).dt.month
#data['day'] = pd.to_datetime(data['Date_Time']).dt.dayofweek
#data['day_number'] = pd.to_datetime(data['Date_Time']).dt.day
data['hour'] = pd.to_datetime(data['Date_Time']).dt.hour
data['year'] = pd.to_datetime(data['Date_Time']).dt.year
#data['m_y']=pd.to_datetime(data['Date_Time']).dt.to_period('month')

#drop original date and time from the data
data=data.drop(['Date','Time',"Date_Time"],axis=1)

data.shape


#correlation matrix
data_corr=data.corr()
plt.figure(figsize = (12,10))
sns.heatmap(data_corr,square=True,annot=True,linewidths=4,linecolor='k')

#pieplot
March=sum(data.RH[data.month==3])/12
April=sum(data.RH[data.month==4])/12
May=sum(data.RH[data.month==5])/12
June=sum(data.RH[data.month==6])/12
July=sum(data.RH[data.month==7])/12
August=sum(data.RH[data.month==8])/12
September=sum(data.RH[data.month==9])/12
October=sum(data.RH[data.month==10])/12
November=sum(data.RH[data.month==11])/12
December=sum(data.RH[data.month==12])/12
January=sum(data.RH[data.month==1])/12
February=sum(data.RH[data.month==2])/12
data2 = pd.DataFrame({'RH':[5288.256168910248,3296.2832291189275,2706.1641416788425,2420.5795578611205,2050.7436686175224,2690.1425535256135,2647.203400165734,3837.3520040198005,3556.3333319756716,3527.6893474698027,3469.698526784486,2898.486892975487]},
                  index=['March','April','May','June','July','August','September','October','November','December','January','February'])
plot = data2.plot.pie(y='RH',autopct='%1.1f%%',figsize=(7,7))

#mean values
data.groupby(['month'])['T'].mean()

#bargraph
data.groupby(['month'])['T'].mean().plot(kind='bar')
plt.ylabel('temperature')

#linegraph
fig, ax = plt.subplots(figsize=(10,7))
data.groupby(['month'])['T'].mean().plot(ax=ax)
plt.show(fig,ax)

#histogram
plt.figure(figsize=(10,7))
plt.hist(data['T'],color='orange', bins=20)
plt.show()

#distplot
plt.figure(figsize = (12,6))
sns.distplot(data['T'],kde=True,bins=20)
plt.xlabel("T")
plt.title("Destribution of frequency")
plt.grid(linestyle='-.',linewidth = .5)

#violin plot 
plt.figure(figsize = (12,6))
sns.violinplot(data['T'],data=data)
plt.xlabel("T")
#plt.xlim(10,40)
plt.grid(linestyle='-.',linewidth = .5)

#Violin plots
plt.figure(figsize=(10,7))
sns.violinplot(x='month',y='T',data=data)
plt.title("Monthly analysis of temperature")
plt.xlabel("month")
plt.ylabel("T")
plt.show()

#boxplot 
plt.figure(figsize = (12,6))
sns.boxplot(data['T'],data=data)
plt.xlabel("T")
#plt.xlim(10,40)
plt.grid(linestyle='-.',linewidth = .5)

#removing outliers
max_t=data['T'].quantile(0.99)
a=data[data['T']>max_t]
min_t=data['T'].quantile(0.01)
b=data[data['T']<min_t]
data=data[(data['T']<max_t) & (data['T']>min_t)]

#boxplot for monthly analysis
plt.figure(figsize=(12,7))
sns.boxplot(x='month', y = 'T', data = data)
plt.xlabel('Month')
plt.ylabel('T')
plt.title("Monthly analysis of RH")
plt.grid(linestyle='-.',linewidth = .2)

#pairplot
sns.pairplot(data);

#Line plots to check the relation
#col_=data.columns.tolist()
for i in data.columns.tolist():
    sns.lmplot(x=i,y='T',data=data,markers='.')

sns.lmplot(x='T',y='RH',data=data,markers='.')

#statsmodel to find intercept and coefficient Hypothesis testing
model_1 = smf.ols('RH ~ CO_GT+PT08_S1_CO+PT08_S2_NMHC+PT08_S3_NOx+PT08_S4_NO2+PT08_S5_O3+NMHC_GT+C6H6_GT+NOx_GT+NO2_GT+T+AH+month', data=data)
#model_1 = smf.ols('T ~ CO_GT+PT08_S1_CO+PT08_S2_NMHC+PT08_S3_NOx+PT08_S4_NO2+PT08_S5_O3+NMHC_GT+C6H6_GT+NOx_GT+NO2_GT+RH+AH+month', data=data)
result=model_1.fit()
result.summary()
#result.params
#est = smf.ols('RH ~ CO_GT+PT08_S1_CO+NMHC_GT+C6H6_GT+PT08_S2_NMHC+NOx_GT+PT08_S3_NOx+NO2_GT+PT08_S4_NO2+PT08_S5_O3+T+AH+month+day+day_number+hour', data=data).fit()
#est.summary().tables[1]
    
#create x and y variable
#x is independent or predictor variable
#y is the dependant or response variable
#y is dependant on x
X=data.drop(['T'],axis=1)
Y=data['T'].values.reshape(-1,1)

#split train and test data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

#perform linear regression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

#find r2 score and mean square error
r2score=r2_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)

#calculate max, mean,min,var of actual and predicted data
yTestMax=y_test.max()
yTestMin=y_test.min()
yTestMean=y_test.mean()
yTestVar=y_test.var()

yPredMax=y_pred.max()
yPredMin=y_pred.min()
yPredMean=y_pred.mean()
yPredVar=y_pred.var()

#using regplot
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

#######################################################################################

#plotting 
T = np.arange(-2,50)
AH = np.arange(0,3)
B1, B2 = np.meshgrid(T, AH, indexing='xy')
Z = np.zeros((AH.size, T.size))

fig = plt.figure(figsize=(10,6))
fig.suptitle('Regression: T ~ AH + RH', fontsize=20)
ax = axes3d.Axes3D(fig)
ax.plot_surface(B1, B2, Z, rstride=10, cstride=5, alpha=0.4)
ax.scatter3D(data['RH'].values, data['AH'].values, data['T'].values, c='r')
ax.set_xlabel('RH')
ax.set_xlim(-2,50)
ax.set_ylabel('AH')
ax.set_ylim(ymin=0)
ax.set_zlabel('T');

#scatter plot
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="red")
plt.show()

#meshgrid
x_mesh, y_mesh = np.meshgrid(np.linspace(x_test.min(), x_test.max(), 100),np.linspace(y_test.min(), y_test.max(), 100))
onlyX = pd.DataFrame({'X': x_mesh.ravel(), 'Y': y_mesh.ravel()})
fittedY=result.predict(exog=onlyX)

#x_mesh,y_mesh=np.meshgrid(x_test,y_test)

fittedY=np.array(fittedY)

#eclipse plotting
ellipse=x_mesh**2.0+4.0*y_mesh**2.0
plt.contour(x_mesh,y_mesh,cmap='jet')
plt.colorbar()
plt.show()