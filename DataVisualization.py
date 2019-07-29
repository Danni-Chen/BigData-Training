#数据可视化
import numpy as np
import matplotlib.pyplot as plt
import os

#综合案例：使用外部数据，在一个画布上绘制四个图
os.chdir('D://Python')#os.chdir() 方法用于改变当前工作目录到指定的路径。此处指定当前工作目录到D盘的Python文件夹，因为数据文件存在该文件夹内，根据文件的不同位置进行适当修改。
plt.rcParams['font.sans-serif']='SimHei'#用于处理中文乱码，此处用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False#用于处理中文乱码，此处用来正常显示负号
data = np.load('国民经济核算季度数据.npz')#读取国民经济核算季度数据.npz文件，并将文件内的数据存储到data中
name = data['columns'] #提取其中的columns数组作为数据的标签
values = data['values'] #提取其中的values数组作为数据的存在位置
p = plt.figure(figsize=(12,12))#新建一个图例，其中figsize以英寸为单位的宽高，也就是新建一个12英寸*12英寸的图例
plt.title('2000-2017年季度生产总值散点图')#对刚刚新建的图例进行命名；技巧：在创建子图之前先写标题，可以让标题呈现在最上面
#以下开始绘画第一个子图
ax1 = p.add_subplot(2,2,1)#将画布分割成2行2列，图像画在从左到右从上到下的第1块
#以下表示在第一个子图中一共绘画了3个散点图
plt.scatter(values[:,0],values[:,3],marker='o',c='red')#以数据的第0列做x轴，第4列做y轴,来描绘散点图；其中o表示数据以小圆点形状呈现，red表示小圆点为红色
plt.scatter(values[:,0],values[:,4],marker='D',c='green')#以数据的第0列做x轴，第5列做y轴,来描绘散点图；其中D表示数据以钻石形状呈现，green表示砖石为绿色
plt.scatter(values[:,0],values[:,5],marker='v',c='blue')#以数据的第0列做x轴，第6列做y轴,来描绘散点图；其中v表示数据以下三角形状呈现，blue表示下三角为蓝色
plt.xlabel('年份')#设置横坐标名称
plt.ylabel('生产总值')#设置纵轴标名称
plt.legend(['第一产业','第二产业','第三产业'])#设置图例

#以下开始绘画第二个子图
ax2 = p.add_subplot(2,2,2)#将画布分割成2行2列，图像画在从左到右从上到下的第2块
#以下表示在第二个子图中一共绘画了9个散点图
plt.scatter(values[:,0],values[:,6],marker='o',c='r')#同上，绘画散点图；其中o表示数据以小圆点的形状呈现，r表示小圆点为红色
plt.scatter(values[:,0],values[:,7],marker='D',c='b')#同上，绘画散点图；其中D表示数据以砖石的形状呈现，b表示砖石为蓝色
plt.scatter(values[:,0],values[:,8],marker='v',c='y')#同上，绘画散点图；其中v表示数据以下三角的形状呈现，y表示下三角为黄色
plt.scatter(values[:,0],values[:,9],marker='8',c='g')#同上，绘画散点图；其中8表示数据以八角形的形状呈现，g表示八角形为绿色
plt.scatter(values[:,0],values[:,10],marker='p',c='c')#同上，绘画散点图；其中p表示数据以五角星的形状呈现，c表示五角星为青色
plt.scatter(values[:,0],values[:,11],marker='+',c='m')#同上，绘画散点图；其中+表示数据以加号的形状呈现，m表示加号为品红色
plt.scatter(values[:,0],values[:,12],marker='s',c='k')#同上，绘画散点图；其中s表示数据以正方形的形状呈现，k表示正方形为黑色
plt.scatter(values[:,0],values[:,13],marker='+',c='purple')#同上，绘画散点图；其中+表示数据以加号的形状呈现，purple表示加号为品紫色
plt.scatter(values[:,0],values[:,14],marker='s',c='brown')#同上，绘画散点图；其中s表示数据以正方形的形状呈现，brown表示正方形为棕色
plt.xlabel('年份')#设置横坐标名称
plt.ylabel('生产总值')#设置纵轴标名称
plt.legend(['农业','工业','建筑','批发','交通','餐饮','金融','房地产','其他'])#设置图例
#plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)第一个参数接受坐标，第二个参数接受各坐标显示的文本；这里表示x轴的刻度为0~70，并步数为4；其中rotation表示文本显示时旋转的角度，以防文字过多造成的重叠

#以下开始绘画第三个子图
ax3 = p.add_subplot(2,2,3)#将画布分割成2行2列，图像画在从左到右从上到下的第3块
plt.plot(values[:,0],values[:,3],'bs-',values[:,0],values[:,4],'ro-',values[:,0],values[:,5],'gH--')#这一句话等于以下三句话
#plt.plot(values[:,0],values[:,3],'bs-')#以数据的第0列做x轴，第4列做y轴,来描绘折线图；其中第三个参数：b表示蓝色、s-表示线条由一个正方形和实线组合而成
#plt.plot(values[:,0],values[:,4],'ro-.')#以数据的第0列做x轴，第5列做y轴,来描绘折线图；其中第三个参数：r表示红色、o-表示线条由一个圆点和实线组合而成
#plt.plot(values[:,0],values[:,5],'gH--')#以数据的第0列做x轴，第6列做y轴,来描绘折线图；其中第三个参数：g表示绿色、h--表示线条由一个横六边形和虚线组合
plt.xlabel('年份')#设置横坐标名称
plt.ylabel('生产总值')#设置纵轴标名称
plt.legend(['第一产业','第二产业','第三产业'])#设置图例

#以下开始绘画第四个子图
ax4 = p.add_subplot(2,2,4)#将画布分割成2行2列，图像画在从左到右从上到下的第4块
plt.plot(values[:,0],values[:,6],'r-',values[:,0],values[:,7],'b-.',values[:,0],values[:,8],'y--',values[:,0],values[:,9],'g.',values[:,0],values[:,10],'c-',values[:,0],values[:,11],'m-.',values[:,0],values[:,12],'k--',values[:,0],values[:,13],'r.',values[:,0],values[:,14],'b-')
plt.xlabel('年份')#设置横坐标名称
plt.ylabel('生产总值')#设置纵轴标名称
plt.legend(['农业','工业','建筑','批发','交通','餐饮','金融','房地产','其他'])#设置图例

plt.show()#显示图例。如果需要的话，在show之前也可以先保存，使用plt.savefig()进行保存图片


'''
#案例一：绘画y=x^2和y=x^4
data = np.arange(0,1,0.01)
plt.title('lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((0,1))
plt.ylim((0,1))
plt.xticks([0.0,0.2,0.4,0.6,0.8,1])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1])
plt.plot(data,data**2)
plt.plot(data,data**4)
plt.legend(['y=x^2','y=x^4'])
plt.savefig('y=x^2.png')
plt.show()
'''


'''
#案例二：在一个画布上绘画两幅图
rad = np.arange(0,np.pi*2,0.01)
p1 = plt.figure(figsize=(8,6),dpi=80)
ax1 = p1.add_subplot(2,1,1)#第一幅子图:绘画y=x^2和y=x^4
plt.title('lines')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((0,1))
plt.ylim((0,1))
plt.xticks([0.0,0.2,0.4,0.6,0.8,1])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1])
plt.plot(rad,rad**2)
plt.plot(rad,rad**4)
plt.legend(['y=x^2','y=x^4'])
#plt.savefig('y=x^2.png')
#plt.show()
ax2 = p1.add_subplot(2,1,2)#第二幅子图:绘画sincos曲线
plt.title('sin/cos')
plt.xlabel('rad')
plt.ylabel('value')
plt.xlim((0,np.pi*2))
plt.ylim((-1,1))
plt.xticks([0.0,np.pi/2,np.pi,np.pi*1.5,np.pi*2])
plt.yticks([-1,-0.5,0,0.5,1])
plt.plot(rad,np.sin(rad))
plt.plot(rad,np.cos(rad))
plt.legend(['sin','cos'])
plt.savefig('sincos.png')
plt.show()
'''


'''
#案例三：使用$数学公式$直接绘制sin曲线
x = np.linspace(0,4*np.pi)
y = np.sin(x)
plt.rcParams['lines.linestyle']='-.'
plt.rcParams['lines.linewidth']=3
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.plot(x,y,label="$sin(x)$")
plt.title('sin曲线')
plt.show()
'''


'''
#案例四：使用外部数据绘制图

plt.rcParams['font.sans-serif']='SimHei'#用于处理中文乱码
plt.rcParams['axes.unicode_minus']=False#用于处理中文乱码
data = np.load('国民经济核算季度数据.npz')#导入数据
name = data['columns']#提取其中的columns数组作为数据的标签
values = data['values']#提取其中的values数据作为数据的存在位置
plt.figure(figsize=(8,7))

#利用上面的数据，绘制散点图
plt.scatter(values[:,0],values[:,3],marker='o',c='red')
plt.scatter(values[:,0],values[:,4],marker='D',c='green')
plt.scatter(values[:,0],values[:,5],marker='v',c='blue')
plt.xlabel('年份')
plt.ylabel('生产总值')
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.legend(['第一产业','第二产业','第三产业'])
plt.title('2000-2017年季度生产总值散点图')
plt.savefig('2000-2017年季度生产总值散点图.png')
plt.show()

#同样的案例数据，绘制折线图
#plt.plot(values[:,0],values[:,2],color ='r',linestyle='--',marker='o')
plt.plot(values[:,0],values[:,3],'bs-',values[:,0],values[:,4],'ro-',values[:,0],values[:,5],'gH--')
plt.xlabel('年份')
plt.ylabel('生产总值')
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
#plt.legend(['第一产业','第二产业','第三产业'])
plt.title('2000-2017年季度生产总值折线图')
plt.savefig('2000-2017年季度生产总值折线图.png')
plt.show()

#同样的案例数据，绘画直方图
plt.bar(range(3),values[-1,3:6],width=0.5)#此处values的-1代表数据的最后一行数据，3:6代表第3到5列的数据
plt.xlabel('年份')#画X轴标签
plt.ylabel('生产总值')#画Y轴标签
plt.legend(['第一产业','第二产业','第三产业'])#画图例
label = ['第一产业','第二产业','第三产业']
plt.xticks(range(3),label)
plt.title('2000-2017年季度生产总值直方图.png')
plt.show()

#同样的案例数据，绘画圆饼图
label = ['第一产业','第二产业','第三产业']
explode = [0.01,0.01,0.01]
plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')
plt.title('2000-2017年季度生产总值饼形图.png')
plt.show()

#同样的案例数据，绘画箱线图
label = ['第一产业','第二产业','第三产业']
gdp = (list(values[:3]),list(values[:,4]),list(values[:,5]))
plt.boxplot(gdp,notch=True,labels=label,meanline=True)
plt.title('2000-2017年季度生产总值箱线图.png')
plt.show()
'''