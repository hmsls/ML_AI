#coding: utf-8
"""
这是Matplotlib库的学习
"""
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline 这个是在notebook中不用show直接可以显示图像的语句
if __name__ == '__main__':
    #**********matplotlib概述*********-----------------------------------------------------
    #画一个简单的二维图
    # plt.plot([1,2,3,4,5],[1,4,9,16,25])
    #指定x和y轴的名字,字体的大小
    # plt.xlabel('xlabel',fontsize=10)
    # plt.ylabel('ylabel')
    # plt.show()

    #不同线条的画制，如虚线图等
    """
    '-'       solid line style
    '--'      dashed line style
    '-.'      dash-dot line style
    ':'       dotted line style
    '.'       point marker
    ','       pixel marker
    'o'       circle marker
    'v'       triangle_down marker
    '^'       triangle_up marker
    '&lt;'       triangle_left marker
    '&gt;'       triangle_right marker
    '1'       tri_down marker
    '2'       tri_up marker
    '3'       tri_left marker
    '4'       tri_right marker
    's'       square marker
    'p'       pentagon marker
    '*'       star marker
    'h'       hexagon1 marker
    'H'       hexagon2 marker
    '+'       plus marker
    'x'       x marker
    'D'       diamond marker
    'd'       thin_diamond marker
    '|'       vline marker
    '_'       hline marker
    """
    #图像的线条的种类和颜色
    # plt.plot([1,2,3,4,5],[1,4,9,16,25],'-',color='r')
    # plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], 'r-')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # **********matplotlib的子图和标注*********-----------------------------------------------------
    tang_numpy = np.arange(0,10,0.5)
    #一张图中画多条线
    # plt.plot(tang_numpy,tang_numpy,'r--',
    #          tang_numpy, tang_numpy ** 2, 'g:',
    #          tang_numpy, tang_numpy * 3, 'bo')
    # plt.show()

    x = np.linspace(-10,10)
    y = np.sin(x)
    #指定线条的粗细，颜色，线条类型，标记每个绘图的点，标记每个绘图的点的颜色，标记每个绘图的点的大小
    # plt.plot(x,y,linewidth=2,linestyle=':',marker='o',markerfacecolor='r',markersize=10)
    #先画图，再进行设计布局风格
    # line = plt.plot(x,y)
    # plt.setp(line,color='g',linewidth=3,alpha=0.5)#最后一个是设置透明度
    # plt.show()

    #画子图
    # plt.subplot(211)  #211表示：一会儿要花的图是2行1列的，最后一个1是子图中的第一个图
    # plt.plot(x,y,color='r')
    # plt.subplot(212) ##211表示：一会儿要花的图是2行1列的，最后一个2是子图中的第2个图
    # plt.plot(x,y,color='b')
    #画3行2列，第一幅和第4副图
    # plt.subplot(321)
    # plt.plot(x,y,color='r')
    # plt.subplot(324)
    # plt.plot(x,y,color='g')
    # plt.show()

    #给图加注释
    # plt.plot(x,y,color='b',linestyle=':',marker='o',markerfacecolor='r',markersize=10)
    # plt.xlabel('x:---')
    # plt.ylabel('y:---')
    # plt.title("li   shuai   :--")
    # #在（0,0）位置加文字
    # plt.text(0,0,"lishuai")
    #加格子
    # plt.grid(True)
    #加注释，参数是：注释内容，注释位置，注释文字的位置，箭头（箭头颜色，箭头收缩比例，箭头的宽度，箭头长度）
    # plt.annotate('zhushi',xy=(-5,0),xytext=(-2,0.3),arrowprops=dict(facecolor='black',shrink=0.05,headwidth=20,headlength=20))
    # plt.show()

    # **********matplotlib的风格设置*********-----------------------------------------------------
    #可选风格
    # print plt.style.available
    # plt.plot(x,y)
    #设置风格
    # plt.style.use('dark_background')
    # plt.show()

    # **********matplotlib的条形图*********-----------------------------------------------------
    np.random.seed(0)
    x = np.arange(5)
    y = np.random.randint(-5,5,5)
    #例一
    # fig,axes = plt.subplots(ncols=2)
    #绘制垂直方向的条形图
    # v_bars = axes[0].bar(x,y,color='red')
    #绘制水平方向的条形图
    # h_bars = axes[1].barh(x,y,color='red')
    #在0位置增加一条线，指定线的颜色和宽度
    # axes[0].axhline(0,color='grey',linewidth=2)
    # axes[1].axvline(0,color='grey',linewidth=3)
    # plt.show()
    #例二
    # fig,ax = plt.subplots()
    # v_bars = ax.bar(x,y,color='lightblue')
    # for bar,height in zip(v_bars,y):
    #     if height < 0:
    #         bar.set(edgecolor='darkred',color='green',linewidth = 3)
    # plt.show()
    #例三
    # x = np.random.randn(100).cumsum()
    # y = np.linspace(0,10,100)
    #
    # fig,ax = plt.subplots()
    # #对图形内部进行填充
    # ax.fill_between(x,y,color='lightblue')
    # plt.show()
    #例四
    # x =np.linspace(0,10,200)
    # y1 = 2 * x +1
    # y2 = 3 * x +1.2
    # y_mean = 0.5*x*np.cos(2*x) + 2.5*x +1.1
    # fig,ax = plt.subplots()
    # ax.fill_between(x,y1,y2,color='red')
    # ax.plot(x,y_mean,color='black')
    # plt.show()
    #例五
    #均值
    mean_values = [1,2,3]
    #误差
    variance = [0.2,0.4,0.5]
    #条形图标签
    # bar_label = ['bar1','bar2','bar3']
    # x_pos = list(range(len(bar_label)))
    # plt.bar(x_pos,mean_values,yerr = variance,alpha=0.3)
    # max_y = max(zip(mean_values,variance))
    # plt.ylim([0,(max_y[0] + max_y[1]*1.2)])
    # plt.ylabel('variable_y')
    # plt.xticks(x_pos,bar_label)
    # plt.show()
    #例六
    # x1 = np.array([1,2,3])
    # x2 = np.array([2,2,3])
    #
    # bar_labels = ['bar1','bar2','bar3']
    # fig = plt.figure(figsize=(8,6))
    # y_pos = np.arange(len(x1))
    # y_pos = [x for x in y_pos]
    # plt.barh(y_pos,x1,color='grey',alpha=0.5)
    # plt.barh(y_pos,-x2,color='b',alpha=0.5)
    # plt.xlim(-max(x2)-1,max(x1)+1)
    # plt.ylim(-1,len(x1)+1)
    # plt.show()
    #例7
    # green_data = [1,2,3]
    # blue_date = [3,2,1]
    # red_data = [2,3,3]
    # labels = ['group1','group2','group3']
    #
    # pos= list(range(len(green_data)))
    # width = 0.2
    # fig,ax = plt.subplots(figsize=(8,6))
    # plt.bar(pos,green_data,width,alpha=0.5,color='g',label=labels[0])
    # plt.bar([p+width for p in pos], green_data, width, alpha=0.5, color='b', label=labels[1])
    # plt.bar([p+width*2 for p in pos], green_data, width, alpha=0.5, color='r', label=labels[2])
    # plt.show()

    # **********matplotlib的条形图外观*********-----------------------------------------------------
    # data = range(200,225,5)
    # bar_labels = ['a','b','c','d','e']
    # fig = plt.figure(figsize=(10,8))
    # y_pos = np.arange(len(data))
    # plt.yticks(y_pos,bar_labels,fontsize=16)
    # bars = plt.barh(y_pos,data,alpha = 0.5,color='g')
    # plt.vlines(min(data),-1,len(data)+0.5,linestyles='dashed')
    # for b,d in zip(bars,data):
    #     plt.text(b.get_width() + b.get_width()*0.05,b.get_y() + b.get_height()/2,'{0:.2%}'.format(d/min(data)))
    # plt.show()

    # mean_values = range(10,18)
    # x_pos = range(len(mean_values))

    import matplotlib.colors as col
    import matplotlib.cm as cm

    # cmap1 = cm.ScalarMappable(col.Normalize(min(mean_values),max(mean_values)),cm.hot)
    # cmap2 = cm.ScalarMappable(col.Normalize(0,20,cm.hot))

    # plt.subplot(121)
    # plt.bar(x_pos,mean_values,color=cmap1.to_rgba(mean_values))
    # plt.subplot(122)
    # plt.bar(x_pos, mean_values, color=cmap2.to_rgba(mean_values))
    # plt.show()

    # patterns = ('-','+','x','\\','*','o','0','.')
    # fig = plt.gca()
    # mean_values = range(1,len(patterns)+1)
    # x_pos = list(range(len(mean_values)))
    #
    # bars = plt.bar(x_pos,mean_values)
    # for bar,pattern in zip(bars,patterns):
    #     bar.set_hatch(patterns)
    # plt.show()

    # **********matplotlib的盒图*********-----------------------------------------------------

    # tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
    # fig = plt.figure(figsize = (8,6))
    # plt.boxplot(tang_data,notch=False,sym = 's',vert=True)
    # plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
    # plt.xlabel('x')
    # plt.title('box plot')
    # plt.show()

    # **********matplotlib的盒图细节*********-----------------------------------------------------
    # tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
    # fig = plt.figure(figsize=(8,6))
    # bplot = plt.boxplot(tang_data,notch=False,sym='s',vert=False,patch_artist=True)
    # plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
    # plt.xlabel('x')
    # plt.title('box plot')
    # colors = ['pink','lightblue','lightgreen']
    # #设置线条的颜色
    # for components in bplot.keys():
    #     for line in bplot[components]:
    #         line.set_color('black')
    # #盒子的颜色填充
    # for patch,color in zip(bplot['boxes'],colors):
    #     patch.set_facecolor(color)
    # plt.show()

    #小提琴图violinplot
    # fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
    # tang_data = [np.random.normal(0,std,10) for std in range(6,10)]
    # axes[0].violinplot(tang_data,showmeans=False,showmedians=True)
    # axes[0].set_title('violin plot')
    # axes[1].boxplot(tang_data)
    # axes[1].set_title('box plot')
    #
    # for ax in axes:
    #     ax.yaxis.grid(True)
    #     ax.set_xticks([y+1 for y in range(len(tang_data))])
    # plt.setp(axes,xticks=[y+1 for y in range(len(tang_data))],xticklabels=['x1','x2','x3','x4'])
    # plt.show()

    # **********matplotlib的绘图细节设置*********-----------------------------------------------------
    # x = range(10)
    # y = range(10)
    # fig = plt.gca()
    # plt.plot(x,y)
    # #选择留下什么，去掉什么，去掉x和y轴的刻度显示
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # plt.show()

    import math
    # x = np.random.normal(loc=0,scale=1,size=300)
    # width = 0.5
    # bins = np.arange(math.floor(x.min())-width,math.ceil(x.max())+width,width)
    # ax = plt.subplot(111)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tick_params(bottom='off',top='off',left='off',right='off')
    # plt.grid()
    # plt.hist(x,alpha=0.5,bins=bins)
    # plt.show()

    # x = range(10)
    # y = range(10)
    # labels = ['lishuain' for n in range(10)]
    # fig,ax = plt.subplots()
    # plt.plot(x,y)
    # #设置x标签的名称，显示方式（斜45度显示），对齐方式
    # ax.set_xticklabels(labels,rotation=45,horizontalalignment='right')
    # plt.show()

    # **********matplotlib的绘图细节设置2*********-----------------------------------------------------
    # x = np.arange(10)
    # for i in range(1,4):
    #     plt.plot(x,i*x**2,label='group %d'%i)
    # #添加图示
    # plt.legend(loc='best')
    # plt.show()

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # x = np.arange(10)
    # for i in range(1,4):
    #     plt.plot(x,i*x**2,label='Group %d'%i,marker='o')
    # #loc表示位置在哪，bbox_to_anchor表示左右调整，ncol表示分成几列表示图示,framealpha表示透明程度
    # ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=3,framealpha=0.5)
    # plt.show()

    # **********matplotlib的直方图和散点图*********-----------------------------------------------------
    # data = np.random.normal(0,20,1000)
    # bins = np.arange(-100,100,5)
    # plt.hist(data,bins=bins)
    # plt.xlim([min(data)-5,max(data)+5])
    # plt.show()
    # import random
    # data1 = [random.gauss(15,10) for i in range(500)]
    # data2 = [random.gauss(5,5,) for i in range(500)]
    # bins = np.arange(-50,50,2.5)
    #
    # plt.hist(data1,bins=bins,label='class 1',alpha=0.3)
    # plt.hist(data2,bins=bins,label='class 2',alpha=0.3)
    # plt.legend(loc='best')
    # plt.show()

    #散点图
    # mu_vec1 = np.array([0,0])
    # cov_mat1 = np.array([[2,0],[0,2]])
    #
    # x1_sample = np.random.multivariate_normal(mu_vec1,cov_mat1,100)
    # x2_sample = np.random.multivariate_normal(mu_vec1+0.2,cov_mat1+0.2,100)
    # x3_sample = np.random.multivariate_normal(mu_vec1+0.4,cov_mat1+0.4,100)
    #
    # plt.figure(figsize=(8,6))
    # plt.scatter(x1_sample[:,0],x1_sample[:,1],marker='x',color='blue',alpha=0.3,label='x1')
    # plt.scatter(x2_sample[:, 0], x2_sample[:, 1], marker='o', color='red', alpha=0.3, label='x2')
    # plt.scatter(x3_sample[:, 0], x3_sample[:, 1], marker='^', color='green', alpha=0.3, label='x3')
    # plt.legend(loc='best')
    # plt.show()

    # x_coords = [0.13,0.22,0.39,0.59,0.68,0.74,0.93]
    # y_coords = [0.75,0.34,0.44,0.52,0.80,0.25,0.55]
    # plt.figure(figsize=(8,6))
    # plt.scatter(x_coords,y_coords,marker='s',s=10)
    # #显示坐标
    # for x,y in zip(x_coords,y_coords):
    #     plt.annotate('(%s,%s)'%(x,y),xy = (x,y),xytext=(0,-15),textcoords = 'offset points',ha='center')
    # plt.show()

    # mu_vecl = np.array([0,0])
    # cov_mat1 = np.array([[1,0],[0,1]])
    # x = np.random.multivariate_normal(mu_vecl,cov_mat1,500)
    # fig = plt.figure(figsize=(8,6))
    # r = x ** 2
    # r_sum = r.sum(axis=1)
    # plt.scatter(x[:,0],x[:,1],color='grey',marker='o',s=20*r_sum,alpha=0.5)
    # plt.show()

    # **********matplotlib的3D图*********-----------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # x = np.arange(-4,4,0.25)
    # y = np.arange(-4,4,0.25)
    #
    # X,Y = np.meshgrid(x,y)
    # Z = np.sin(np.sqrt(X**2+Y**2))
    #
    # ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    #画投影
    # ax.contour(X,Y,X,zdim='z',offset=2,cmap='rainbow')
    # ax.set_zlim(-2,2)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # theta = np.linspace(-4*np.pi,4*np.pi,100)
    # z = np.linspace(-2,2,100)
    # r = z**2 + 1
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)
    # ax.plot(x,y,z)
    # plt.show()

    # np.random.seed(1)
    # def randrange(n,vmin,vmax):
    #     return (vmax-vmin)*np.random.rand(n)+vmin
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # n = 100
    # for c,m,zlow,zhigh in [('r','o',-50,-25),('b','x',-30,-5)]:
    #     xs = randrange(n,23,32)
    #     ys = randrange(n,0,100)
    #     zs = randrange(n,zlow,zhigh)
    #     ax.scatter(xs,ys,zs,color=c,marker=m)
    # plt.show()

    # np.random.seed(1)
    # def randrange(n, vmin, vmax):
    #     return (vmax - vmin) * np.random.rand(n) + vmin
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # n = 100
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', 'x', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     #画散点图
    #     ax.scatter(xs, ys, zs, color=c, marker=m)
    # #指定显示角度
    # ax.view_init(40,0)
    # plt.show()

    # fig = plt.figure()
    # ax =fig.add_subplot(111,projection='3d')
    # for c,z in zip(['r','g','b','y'],[30,20,10,]):
    #     xs= np.arange(20)
    #     ys = np.random.rand(20)
    #     cs = [c]*len(xs)
    #     ax.bar(xs,ys,zs=z,zdir='y',color=cs,alpha=0.5)
    # plt.show()

    # **********matplotlib的pie图*********-----------------------------------------------------
    # m = 51212
    # f = 40742
    # m_perc = m/(m+f)
    # f_perc = f/(m+f)
    # colors = ['navy','lightcoral']
    # labels = ['Male','Female']
    # plt.figure(figsize=(8,8))
    # paches,texts,autotexts = plt.pie([m_perc,f_perc],labels=labels,autopct='%1.1f%%',explode=[0,0.5],colors=colors)
    # for text in texts+autotexts:
    #     text.set_fontsize(20)
    # for text in autotexts:
    #     text.set_color('white')
    # plt.show()
    #设置子图布局
    # ax1 = plt.subplot2grid((3,3),(0,0))
    # ax2 = plt.subplot2grid((3,3),(1,0))
    # ax3 = plt.subplot2grid((3,3),(0,2),rowspan=3)
    # ax4 = plt.subplot2grid((3,3),(2,0),colspan=2)
    # ax5 = plt.subplot2grid((3,3),(0,1),rowspan=2)
    # plt.show()
    #图里面画子图
    # x = np.linspace(0,10,1000)
    # y2 = np.sin(x**2)
    # y1 = x**2
    # fig,ax1 = plt.subplots()
    # left,bottom,width,height = [0.22,0.45,0.3,0.35]
    # ax2 = fig.add_axes([left,bottom,width,height])
    # ax1.plot(x,y1)
    # ax2.plot(x,y2)
    # plt.show()

    # **********matplotlib的子图布局*********-----------------------------------------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #根据柱形图的柱的高度值作为标题（可作为模板）
    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax1.text(rect.get_x()+rect.get_width()/2,1.02*height,"{:,}".format(float(height)),ha='center',va='bottom',fontsize=18)
    #
    # top10_arrivals_countries = ['CANADA', 'MEXICO', 'UNITED\KINGDOM', 'JAPAN', 'CHINA', 'GERMANY', 'SOUTH\KOREA',
    #                             'FRANCE', 'BRAZIL', 'AUSTRALIA']
    # top10_arrivals_values = [16.625687, 15.378026, 3.934508, 2.999718, 2.618737, 1.769498, 1.628563, 1.419409, 1.393710,
    #                          1.136974]
    # arrivals_countries = ['WESTERN\EUROPE', 'ASIA', 'SOUTH\AMERICA', 'OCEANIA', 'CARIBBEAN', 'MIDDLE\EAST',
    #                       'CENTRAL\AMERICA', 'EASTERN\EUROPE', 'AFRICA']
    # arrivals_percent = [36.9, 30.4, 13.8, 4.4, 4.0, 3.6, 2.9, 2.6, 1.5]
    # #
    # fig,ax1 = plt.subplots(figsize=(10,12))
    # rects1 = ax1.bar(range(10),top10_arrivals_values,color='blue')
    # # 添加注释
    # plt.xticks(range(10), top10_arrivals_countries, fontsize=18)
    # #这个工具就是在图中画子图
    # ax2 = inset_axes(ax1,width=6,height=6,loc=5)
    # explode = (0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    # patches,texts,autotexts = ax2.pie(arrivals_percent,labels=arrivals_countries,autopct='%1.1f%%',explode=explode)
    # for text in texts+autotexts:
    #     text.set_fontsize(16)
    # for spine in  ax1.spines.values():
    #     #隐藏横轴
    #     spine.set_visible(False)
    # autolabel(rects1)
    # plt.show()


    from matplotlib.patches import Circle, Wedge, Polygon, Ellipse
    from matplotlib.collections import PatchCollection

    # fig, ax = plt.subplots()
    # patches = []
    # # Full and ring sectors drawn by Wedge((x,y),r,deg1,deg2),
    # leftstripe = Wedge((.46, .5), .15, 90, 100)
    # # Full sector by default
    # midstripe = Wedge((.5, .5), .15, 85, 95),
    # rightstripe = Wedge((.54, .5), .15, 80, 90)
    # lefteye = Wedge((.36, .46), .06, 0, 360, width=0.03)  # Ring sector drawn when width <1
    # righteye = Wedge((.63, .46), .06, 0, 360, width=0.03)
    # nose = Wedge((.5, .32), .08, 75, 105, width=0.03)
    # mouthleft = Wedge((.44, .4), .08, 240, 320, width=0.01)
    # mouthright = Wedge((.56, .4), .08, 220, 300, width=0.01)
    # patches += [leftstripe, midstripe, rightstripe, lefteye, righteye, nose, mouthleft, mouthright]
    # # Circles
    # leftiris = Circle((.36, .46), 0.04)
    # rightiris = Circle((.63, .46), 0.04)
    # patches += [leftiris, rightiris]
    # # Polygons drawn by passing coordinates of vertices,
    # leftear = Polygon([[.2, .6], [.3, .8], [.4, .64]], True)
    # rightear = Polygon([[.6, .64], [.7, .8], [.8, .6]], True)
    # topleftwhisker = Polygon([[.01, .4], [.18, .38], [.17, .42]], True)
    # bottomleftwhisker = Polygon([[.01, .3], [.18, .32], [.2, .28]], True)
    # toprightwhisker = Polygon([[.99, .41], [.82, .39], [.82, .43]], True)
    # bottomrightwhisker = Polygon([[.99, .31], [.82, .33], [.81, .29]], True)
    # patches += [leftear, rightear, topleftwhisker, bottomleftwhisker, toprightwhisker, bottomrightwhisker]
    # # Ellipse drawn by Ellipse((x,y),width,height)
    # body = Ellipse((0.5, -0.18), 0.6, 0.8)
    # patches.append(body)
    # # Draw the patches
    # colors = 100 * np.random.rand(len(patches))  # set random colors
    # p = PatchCollection(patches, alpha=0.4)
    # p.set_array(np.array(colors))
    # ax.add_collection(p)
    # # Show the figure
    # plt.show()

    # **********matplotlib的结合pandas和sklearn*********-----------------------------------------------------
    np.random.seed(0)
    import pandas as pd
    # df = pd.DataFrame({'condition 1':np.random.rand(20),
    #                    'condition 2':np.random.rand(20)*0.9,
    #                    'condition 3':np.random.rand(20)*1.1})
    # print df.head()
    from matplotlib.ticker import FuncFormatter
    #堆叠的图按照百分比的形式画出来，不是具体的值是多少
    # df_ratio = df.div(df.sum(axis=1),axis=0)
    # fig,ax = plt.subplots()
    # df_ratio.plot.bar(ax=ax,stacked=True)#是否堆叠
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:'{:.0%}'.format(y)))
    # plt.show()

    # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'
    # df = pd.read_csv(url,na_values = "?")
    # # print df.head()
    # from sklearn.preprocessing import Imputer
    # impute = pd.DataFrame(Imputer().fit_transform(df))
    # impute.columns = df.columns
    # impute.index = df.index
    # # impute.head()
    #
    # import seaborn as sns
    # from sklearn.decomposition import PCA
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # features = impute.drop('Dx:Cancer',axis=1)
    # y = impute['Dx:Cancer']
    # pca = PCA(n_components=3)
    # X_r = pca.fit_transform(features)
    # print("Explained variance:PC1 {:.2%}PC2 {:.2%}PC3 {:.2%}".format(pca.explained_variance_ratio_[0],
    #                                                                  pca.explained_variance_ratio_[1],
    #                                                                  pca.explained_variance_ratio_[2]))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y, cmap=plt.cm.coolwarm)
    # # Label the axes
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # plt.show()

    # **********matplotlib的可视化库seaborn的整体布局风格设置*********-----------------------------------------------------
    import seaborn as sns
    def sinplot(flip=1):
        x = np.linspace(0,14,100)
        for i in range(1,7):
            plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)
    # sns.set()
    # sinplot()
    # plt.show()

    #5种主题风格darkgrid,whitegrid,dark,white,ticks
    # sns.set_style('whitegrid')
    data = np.random.normal(size=(20,6))+np.arange(6)/2
    # sns.boxplot(data=data)
    # sns.set_style("white")
    # sinplot()
    # plt.show()

    # **********matplotlib的可视化库seaborn的风格细节设置*********-----------------------------------------------------
    # sns.violinplot(data)
    #设置图与轴的距离
    # sns.despine(offset=10)
    # plt.show()

    # sns.set_style("whitegrid")
    # sns.boxplot(data=data,palette='deep')
    #隐藏坐边的轴
    # sns.despine(left=True)
    # plt.show()

    # with sns.axes_style('darkgrid'):
    #     plt.subplot(211)
    #     sinplot()
    # plt.subplot(212)
    # sinplot(-1)
    # plt.show()

    # sns.set_context('paper')
    # sns.set_context('talk')
    # sns.set_context('poster')
    # sns.set_context('notebook',font_scale=1.5,re={'lines.linewidth':2.5})
    # plt.figure(figsize=(8,6))
    # sinplot()
    # plt.show()

    # **********matplotlib的可视化库seaborn的调色板*********-----------------------------------------------------
    """
    颜色很重要
    color_palette()能传入任何Matplotlib所支持的颜色
    color_palette()不写参数则默认颜色
    set_palette()设置所有图的颜色
    """
    current_palette = sns.color_palette()
    sns.palplot(current_palette)
    plt.show()

