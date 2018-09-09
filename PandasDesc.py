#coding: utf-8
"""
Pandas：数据分析处理库，封装了Numpy，一般用这个
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
if __name__ == '__main__':
    #**********************************************************Pandas概述**********************************************
    #从csv中读取数据
    df_file = pd.read_csv(r'titanic.csv')
    # print df
    #读取前几行的数据
    # print df.head(10)
    #返回当前对象的信息：.info
    # print df.info()
    #索引值
    # print df.index
    #列名
    # print df.columns
    #列类型
    # print df.dtypes
    #列的值
    # print df.values

   #**********************************************************Pandas基本操作**********************************************
    #自己创建一个DataFrame对象
    data = {'country':['aaa','bbb','ccc'],
            'population':[10,12,14]}
    df_data = pd.DataFrame(data)
    # print df_data
    # print df_data.info()
    #选取某一列，Nan，表示不是一个数值
    age  = df_file['Age']
    # print age[:5]
    #series结构，是dataframe中的一行或者一列，比如上面的Age
    # print age.index
    # print age.values
    # print age.head(10)
    #手动设置索引，默认是0，1,2,3,4顺序
    df = df_file.set_index('Name')
    age1 = df['Age']
    # print age1[:5]
    #查找某一行数据，根据索引
    # print age1['Braund, Mr. Owen Harris']
    #运算
    age1 = age1 + 10
    # print age1
    # print age.mean()
    # print age.max()
    # print age.min()
    #自动的一些统计指标
    # print df.describe()

    # **********************************************************Pandas索引**********************************************
    # print df['Age'][:5]
    #选取2列来展示
    # print df[['Age','Fare']][:5]
    #获取指定数据：1、loc用label（标签、字段）去定位。2、iloc用position（具体位置坐标）去定位
    # print df.iloc[0]
    # print df.iloc[0:5]
    #选取部分列,选取123列，选取5行内容
    # print df.iloc[0:5,1:3]
    #根据标签（字段）来定位一个数据
    # print df.loc['Heikkinen, Miss. Laina']
    # print df.loc['Heikkinen, Miss. Laina':'Allen, Mr. William Henry',:]
    #修改数据
    # df.loc['Heikkinen, Miss. Laina','Fare'] = 1000
    # print df.head()

    #bool类型的索引
    # print df[df['Fare']>40][:5]
    # print df[df['Sex'] == 'male'][:5]
    # print df.loc[df['Sex'] == 'male','Age'].mean()
    # print (df['Age']>40).sum()

    # **********************************************************Pandas的group by**********************************************
    df1 = pd.DataFrame({'key':['A','B','C','A','B','C','A','B','C'],
                        'data':[5,10,15,5,10,23,10,15,5]})
    # print df1
    # print df1[df1['key'] == 'A']
    #计算某个字段的值的和
    # for k in ['A','B','C']:
    #     print k,'***',(df1[df1['key'] == k]).sum()
    #group by
    # print df1.groupby('key').sum()

    # print df1.groupby('key').aggregate(np.mean)
    # print df.groupby('Sex')['Age'].mean()
    # print df.groupby('Sex')['Survived'].mean()

    # **********************************************************Pandas的数值运算**********************************************
    df2 = pd.DataFrame([[1,2,3],[4,5,6]],index=['a','b'],columns=['A','B','C'])
    # print df2
    # print df2.sum(axis=1)
    # print df2.sum()
    # print df2.sum(axis='columns')
    # print df2.mean()
    # print df2.min()
    # print df2.median()
    #二元统计
    #协方差
    # print df.cov()
    #相关系数
    # print df.corr()
    #不同值的个数统计，如年龄为23的有多少个
    # print df['Age'].value_counts()
    # print df['Age'].value_counts(ascending=True)  #升序
    # print df['Age'].value_counts(ascending=True,bins=10) #分组，分为5组，一组为一个范围内的值的个数
    #统计某一列的样本的数量
    # print df['Age'].count()
    #查看帮助
    # print help(pd.value_counts())

    # **********************************************************Pandas的对象操作**********************************************
    #series结构，
    data = [10,11,12]
    index = ['a','b','c']
    s = pd.Series(data=data,index=index)
    # print s
    # print s[0]
    # print s[0:2]
    # print s[[True,False,True]]
    # print s.loc['b']
    # print s.iloc[1]
    #复制
    s1 = s.copy()
    # s1['a'] = 100
    # print s1
    #是否修改复制的原始值
    # s1.replace(to_replace=100,value=101,inplace=False)
    # print s1
    #修改索引
    # print s1.index
    # s1.index = ['a','b','d']
    # print s1
    # s1.rename(index={'a':'A'},inplace = True)
    # print s1
    #增加行包括索引和数据
    s2 = pd.Series([100,500],index=['g','h'])
    # print s1.append(s2)
    s3 = s1.append(s2,ignore_index=True)  #这里ignore_index表示是否要重新创建索引，如果true，则改为默认的0,1,2，。。。
    s3['j'] = 600
    # print s3

    ###删除操作####
    # print s1
    # del s1['a']
    # print s1
    # s1.drop(['b','c'],inplace = True)
    # print s1

    ###DataFrame的增删改查
    data = [[1,2,3],[4,5,6]]
    index = ['a','b']
    columns = ['A','B','C']
    df = pd.DataFrame(data=data,index=index,columns=columns)
    # print df
    #查
    # print df.iloc[0]
    # print df.loc['a']
    #改
    # df.loc['a']['A']=150
    # print df
    # df.index = ['f','g']
    # print df
    #增
    # df.loc['c'] = [1,2,3]
    # print df
    df2 = pd.DataFrame(data=[[1,2,3],[4,5,6]],index=['j','k'],columns=['A','B','C'])
    df3 = pd.concat([df,df2],axis=1)
    # print df3
    # df2['tang'] = [10,11]
    # print df2
    df4 = pd.DataFrame([[10,11],[12,13]],index=['j','k'],columns=['D','E'])
    df5 = pd.concat([df2,df4],axis=1)
    # print df5
    #删
    #删行
    # df5.drop(['j'],axis=0,inplace=True)
    # print df5
    #删列
    # del df5['E']
    # print df5
    #批量删除列
    # df5.drop(['A','B','C'],axis=1,inplace=True)
    # print df5

    # **********************************************************Pandas的merge操作**********************************************
    left = pd.DataFrame({'key1':['K0','K1','K2','k1'],
                         'key2': ['K0', 'K1', 'K2', 'k2'],
                            'A':['A0','A1','A2','A3'],
                            'B':['B0','B1','B2','B3']})
    right = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'k3'],
                          'key2': ['K0', 'K1', 'K2', 'k4'],
                         'C': ['C0', 'C1', 'C2', 'C3'],
                         'D': ['D0', 'D1', 'D2', 'D3']})
    # print left,right
    #合并（内连接）、左连接、右连接
    # print pd.merge(left,right,on='key1')
    # print pd.merge(left, right, on=['key1','key2'],how='outer',indicator=True)
    # print pd.merge(left, right, on=['key1','key2'],how='left',indicator=True)

    # **********************************************************Pandas的显示设置**********************************************
    # print pd.get_option('display.max_rows')
    # s1 = pd.Series(index=range(0,100))
    #设置显示行数
    pd.set_option('display.max_rows',100)
    # print s1
    # print pd.get_option('display.max_columns')
    #设置显示列数
    pd.set_option('display.max_columns',100)
    d1 = pd.DataFrame(columns=range(0,50))
    # print d1
    #打印字符串长度
    # print pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', 70)
    s2 = pd.Series(index=['A'],data=['t'*70])
    # print s2
    #设置数字精度
    # print pd.get_option('display.precision')
    ss = pd.Series(data='1.234567891011121111111')
    pd.set_option('display.precision',20)
    # print ss

    # **********************************************************Pandas的数据透视表**********************************************
    example = pd.DataFrame({'Month': ["January", "January", "January", "January","February", "February", "February", "February", "March", "March", "March", "March"],
                               'Category': ["Transportation", "Grocery", "Household", "Entertainment","Transportation", "Grocery", "Household", "Entertainment","Transportation", "Grocery", "Household", "Entertainment"],
                               'Amount': [74., 235., 175., 100., 115., 240., 225., 125., 90., 260., 200., 120.]})
    # print example
    #数据透视图
    example_pivot = example.pivot(index='Category',columns='Month',values='Amount')
    # print example_pivot
    #统计某行信息
    # print example_pivot.sum(axis=1)
    #统计某列信息
    # print example_pivot.sum(axis=0)
    #统计泰坦尼克号数据的性别船舱等级、价格的平均值（默认就是平均值）
    # print df_file.pivot_table(index='Sex',columns='Pclass',values='Fare')
    #统计泰坦尼克号数据的性别船舱等级、价格的最大值
    # print df_file.pivot_table(index='Sex',columns='Pclass',values='Fare',aggfunc='max')
    # 统计泰坦尼克号数据的性别船舱等级、人数的统计
    # print df_file.pivot_table(index='Sex', columns='Pclass', values='Fare', aggfunc='count')
    #另一种方式
    # print pd.crosstab(index=df['Sex'],columns=df['Pclass'])  #有问题

    # **********************************************************Pandas的时间操作**********************************************
    dt = datetime.datetime(year=2017,month=11,day=24,hour=10,minute=30)

    ts = pd.Timestamp('2017-11-24')
    # print ts.month
    # print ts.day
    # print ts + pd.Timedelta('5 days')
    # print pd.to_datetime('2018-06-26')

    s = pd.Series(['2018-06-26','2018-06-27','2018-06-28'])
    # print s
    #由字符串转换为时间类型，这样的话可以找出年月日、时分秒等信息
    ts = pd.to_datetime(s)
    # print ts
    # print ts.dt.hour
    # print ts.dt.weekday
    #构造时间列表
    # print pd.Series(pd.date_range(start='2018-06-26',periods=10,freq='12H'))
    #从文件中对时间操作，方法一，一步到位
    data = pd.read_csv('flowdata.csv',index_col=0,parse_dates=True)
    # print data
        #方法二，先读数据，然后处理
    # data['Time'] = pd.to_datetime(data['Time'])
    # data = data.set_index('Time')
    # print data.index

    # **********************************************************Pandas的时间序列操作**********************************************
    # print data[pd.Timestamp('2012-01-01 09:00'):pd.Timestamp('2012-01-01 19:00')]
    # print data[(data.index.hour>8) & (data.index.hour<12)]
    # print data.between_time('08:00','12:00')
    # print data.resample('3D').mean().head()
    # print data.resample('M').mean().plot()
    # **********************************************************Pandas的常用操作**********************************************
    data = pd.DataFrame({'group':['a','a','a','b','b','b','c','c','c'],
                         'data':[4,3,2,1,12,3,4,5,7]})
    # print data
    #按照值进行排序，在原来的数据上，group倒序，data正序
    data.sort_values(by=['group','data'],ascending=[False,True],inplace=True)
    # print data
    data = pd.DataFrame({'k1':['one']*3+['two']*4,
                         'k2':[3,2,1,3,3,4,4]})
    # print data
    # data.sort_values(by='k2')
    #去掉重复值，去重不是把原数组去重，而是生成新的数组
    # print data.drop_duplicates()
    #指定去重的列
    # print data.drop_duplicates(subset='k1')
    # print data
    data = pd.DataFrame({'food':['A1','A2','B1','B2','B3','C1','C2'],
                         'data':[1,2,3,4,5,6,7]})
    # print data
    def food_map(series):
        if series['food'] == 'A1':
            return 'A'
        elif series['food'] == 'A2':
            return 'A'
        elif series['food'] == 'B1':
            return 'B'
        elif series['food'] == 'B2':
            return 'B'
        elif series['food'] == 'B3':
            return 'B'
        elif series['food'] == 'C1':
            return 'C'
        elif series['food'] == 'C2':
            return 'C'
    #apply函数，通过这个函数，将dataframe中按照行或列都去执行一个函数，从而得到一个新的值
    data['food_map'] = data.apply(food_map,axis='columns')
    # print data
    #map函数，通过这个函数，将dataframe中的按照行或列都去执行一个映射（map映射），从而得到一个新的值
    food2Upper = {
        'A1': 'A',
        'A2':'A',
        'B1':'B',
        'B2': 'B',
        'B3': 'B',
        'C1': 'C',
        'C2': 'C',
    }
    data['upper'] = data['food'].map(food2Upper)
    # print data
    # **********************************************************Pandas的常用操作2**********************************************
    df = pd.DataFrame({'data1':np.random.randn(5),
                       'data2':np.random.randn(5)})
    # print df.assign(ration=df['data1']/df['data2'])
    df2 = df.assign(ration=df['data1']/df['data2'])
    # print df2
    # 删除一列
    df2.drop('ration',axis='columns',inplace=True)
    # print df2

    data = pd.Series([1,2,3,4,5,6,7,8,9])
    #替换值
    data.replace(9,np.nan,inplace=True)
    # print data
    ages = [15,18,20,21,22,34,41,52,63,70]
    #离散化
    # bins = [10,40,80]
    # bins_res = pd.cut(ages,bins)
    # print bins_res
    #为分组后的各组进行编号码，测试不行
    # a = bins_res.labels
    # print a
    #计算个数
    # print pd.value_counts(bins_res)
    #设置标签名称
    group_name = ['Yonth','Mille','old']
    # print pd.value_counts(pd.cut(ages,[10,20,50,80],labels=group_name))

    df = pd.DataFrame([range(3),[0,np.nan,0],[0,0,np.nan],range(3)])
    # print df
    #查找缺失值，按列看
    # print df.isnull().any()
    #按照行看
    # print df.isnull().any(axis=1)
    #填充缺失值
    # print df.fillna('niha')
    # print df[df.isnull().any(axis=1)]

    # **********************************************************Pandas的groupby延伸**********************************************
    df = pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','foo'],
                       'B':['one','one','tow','three','two','two','one','three'],
                       'C':np.random.randn(8),
                       'D':np.random.randn(8)})
    # print df
    #groupby
    grouped = df.groupby('A')
    # print grouped.count()
    # print df.groupby(['A','B']).count()
    def get_letter_type(letter):
        if letter.lower() in 'aeiou':
            return 'a'
        else:
            return 'b'

    grouped = df.groupby(get_letter_type,axis=1)
    # print grouped.count().iloc[0]

    s = pd.Series([1,2,3,1,2,3],[8,7,5,8,7,5])
    # print s
    grouped = s.groupby(level=0,sort=False)
    # print grouped.first()
    # print grouped.last()
    # print grouped.sum()
    # grouped = df.groupby(['A','B'],as_index=False)
    # grouped = df.groupby(['A', 'B']).sum().reset_index()
    # print grouped.aggregate(np.sum)
    # print grouped
    #查看个数
    # print df.groupby(['A','B']).size()
    #查看统计值
    # print df.groupby(['A','B']).describe().head()
    #
    # print df.groupby('A')['C'].agg([np.sum,np.mean,np.std])
    df2 = pd.DataFrame({'X':['A','B','A','B'],
                        'Y':[1,2,3,4]})
    # print df2
    #如果只想看到A的排序，则只选A的
    # print df2.groupby(['X']).get_group('A')
    arrays = [['bar','bar','baz','baz','foo','foo','qux','qux'],
              ['one','two','one','two','one','two','one','two']]
    #指定多级索引
    index = pd.MultiIndex.from_arrays(arrays,names=['first','second'])
    s = pd.Series(np.random.randn(8),index=index)
    # print s
    grouped = s.groupby(level='first')
    # print grouped.sum()
    # **********************************************************Pandas的字符串操作**********************************************
    s = pd.Series(['A','b','B','gare','AWGA',np.nan])
    #小写转换
    # print s.str.lower()
    #转换大写
    # print s.str.upper()
    #长度
    # print s.str.len()
    index = pd.Index(['   tang','  yu   ','di'])
    #去空格
    # print index.str.strip()
    #去左边的空格
    # print index.str.lstrip()
    # print index.str.rstrip()
    df = pd.DataFrame(np.random.randn(3,2),columns=['A a','B b'],index = range(3))
    #更换列名
    df.columns = df.columns.str.replace(' ','_')
    # print df
    s = pd.Series(['a_b_c','c_d_e','F_g_h'])
    #切分，并且生成一个df，并且指定切分次数
    # print s.str.split('_',expand=True,n=1)
    #判断是否包含
    s = pd.Series(['A','Adfg','Add','bbadf','dsad'])
    # print s.str.contains('Adf')
    #有些有分割符，有些没有，都拿出来
    s = pd.Series(['a','a|b','c|d'])
    # print s.str.get_dummies(sep='|')
    # **********************************************************Pandas的索引进阶**********************************************
    s = pd.Series(np.arange(5),index=np.arange(5)[::-1],dtype='int64')
    #判断是否在某个series中
    # print s.isin([1,2,4])
    #通过这个isin返回的bool值来取数据
    # print s[s.isin([1,3,5])]
    #多重索引
    s2 = pd.Series(np.arange(6),index=pd.MultiIndex.from_product([[0,1],['a','b','c']]))
    #根据索引的isin反回的bool值来定位所在位置
    # print s2.iloc[s2.index.isin([(1,'a'),(0,'b')])]

    #
    dates = pd.date_range('20180708',periods=8)
    df = pd.DataFrame(np.random.randn(8,4),index=dates,columns=['A','B','C','D'])
    #选取数据（其实可直接df['A']
    A = df.select(lambda x:x=='A',axis='columns')
    # print A
    #where操作，选取满足条件的值，不满足条件的置为空
    # print df.where(df<0)
    #where操作，选择满足条件的值，不满足的按照后面的进行操作
    # print df.where(df<0,-df)

    #query
    df = pd.DataFrame(np.random.randn(10,3),columns=list('abc'))
    # print df.query('a<b')
    # print df.query('(a<b) & (b<c)')

    # **********************************************************Pandas的绘图**********************************************
    #注意：如果要在IED中画图，则要加上plt.show()这个plt是matplotlib.pyplot
    s = pd.Series(np.random.randn(10),index=np.arange(0,100,10))
    #series的绘图
    # s.plot()
    #下面这个是显示画图的效果，必须加在plot的后面才生效
    # plt.show()
    df = pd.DataFrame(np.random.randn(10,4).cumsum(0),index=np.arange(0,100,10),columns=['A','B','C','D'])
    # df.plot()
    # plt.show()
    # fig,axes = plt.subplots(2,1)
    #注意，关于randn和rand这两个随机生成数据的函数，带n的是可以生成负数，不带都是正数（我猜的）
    data = pd.Series(np.random.rand(16),index=list('abcdefghigklmnop'))
    #bar指的是柱状图，barh是指柱状图，但是横过来
    # data.plot(ax=axes[0],kind='bar')
    # data.plot(ax=axes[1],kind='barh')
    # plt.show()

    df = pd.DataFrame(np.random.rand(6,4),index=['one','two','three','four','five','six'],columns=
                      pd.Index(['A','B','C','D'],name = 'Genus'))
    # df.plot(kind='bar')
    # plt.show()

    tips = pd.read_csv(r'tips.csv')
    #画直方图，bins表示分成50个小格来表示
    # tips.total_bill.plot(kind='hist',bins=50)
    # plt.show()

    #散点图
    macro = pd.read_csv(r'macrodata.csv')
    data = macro[['quarter','realgdp','realcons']]
    # pd.scatter_matrix(data,color='k',alpha=0.3)
    # data.plot.scatter('quarter','realgdp')
    # plt.show()
    # **********************************************************Pandas的大数据处理技巧**********************************************
    g1 = pd.read_csv(r'game_logs.csv')
    # print g1.head()
    #查看数据的行数与列数
    # print g1.shape
    #查看这个数据的信息,详细查看内存的使用量
    # print g1.info(memory_usage='deep')
    # for dtype in ['float64','int64','object']:
    #     selected_dtype = g1.select_dtypes(include=[dtype])
    #     mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    #     mean_usage_mb = mean_usage_b / 1024 ** 2
        # print '平均内存占用:',dtype,"---",mean_usage_mb
    #查看各个类型能取到的最大数值是多少
    int_types = ['uint8','int8','int16','int32','int64']
    # for it in int_types:
    #     print np.iinfo(it)

    def mem_usage(pandas_obj):
        if isinstance(pandas_obj,pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else:
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2
        return '{:03.2f} MB'.format(usage_mb)
    #降低内存使用，通过改变类型，由int64转换为无符号类型，省出空间
    g1_int = g1.select_dtypes(include=['int64'])
    coverted_int = g1_int.apply(pd.to_numeric,downcast='unsigned')
    # print mem_usage(g1_int)
    # print mem_usage(coverted_int)
    #通过从float64转换为float类型，内存空间可以省一半
    g1_float = g1.select_dtypes(include=['float64'])
    coverted_float = g1_float.apply(pd.to_numeric,downcast='float')
    # print mem_usage(g1_float)
    # print mem_usage(coverted_float)
    #调整整个的内存占用
    optimized_g1 = g1.copy()
    optimized_g1[coverted_int.columns] = coverted_int
    optimized_g1[coverted_float.columns] = coverted_float

    # print mem_usage(g1)
    # print mem_usage(optimized_g1)

    g1_obj = g1.select_dtypes(include=['object'])
    # print g1_obj.describe()
    #字符串是一个占用一块空间，可以吧相同的指向同一个空间，这就省出了空间
    dow = g1_obj.day_of_week
    dow_cat = dow.astype('category')#将字符串格式转换成category格式，节省空间
    #查看编码格式，相同的值用相同的编码表示
    # print dow_cat.head(10).cat.codes
    # print mem_usage(dow)
    # print mem_usage(dow_cat)

    coverted_obj = pd.DataFrame()
    for col in g1_obj.columns:
        num_unique_values = len(g1_obj[col].unique())
        num_total_values = len(g1_obj[col])
        if num_unique_values / num_total_values < 0.5:
            coverted_obj.loc[:,col] = g1_obj[col].astype('category')
        else:
            coverted_obj.loc[:,col] = g1_obj[col]

    # print mem_usage(g1_obj)
    # print mem_usage(coverted_obj)
    #处理日期
    date = optimized_g1.date
    # print mem_usage(date)
    #这个是转换成标准时间格式，但是这种格式占空间
    optimized_g1['date'] = pd.to_datetime(date,format='%Y%m%d')
    print mem_usage(optimized_g1['date'])