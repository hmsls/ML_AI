# coding: utf-8
"""
numpy学习
"""
import numpy as np

if __name__ == '__main__':
    #创建array
    array0 = [1,2,3,4,5]
    array1 = np.array(array0)
    #底层结构：<type 'numpy.ndarray'>
    # print type(array1)
    #每个元素都+1
    array2 = array1 + 1
    #对应元素相加
    array3 = array1 + array2
    #对应元素相乘
    array4 = array1 * array2
    #np数组的形状:(5L,)
    # print array4.shape
    #多位数组
    array5 = np.array([[1,2,3],[4,5,6]])
    # print array5.shape
    #取元素
    # print array5[0],'|',array5[1][1]

    tang_list = [1,2,3,4,5]
    tang_array = np.array(tang_list)
    #查看元素类型，一个ndarray结构中的元素类型是一样的，如果不一样，则会进行默认的转换。
    # print tang_array.dtype
    #查看元素占了多少字节
    # print tang_array.itemsize
    #查看元素的个数
    # print tang_array.size  #np.size(tang_array)
    #查看array的维度
    # print tang_array.ndim
    #以某个值来填充一个array
    # tang_array1 = tang_array.fill(8)
    # print tang_array1

    #索引与切片
    # print tang_array[0]
    # print tang_array[-2:]

    #矩阵格式（多维的形式）
    tang_array2 = np.array([[1,2,3],
                            [4,5,6],
                            [7,8,9]])
    # print tang_array2
    # print tang_array2.shape
    # print tang_array2.size
    # print tang_array2.ndim
    # print tang_array2[1,1]
    tang_array2[1,1] = 10
    # print tang_array2
    # print tang_array2[1]
    # print tang_array2[:,1]
    # print tang_array2[1,0:2]
    #这里是将array3指向了array2，所以修改array3会修改array2
    tang_array3 = tang_array2
    tang_array3[1,1] = 100
    # print tang_array2

    #这个是复制一下，修改一个，另一个不会改变
    tang_array4 = tang_array2.copy()
    tang_array4[1,1] = 1000
    # print tang_array4
    # print tang_array2

    #等差数组
    tang_array5 = np.arange(0,100,10)
    # print tang_array5
    # print tang_array5[2]

    #boolean类型
    mask = np.array([0,0,0,1,2,3,1,2,3,4],dtype = bool)
    # print mask

    #使用Boolean类型的数组做索引取值
    # print tang_array5[mask]

    #随机从0-1之间生成数
    random_array = np.random.rand(10)
    mask1 = random_array > 0.5
    # print tang_array5[mask1]

    #得到索引的位置，并通过这个索引的位置找到对应位置的值
    index_location = np.where(tang_array5 > 50)
    # print index_location
    # print tang_array5[np.where(tang_array5 > 50)]

    #**************************************************数组类型***********************************************
    tang_array6 = np.array([1,2,3,4,5],dtype=np.float32)
    #打印这个数组的类型
    # print type(tang_array6)
    #打印这个数组中的元素的类型
    # print tang_array6.dtype
    #打印这个数组占用的字节
    # print tang_array6.nbytes

    tang_array7 = np.array([1,10,3.5,'str'],dtype=object)
    # print tang_array7
    # print tang_array7 * 2
    #改变数组中元素的类型
    tang_array8 = np.array([1,2,3,4,5])
    # print tang_array8.dtype
    #这个方法是把之前的数组中每个元素类型改变后重新创建一个数组
    tang_array9 = np.asarray(tang_array8,dtype=np.float32)
    # print tang_array9.dtype
    #这个方法是把之前的数组中每个元素类型改变后重新创建一个数组
    tang_array10 = tang_array8.astype(np.float32)
    # print tang_array10.dtype

    # **************************************************数值计算***********************************************
    tang_array11 = np.array([[1,2,3],
                             [4,5,6]])
    #所有元素求和
    # print np.sum(tang_array11)
    #按照维度求和
    # print np.sum(tang_array11,axis=0)
    # print np.sum(tang_array11,axis=1)
    #所有元素累乘
    # print np.prod(tang_array11,axis=1)
    #最小值
    # print np.min(tang_array11,axis=1)
    #最大值
    # print np.max(tang_array11,axis=0)
    #最小数值的索引位置
    # print np.argmin(tang_array11,axis=0)
    # print np.argmax(tang_array11,axis=1)
    #均值
    # print np.mean(tang_array11)
    #标准差
    # print np.std(tang_array11,axis=1)
    #方差
    # print np.var(tang_array11,axis=0)
    #限制，把小于等于2的元素都变为2，大于等于4的元素变4
    # print np.clip(tang_array11,2,4)
    #四舍五入
    # print np.round(tang_array11,decimals=2)

    # **************************************************排序操作***********************************************
    tang_array12 = np.array([[1.5,1.3,7.5],
                             [5.6,7.8,1.2]])
    #排序
    # print np.sort(tang_array12,axis=0)
    #返回一个排序后原来的元素的索引位置组成的数组
    # print np.argsort(tang_array12)
    tang_array13 = np.linspace(0,10,10)
    # print tang_array13
    values = np.array([2.5,6.5,9.5])
    #返回一个将values元素插入到tang_array13（需要先排序）数组后，插入位置的索引组成的数组，
    # print np.searchsorted(tang_array13,values)

    #某个列为降序，某个列为升序
    tang_array14 = np.array([
        [1,0,6],
        [1,7,0],
        [2,3,1],
        [2,4,0]
    ])
    #以下是按列排序后，返回一个之前行所在索引值组成的数组
    index =  np.lexsort([-1*tang_array14[:,0],tang_array14[:,2]])
    # print index
    #通过这个索引值数组，然后取出元素组成一个新的数组，这样就实现了两个列不同方向排序的功能
    tang_array15 = tang_array14[index]
    # print tang_array15

    # **************************************************数组形状操作***********************************************
    tang_array16 = np.arange(10)
    # print tang_array16
    #改变数组形状，第一个返回一个数组，第二个是把原有的数组形状改变了
    # tang_array16.reshape(5,2)
    tang_array16.shape = 2,5
    # print tang_array16
    # print tang_array16.reshape(5,2)
    #增加新维度
    tang_array17 = tang_array16[:,np.newaxis]
    # print tang_array17.shape
    # print tang_array17
    #数组压缩，把空的维度压缩掉
    # tang_array17 = tang_array17.squeeze()
    # print tang_array17
    #数组转置
    # print tang_array16.transpose()
    # print tang_array16.T

    #数组的连接
    a = np.array([
        [123,456,789],
        [1234,5678,900]
    ])
    b = np.array([
        [1111,2222,3333],
        [4444,55555,6666]
    ])
    c = np.concatenate((a,b),axis=1)
    c1 = np.vstack((a,b))
    c2 = np.hstack((a,b))
    # print c,c1,c2
    #拉平
    # print a.flatten()
    # print a.ravel()

    # **************************************************数组生成函数***********************************************
    # print np.arange(10,20,2,dtype=np.float32)
    #自动设置平均距离来创建数组，如创建1到10之间的数组，平均距离是10除以9，然后每个数加这个商就是下一个数
    #线性空间分割
    # print np.linspace(0,10,10)
    #以10为底的对数分割
    # print np.logspace(0,1,5)

    #通过两个以为向量构造一个网格的矩阵
    x = np.arange(1,10,2)
    # print x
    y = np.arange(10,20,2)
    # print y
    x,y = np.meshgrid(x,y)
    # print x,y

    #构造行向量和列向量
    # print np.r_[0:10:1]
    # print np.c_[10:20:1]

    # **************************************************常用生成函数***********************************************
    #构造一个全是0的数组，可以是一维可以是多维
    # print np.zeros((3,2))
    #构造一个全是1 的数组，然后倍成8
    # print np.ones((2,3)) * 8
    # print np.ones((2,3),dtype=np.float32)
    #通过empty来构造数组和填充数组
    a = np.empty((2,2))
    # print a.shape
    # a.fill(9)
    # print a
    #根据已知的数组形状，然后构建一个全是0 或1 的数组
    # print np.zeros_like(a)
    # print np.ones_like(a)
    #构建单位矩阵
    b = np.identity(5)
    # print b

    # **************************************************四则运算***********************************************
    x = np.array([5,5])
    y = np.array([2,2])
    #矩阵的X乘，注意点乘和X乘的区别
    # print np.multiply(x,y)
    #矩阵的点乘
    # print np.dot(x,y)

    #数组的直接相乘，虽然会自动补全维度，但是有时候可能出现一些问题，一般要规范写，不要这样。
    # print x * y
    #判断是否相等，这个是对应每个位置的元素都要判断
    # print x == y
    #逻辑与的判断，对应位置的元素都为真为真
    # print np.logical_and(x,y)
    #逻辑或的判断
    # print np.logical_or(x,y)
    #逻辑非判断
    # print np.logical_not(x,y)

    # **************************************************随机模块***********************************************
    #构造一个指定形状的数组，默认值的范围是从0到1
    # print np.random.rand(2,3)
    #构造一个随机的int值，可以指定大小范围和数组形状
    # print np.random.randint(10,size=(5,4))
    # print np.random.randint(10,20,4)
    #得到一个随机样本，默认是0到1之间的
    # print np.random.random_sample()
    #构造一个指定均值和标准差的高斯分布
    mu,sigma = 0,0.1
    # print np.random.normal(mu,sigma,10)
    #设置打印出来的数值的精度
    # np.set_printoptions(precision=2)
    # print np.random.normal(mu,sigma,10)
    #洗牌，打乱之前的顺序
    tang_array19 = np.arange(10)
    # print tang_array19
    np.random.shuffle(tang_array19)
    # print tang_array19
    #随机的种子，这样同一个种子生成的随机数都是一样的。
    np.random.seed(0)
    mu, sigma = 0, 0.1
    # print np.random.normal(mu,sigma,10)

    # **************************************************文件操作***********************************************
    #方法一
    # data = []
    # with open('file.txt') as f :
    #     for line in f.readlines() :
    #         fields = line.split(" ")
    #         cur_data = [float(x) for x in fields]
    #         data.append(cur_data)
    # ar = np.array(data)
    # print ar
    #方法二，指定文件，字段分割符，跳过的行
    ar1 = np.loadtxt('file.txt',delimiter=" ",skiprows=1,usecols=(0,1,2,3))
    # print ar1

    # **************************************************数组保存***********************************************
    ar2 = np.array([[1,2,3],[4,5,6]])
    #保存成TXT格式的，指定格式化形式，和字段分隔符
    # np.savetxt('file1.txt',ar2,fmt='%d',delimiter=',')
    #保存ndarray结构
    # np.save('file2.npy',ar2)
    ar3 = np.load('file2.npy')
    # print ar3
    ar4 = np.arange(10)
    #保存为压缩文件，文件是键值对存储的
    np.savez('file4.npz',a=ar2,b=ar4)
    data1 = np.load('file4.npz')
    # print data1.keys()
    # print data1['b']

    # **************************************************练习题一***********************************************
    #打印Numpy的版本
    # print np.__version__
    #构造一个全零的矩阵，并打印其占用的内存大小
    test_array1 = np.zeros((2,2))
    # print test_array1
    # print ('%d bytes'%(test_array1.size * test_array1.itemsize))
    #打印一个函数的帮助文档，比如numpy.add
    # print np.info(np.add)
    #创建一个10-49的数组，并将其倒序排列
    test_array2 = np.arange(10,50,1)
    # print test_array2
    # print test_array2[::-1]
    #找到一个数组中不为0的索引
    # print np.nonzero([1,2,3,0,2,1,0])
    #随机构造一个3*3矩阵，并打印其中最大与最小值
    test_array3 = np.random.random((3,3))
    # print test_array3.min()
    # print test_array3.max()
    #构造一个5*5的矩阵，令其值都为1，并在最外层加上一圈0
    test_array4 = np.ones((5,5))
    test_array4 = np.pad(test_array4,pad_width=1,mode='constant',constant_values=0)
    # print test_array4
    # print np.info(np.pad)

    # **************************************************练习题二***********************************************
    #构建一个shape为（6,7,8）的矩阵，并找到第100个元素的索引值
    # print np.unravel_index(100,(6,7,8))
    #对于一个5*5的矩阵做归一化操作：每个元素减去最小值，除以，最大值减去最小值
    test_array5 = np.random.random((5,5))
    test_array5_max = test_array5.max()
    test_array5_min = test_array5.min()
    toOne = (test_array5 - test_array5_min) / (test_array5_max - test_array5_min)
    # print test_array5
    #找到2个数组中相同的值
    z1 = np.random.randint(0,10,10)
    z2 = np.random.randint(0,10,10)
    # print z1,'***',z2
    # print np.intersect1d(z1,z2)
    #得到今天、明天、昨天的日期
    yesterday = np.datetime64('today','D') - np.timedelta64(1,'D')
    today = np.datetime64('today','D')
    tommorow = np.datetime64('today','D') + np.timedelta64(1,'D')
    # print yesterday,'***',today,'***',tommorow
    #得到一个月中所有的天
    # print np.arange('2017-10','2017-11',dtype='datetime64[D]')
    #得到一个数的整数部分
    z = np.random.uniform(0,10,10)
    # print np.floor(z)
    #构造一个数组，让他不能被改变
    z3 = np.zeros(5)
    z3.flags.writeable = False
    # print z3
    #打印一个数据量很大的数据的部分值，全部值
    # np.set_printoptions(threshold=5)  #如果要全部打印，则设置阈值为：np.nan
    z4 = np.zeros((15,15))
    # print z4

    # **************************************************练习题三***********************************************
    #找到在一个数组中，最接近一个数的索引
    z5 = np.arange(100)
    v = np.random.uniform(0,100)
    # print v
    index = (np.abs(z5-v)).argmin()
    # print z5[index]
    #32位float类型和32位int类型转换
    z6 = np.arange(10,dtype=np.int32)
    # print z6.dtype
    z6 = z6.astype(np.float32)
    # print z6.dtype
    #打印数组元素位置坐标与数值
    a = np.arange(9).reshape((3,3))
    # for index,value in np.ndenumerate(a):
    #     print (index,value)
    #按照数组的某一列进行排序
    b = np.random.randint(0,10,(3,3))
    # print b
    # print b[:,1].argsort()
    # print b[b[:,1].argsort()]
    #统计数组中每个数值出现的次数
    c = np.array([1,2,1,2,1,2,3,3,4,4,4,4,4])
    # print np.bincount(c) #结果是从0开始，0有0个，1有3个，2有3个，3有2个，4有5个
    #如何对一个思维数组的最后两维求和
    d = np.random.randint(0,10,(4,4,4,4))
    # print d
    res = d.sum(axis=(-2,-1))
    # print res
    #交换矩阵中的2行
    e = np.arange(25).reshape(5,5)
    # print e[[0,1]]
    e[[0,1]] = e[[1,0]]
    # print e
    #找到一个数组中最常出现的数字
    # np.set_printoptions(threshold=np.nan)
    f = np.random.randint(0,7,10)
    # print f
    # print np.bincount(f).argmax()
    #快速查找top K
    g = np.arange(10)
    np.random.shuffle(g)
    K = 5
    # print g[np.argpartition(-g,K)[:K]]
    # 去掉一个数组中，所有元素都相同的数据
    h = np.random.randint(0,5,(10,3))
    print h
    h1 = np.array([1,2,3,4])
    h2 = np.array([1,2,3,5])
    # print np.all(h1 == h2)
    # print np.any(h1 == h2)
    print np.all(h[:,1:] == h[:,:-1],axis=1)