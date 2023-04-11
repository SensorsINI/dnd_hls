import numpy as np
import cv2
# import Sequence

from keras.utils import Sequence
class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """
    def __init__(self, df, list_IDs,
                 to_fit=True, batch_size=8, dim=(256, 472),
                 n_channels=3, n_classes=13, shuffle=True):
        """ 初始化方法
        :param df: 存放数据路径和标签的数据框
        :param list_IDs: 数据索引列表
        :param to_fit: 设定是否返回标签y
        :param batch_size: batch size 
        :param dim: 图像大小
        :param n_channels: 图像通道
        :param n_classes: 标签类别
        :param shuffle: 每一个epoch后是否打乱数据
        """
        self.df = df
        self.list_IDs = list_IDs
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __getitem__(self, index):
        """generate each batch of training data
        """
        # 生成批次索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 索引列表
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # 生成数据
        X = self._generate_X(list_IDs_temp)
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X
        
    def __len__(self):
        """每个epoch下的批次数量
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        
    def _generate_X(self, list_IDs_temp):
        # generate each batch image 
        # 初始化
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # 生成数据
        for i, ID in enumerate(list_IDs_temp):
            # 存储一个批次
            # X[i,] = self._load_image(self.df.iloc[ID].images)
            X[i,] = self.df.iloc[ID]
        return X
 
 
    def _generate_y(self, list_IDs_temp):
        """生成每一批次的标签
        :param list_IDs_temp: 批次数据索引列表
        :return: 一个批次的标签
        """
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # y[i,] = self._labels_encode(self.df.iloc[ID].labels, config.LABELS)
            y[i,] = self.df.iloc[ID]

        return y
        
    def on_epoch_end(self):
        """每个epoch之后更新索引
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _load_image(self, image_path):
        """cv2读取图像
        """
        # img = cv2.imread(image_path)
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        w, h, _ = img.shape
        if w>h:
            img = np.rot90(img)
        img = cv2.resize(img, (472, 256))
        return img
        
    def _labels_encode(self, s, keys):
        """标签one-hot编码转换
        """
        cs = s.split('_')
        y = np.zeros(13)
        for i in range(len(cs)):
            for j in range(len(keys)):
                for c in cs:
                    if c == keys[j]:
                        y[j] = 1
        return y