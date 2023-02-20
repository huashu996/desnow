import rospy
import time
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
import model.model
import tensorflow as tf
import keras.backend as K
from PIL import Image as Img
from keras.models import load_model
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator


class Desnow:
    
    def __init__(self):
        
        # 定义发布者
        self.pub = rospy.Publisher("/desnow_img", Image, queue_size=1)
        self.pub2= rospy.Publisher("/refog_img", Image, queue_size=1)
        self.get_image(self.pub)
       

    def progress(self, count, total, status=''):

	    self.bar_len = 60
	    self.filled_len = int(round(self.bar_len * count / float(total)))
	    self.percents = round(100.0 * count / float(total), 1)
	    self.bar = '|' * self.filled_len + '-' * (self.bar_len - self.filled_len)
	    sys.stdout.write('[%s] %s%s ...%s\r' % (self.bar, self.percents, '%', status))
	    if count != total:
	        sys.stdout.flush()
	    else:
	        print()
    
	    
    def generate_data_generator(self, datagenerator, X, BATCHSIZE):

	    self.genX1 = datagenerator.flow(X,batch_size = BATCHSIZE,shuffle=False)
	    self.count = 0
	    while True:
	        self.Xi1 = self.genX1.next()
	        self.Xi1 = self.Xi1/255
	        yield [self.Xi1]


    def publish_image(self, pub, data, frame_id='base_link'):

	    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
	    self.header = Header(stamp=rospy.Time.now())
	    self.header.frame_id = frame_id    
	    self.msg = Image()
	    self.msg.height = data.shape[0]
	    self.msg.width = data.shape[1]
	    self.msg.encoding = 'rgb8'
	    self.msg.data = np.array(data).tostring()
	    self.msg.header = self.header
	    self.msg.step = self.msg.width * 1 * 3
	    self.pub.publish(self.msg)


    def get_image(self,pub):

        print('Start...')
        rospy.loginfo(rospy.get_caller_id() + "=Loading The Image...")

        # 输入图像
        folder_path = "/home/seucar/task/desnow/src/image/";
        count = os.listdir(folder_path)
        count.sort()
        print(len(count))
        i=0
        while i<len(count):
            image_path = "/home/seucar/task/desnow/src/image/"+count[i]
            self.save_path ="/home/seucar/task/desnow/src/desnow_image/"+count[i]
            self.img = cv2.imread(image_path)
            self.DATA = []
            if self.img.shape[1]<self.img.shape[0]:
                self.img = np.rot90(self.img)
            if self.img.shape[0] != 480 or self.img.shape[1] != 640:
                self.img = cv2.resize(self.img, (640, 480), interpolation=cv2.INTER_CUBIC)
            self.DATA.append(self.img)
            self.DATA = np.array(self.DATA)
            print(self.DATA.shape,'DATA shape3')
            print('Start Padding')
            self.progress(2,self.DATA.shape[0],'Paddding and convert DATA to YCRCB...')
            self.DATA[0] = cv2.cvtColor(self.DATA[0],cv2.COLOR_BGR2YCR_CB)
            self.DATA = np.pad(self.DATA,((0,0),(16,16),(16,16),(0,0)),'constant')    
            print('Got The Image!')
            
            # 计算话题输出
            print('----------Computing...----------')
            with graph.as_default():
                self.val_data_gen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)
                self.pred = model.predict_generator(self.generate_data_generator(self.val_data_gen,self.DATA, args.batch_size),steps = self.DATA.shape[0]/args.batch_size,verbose=1)
                print('Computing Ok!')

                # 输出结果处理
                self.progress(2,self.pred.shape[0],'Saving output...')
                self.pred[0] = np.clip(self.pred[0],0.0,1.0)
                self.image = cv2.cvtColor( (self.pred[0] * 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
                self.refog(self.pub2,self.image,self.save_path)
                print('Got The Result!')
                
                # 发布话题
                self.publish_image(self.pub, self.image, frame_id='base_link')
            i+=1
    def refog(self,pub,image,path):
        img = cv2.resize(image, (720, 405), interpolation=cv2.INTER_LINEAR)
        m = self.deHaze(img/255)*255
        m = np.clip(m,a_min=0,a_max=255)
        new_image_save = cv2.resize(m, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, new_image_save)
        m = m.astype(np.int8)  #在转化为8位数
        self.publish_image2(self.pub2,m,'base_link') 
    def deHaze(self,img, r=600, eps=0.001, w=0.55, maxV1=0.01, bGamma=True): #r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False

        Y = np.zeros(img.shape)
        V1,A = self.getV1(img, r, eps, w, maxV1)        #得到遮罩图像和大气光照
        for k in range(3):
	        Y[:,:,k] = (img[:,:,k]-V1)/(1-V1/A)      #颜色校正
        Y = np.clip(Y, 0, 1)
        #if bGamma:
        Y = Y**(np.log(0.3)/np.log(Y.mean()))    #gamma校正, 越大越亮  0.3
        Y = self.imgBrightness(Y, 2.9, -0.05)   #第一个值是亮度，第二个是对比度 2.8 -0.15

        return Y
    def imgBrightness(self,img1, a, b): 
        h, w, ch = img1.shape#获取shape的数值，height和width、通道
        #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        src2 = np.zeros([h, w, ch], img1.dtype)
        dst = cv2.addWeighted(img1, a, src2, 1-a, b)#addWeighted函数说明如下
        return dst
    def getV1(self,m, r, eps, w, maxV1): #输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = np.min(m,2)                     #得到暗通道图像
        V1 = self.guidedfilter(V1, self.zmMinFilterGray(V1,7), r, eps)   #使用引导滤波优化
        bins = 2000    #2000
        ht = np.histogram(V1, bins)               #计算大气光照A
        d = np.cumsum(ht[0])/float(V1.size)
        for lmax in range(bins-1, 0, -1):
	        if d[lmax]<=0.999:
		        break
        A = np.mean(m,2)[V1>=ht[1][lmax]].max()
		
        V1 = np.minimum(V1*w, maxV1)          #对值范围进行限制
	
        return V1,A
    def zmMinFilterGray(self,src, r=7):
        return cv2.erode(src, np.ones((2*r+1, 2*r+1)))           #使用opencv的erode函数更高效
    def guidedfilter(self,I, p, r, eps):
	        '''引导滤波，直接参考网上的matlab代码'''
	        height, width = I.shape
	        m_I = cv2.boxFilter(I, -1, (r,r))
	        m_p = cv2.boxFilter(p, -1, (r,r))
	        m_Ip = cv2.boxFilter(I*p, -1, (r,r))
	        cov_Ip = m_Ip-m_I*m_p
	
	        m_II = cv2.boxFilter(I*I, -1, (r,r))
	        var_I = m_II-m_I*m_I
	
	        a = cov_Ip/(var_I+eps)
	        b = m_p-a*m_I
	
	        m_a = cv2.boxFilter(a, -1, (r,r))
	        m_b = cv2.boxFilter(b, -1, (r,r))
	        return m_a*I+m_b

    def publish_image2(self,pub, data, frame_id='base_link'):
        assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
        #print(data.shape)
        header = Header(stamp=rospy.Time.now())
        header.frame_id = frame_id

        msg = Image()
        msg.height = data.shape[0]
        msg.width = data.shape[1]
        msg.encoding = 'rgb8'  #传入数据要求是8位
        msg.data = np.array(data).tostring()
        msg.header = header
        msg.step = msg.width *1*3
        #print("**************************")
        pub.publish(msg)
        #print("---------------------------")

def parse_args():

    parser = ArgumentParser(description='Predict')
    parser.add_argument('-dataroot', '--dataroot', type=str, default='./input', help='root of the image, if data type is npy, set datatype as npy')
    parser.add_argument('-datatype', '--datatype', type=str, default=['jpg','tif','png'], help='type of the image, if == npy, will load dataroot')
    parser.add_argument('-predictpath', '--predictpath', type=str, default='./predictImg', help='root of the output')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=3, help='batch_size')
    return  parser.parse_args()

# 定义主函数
if __name__ == '__main__':
    
    # 初始化节点
    rospy.init_node('desnow_node', anonymous=True)
   
    # 加载参数
    args = parse_args()

    # 加载模型
    print('Building Model...')
    model = model.model.build_DTCWT_model((512,672,3))
    model.load_weights('./modelParam/finalmodel.h5',by_name=False)
    graph = tf.get_default_graph()

    # 定义对象
    desnow = Desnow()

    # 及时输出队列内容
    rospy.spin()

