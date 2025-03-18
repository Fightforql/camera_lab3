import json
import cv2
import numpy as np
from task1 import task,de_distortion,Reprojection_Error,part_4

def main():
        print("第一部分—————————————————————")
        #从配置文件读取参数
        with open('config.json', 'r') as file:
                config = json.load(file)

        global_num = config['global_num']
        size = config['size']
        mysize=config['mysize']
        square_size = config['square_size']
        mtx,dist,_,__,___,____=task(global_num,size,square_size,'set1')
        print("*********分界线**********")
        print("第一部分之去畸变————————————————————")
        de_distortion(mtx,dist)
        print("第二部分—————————————————————")
        mtx,dist,rvecs,tvecs,objectpoints,imagepoints_b=task(global_num,mysize,square_size,'set2')
        print("第三部分—————————————————————")
        total_error=0
        for i in range(len(rvecs)):
               #print(f"第{i+1}张图片反投影")
               imagepoints_a=cv2.projectPoints(objectpoints,rvecs[i],tvecs[i],mtx,dist)
               imagepoints_b = np.array(imagepoints_b, dtype=np.float32)
               p_a=imagepoints_a[0].reshape(-1,2)
               p_b=imagepoints_b[i].reshape(-1,2)
               print(p_a)
               print(p_b)
               total_error+=Reprojection_Error(p_a,p_b,48) #计算重投影误差
        print(total_error/len(rvecs))
        print("第四部分—————————————————————")
        part_4("ar_pic.jpg","myimage/myImage1.jpg",(8,6))
        part_4("ar_pic.jpg","myimage/myImage12.jpg",(8,6))
        part_4("ar_pic.jpg","myimage/myImage14.jpg",(8,6))
        

               

        

if __name__ == "__main__":
    main()