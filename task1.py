import cv2
import numpy as np
from typing import List


def readimage(global_num:int,set:str)-> List[np.ndarray]:
        images=[]
        for i in range(1,global_num):
                if set=='set1':
                       filename = f"Image{i}.tif"
                else: 
                       filename = f"myimage/myImage{i}.jpg"  
                img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
                if img is None:
                        print(f"Failed to load {filename}")
                else:
                        print(f"Successfully loaded {filename}")
                        if set=='set2':
                                img_resized = cv2.resize(img, (800,600))
                                images.append(img_resized)
                        else:
                               images.append(img)
        return images

def FindandDraw_Corners(images:List[np.ndarray],global_num:int,size:tuple,square_size:int,set:str):
        imgpoints = []
        objectpoints=[]
        objp = np.zeros((1, size[0] * size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)*square_size
        #print(objp)
        for i in range(1,global_num):
                img=images[i-1]
                if set=='set1':
                       filename = f"Image{i}.tif" 
                else:
                       filename = f"myimage/myImage{i}.jpg"
                ret,corners=cv2.findChessboardCorners(img,size)
                if ret !=0:
                        #print(corners)
                        imgpoints.append(corners)
                        objectpoints.append(objp)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        print(f"Successfully found {filename}'s corners")
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        cv2.drawChessboardCorners(img, size, corners2, ret)
                        cv2.imshow(winname="image title", mat=img)
                        cv2.waitKey(700)
                else:
                        print(f"Failed to find {filename}'s corners")
        return imgpoints,objectpoints,gray.shape,objp



def task(global_num:int,size:tuple,square_size:int,set:str):
        images=readimage(global_num,set)
        imgpoints,objectpoints,img_size,_=FindandDraw_Corners(images,global_num,size,square_size,set)
        ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objectpoints,imgpoints,img_size,None,None)
        print("Ret:", ret)
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)
        print("Rotation Vectors:\n", rvecs)
        print("Translation Vectors:\n", tvecs)
        #得到每个旋转向量的旋转矩阵
        Rotation_matrix=[]
        for i, rvec in enumerate(rvecs):
               rotation_matrix, __ = cv2.Rodrigues(rvec)
               Rotation_matrix.append(rotation_matrix)
        print("Rotation Matrix:\n",Rotation_matrix)
        return mtx,dist,rvecs,tvecs,_,imgpoints

def de_distortion(mtx,dist):
        print("选取image7")
        cur_img=cv2.imread("Image7.tif", flags=cv2.IMREAD_COLOR)
        dst = cv2.undistort(cur_img, mtx, dist, dst=None, newCameraMatrix=None)
        if dst is None:
                print("Error: Undistortion failed.")
                return
        else:
                cv2.imshow('Distorted Image', cur_img)
                cv2.imshow('Undistorted Image', dst)
                cv2.waitKey(700)

def Reprojection_Error(imagepoints_a,imagepoints_b,length):
        total_error=0
        for i in range(length):
                error = cv2.norm(imagepoints_a[i], imagepoints_b[i], cv2.NORM_L2) / length
                total_error+=error
        return total_error





def compute_matrix(corner_points,image_points):
        A=[]
        for i in range(4):
                row1=[image_points[i][0],image_points[i][1],1,0,0,0,-image_points[i][0]*corner_points[i][0],-image_points[i][1]*corner_points[i][0],-corner_points[i][0]]
                row2=[0,0,0,image_points[i][0],image_points[i][1],1,-image_points[i][0]*corner_points[i][1],-image_points[i][1]*corner_points[i][1],-corner_points[i][1]]
                A.append(row1)
                A.append(row2)
        A=np.array(A)
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        H=h.reshape(3,3)
        H = H / H[2, 2]
        return H
def my_warpPerspective(matrix,img,size):
        width, height = size
        warped_img = np.zeros((height, width, 3), dtype=np.uint8)

        H_inv = np.linalg.inv(matrix)

        for y in range(height):
                for x in range(width):
                        src_pt = np.array([x, y, 1]).reshape(3, 1)
                        src_pt = H_inv @ src_pt
                        x_src, y_src = src_pt[0, 0] / src_pt[2, 0], src_pt[1, 0] / src_pt[2, 0]

                        # 进行双线性插值
                        if 0 <= x_src < img.shape[1] - 1 and 0 <= y_src < img.shape[0] - 1:
                                x0, y0 = int(x_src), int(y_src)
                                x1, y1 = x0 + 1, y0 + 1

                                dx, dy = x_src - x0, y_src - y0

                                # 4个相邻点的像素值
                                I00 = img[y0, x0]
                                I01 = img[y0, x1]
                                I10 = img[y1, x0]
                                I11 = img[y1, x1]

                                # 双线性插值计算
                                warped_img[y, x] = (
                                I00 * (1 - dx) * (1 - dy) +
                                I01 * dx * (1 - dy) +
                                I10 * (1 - dx) * dy +
                                I11 * dx * dy
                                ).astype(np.uint8)

        return warped_img

def part_4(src_file,dst_file,chessboard_size):
        chessboard=cv2.imread(dst_file)
        chessboard = cv2.resize(chessboard, (800,600))
        ret,corners=cv2.findChessboardCorners(chessboard,chessboard_size)
        #由于我拍的图片均可以找到角点 所以这里不做判断
        print(corners)
        dst_points=[corners[0],corners[chessboard_size[0]-1],corners[-chessboard_size[0]],corners[-1]]
        dst_points = [tuple(pt[0]) for pt in dst_points]
        src_img=cv2.imread(src_file)
        print(src_img.shape)
        src_points=[(0,0),(src_img.shape[1],0),(0,src_img.shape[0]),(src_img.shape[1],src_img.shape[0])]
        #计算单应矩阵，这里利用svd分解计算，就可以实现较好的效果
        H=compute_matrix(dst_points,src_points)
        print(H)
        print(dst_points)
        chessboard_height, chessboard_width = chessboard.shape[:2]

        warped_img = my_warpPerspective(H, src_img,(chessboard_width, chessboard_height))

        # 创建一个蒙版，用于合成变换后的图像
        gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        _, mask_warped = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

        result = chessboard.copy()
        result[mask_warped > 0] = warped_img[mask_warped > 0]

        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
