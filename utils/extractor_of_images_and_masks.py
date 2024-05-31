import cv2
import numpy as np
import os

def save_images(blackAndWhiteImage3, dst, real_image, up, down, left, right, c, num):
    cv2.imwrite(f'C:/Users/Michael/Desktop/SEEDS/my_code/output1/mask/seed_mask_{num}_{c}.png', blackAndWhiteImage3[up:down+1, left:right+1])
    cv2.imwrite(f'C:/Users/Michael/Desktop/SEEDS/my_code/output1/trans/seed_transp_{num}_{c}.png', dst)
    cv2.imwrite(f'C:/Users/Michael/Desktop/SEEDS/my_code/output1/img/seed_{num}_{c}.png', real_image[up:down+1, left:right+1])
    #if not cv2.imwrite(f'C:\\Users\Michael\Desktop\SEEDS\my_code\output1\mask\seed_mask_{c}.png', blackAndWhiteImage3[up:down, left:right]):
        #raise Exception("Could not write image")

def create_images(real_image, up, down, left, right, blackAndWhiteImage, c, num):
    # Для transparent img
    b, g, r = cv2.split(real_image[up:down+1, left:right+1])
    rgba = [b, g, r, blackAndWhiteImage[up:down+1, left:right+1]]
    dst = cv2.merge(rgba, 4)

    blackAndWhiteImage3 = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2BGR)
    blackAndWhiteImage3[blackAndWhiteImage == 255] = (100, 100, 100)

    save_images(blackAndWhiteImage3, dst, real_image, up, down, left, right, c, num)

def find_seed_bounds_by_contours(contours, hierarchy, image, real_image, blackAndWhiteImage, num):
    height, width, depth = image.shape
    for c in range(len(contours)):
        blank = np.zeros((image.shape), np.uint8)
        cv2.drawContours(blank, [contours[c]], -1, (255, 255, 255), -1)

        left = width
        right = 0
        up = height
        down = 0
        flag = 0
        if (blank[1026][1400][0] == 255):
            print(f"{c} is bad")
            continue
        for i in range(height):
            if (255, 255, 255) in blank[i]:
                for j in range(width):
                    if (blank[i][j][0] == 255):
                        if (up == height and blank[i][j][0] == 255):
                            up = i
                        if (i > down and blank[i][j][0] == 255):
                            down = i
                        if (j < left):
                            left = j
                        if (j > right):
                            right = j
                        if (255, 255, 255) not in blank[i + 1]:
                            flag = 1
                            break
                if (flag == 1):
                    break
        create_images(real_image, up, down, left, right, blackAndWhiteImage, c, num)
        print(f"записано {c + 1}/{len(contours)} картинок")


counter = 0
for i in (os.walk("D:\my_dir\jsons\Bodyak_polevoy")):
    if (len(i[2]) == 0):
        continue
    image = cv2.imread(i[0] + '\\' + i[2][3])
    #исходное изображение
    real_image = cv2.imread(i[0] + '\\' + i[2][0])
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 73, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (960, 540))
    #cv2.imshow("lol", blackAndWhiteImage)
    #cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    find_seed_bounds_by_contours(contours, hierarchy, image, real_image, blackAndWhiteImage, counter)
    counter += 1
#for i in (os.walk("D:\my_dir\jsons\Амброзия полынолистая")):
#    print(i)