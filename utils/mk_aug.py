import os
import numpy as np
import cv2
from PIL import Image
import random
from tqdm import tqdm

def convert_bboxes(bboxes, height, width):
    new_bboxes = []
    for box in bboxes:
        new_bboxes.append((box[0], ((box[2]+box[1])/2)/width, ((box[4]+box[3])/2)/height,  (box[2]-box[1])/width, (box[4] - box[3])/height))
    return new_bboxes

def check_availability(x, y, seed_width, seed_height, bboxes):
    x1_s = x
    y1_s = y
    x2_s = x1_s + seed_width
    y2_s = y1_s + seed_height
    i = 0
    for box in bboxes:
        if i == 0:
            i += 1
            continue
        x1_b = int(box[1])
        y1_b = int(box[3])
        x2_b = int(box[2])
        y2_b = int(box[4])

        if (x2_b < x1_s or x1_b > x2_s or y1_b > y2_s or y2_b < y1_s):
            continue
        else:
            return False
    return True

def add_seed(key, colour, canvas, canvas_mask, seed, seed_mask, bboxes, weed_dir, allow_overlays):
    if not allow_overlays:
        #print(seed)
        seed = Image.open(weed_dir + '/' + str(key) + '/fimg' + '/' + seed)
        seed_mask = cv2.cvtColor(seed_mask, cv2.COLOR_BGR2RGB)
        seed_mask = Image.fromarray(seed_mask)
        angle = random.randint(0, 360)
        seed = seed.rotate(angle, expand=True)
        seed_mask = seed_mask.rotate(angle, expand=True)
        seed_width, seed_height = seed.size
        canvas_width, canvas_height = canvas.size
        checker_width = int(bboxes[0][2])
        x = 0
        y = 0
        while True:
            x = random.randint(checker_width + 10, canvas_width - seed_width - 10)
            y = random.randint(10, canvas_height - seed_width - 10)
            if not check_availability(x, y, seed_width, seed_height, bboxes):
                continue
            try:
                seed_mask = np.array(seed_mask)
                seed_mask = cv2.cvtColor(seed_mask, cv2.COLOR_RGB2BGR)
                seed_mask[np.all(seed_mask == (100, 100, 100), axis=-1)] = colour
                seed_mask = cv2.cvtColor(seed_mask, cv2.COLOR_BGR2RGB)
                seed_mask = Image.fromarray(seed_mask)
                canvas.paste(seed, (x, y), seed)
                canvas_mask.paste(seed_mask, (x, y), seed)
            except Exception as error:
                seed_mask.show()
                print(key)
                continue
            break

        return canvas, canvas_mask, tuple([key, x, x + seed_width, y, y + seed_height])

def add_checker(canvas, canvas_mask, colorchecker, col_check_mask):
    canv_height, canv_width, _ = canvas.shape
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    col_check = Image.open(colorchecker + '/fimg/1.png')
    col_check_width, col_check_height = col_check.size
    canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_BGR2RGB)
    canvas_mask = Image.fromarray(canvas_mask)
    col_check_mask = cv2.cvtColor(col_check_mask, cv2.COLOR_BGR2RGB)
    col_check_mask = Image.fromarray(col_check_mask)

    x_left = random.randint(0, int(0.03 * canv_width))
    y_left = random.randint(0, int(0.06 * canv_height))
    canvas.paste(col_check, (x_left, y_left), col_check)
    canvas_mask.paste(col_check_mask, (x_left, y_left), col_check)
    return canvas, canvas_mask, tuple([0, x_left, x_left + col_check_width, y_left, y_left + col_check_height])

def generate_image(mask_classes_colours, img_num, background, colorchecker, weed_dir, allow_overlays, N_weed,
                   N_spec):
    s = 0
    for i in tqdm(range(img_num)):
        bboxes = []  # (class, x1, x2, y1, y2)
        canvas = cv2.imread(background)
        canvas_mask = np.zeros((np.array(canvas).shape[0], np.array(canvas).shape[1], 3), dtype=np.uint8)
        col_check_mask = cv2.imread(colorchecker + '/mask/1.png')
        canvas, canvas_mask, x = add_checker(canvas, canvas_mask, colorchecker, col_check_mask)
        bboxes.append(x)
        for key in N_spec:
            num = random.randint(N_spec[key][0], N_spec[key][1])
            seed_imgs = os.listdir(weed_dir + '/' + key + '/fimg')
            for i in range(num):
                seed = random.choice(seed_imgs)
                seed_mask = cv2.imread(weed_dir + '/' + key + '/mask' + '/seed_mask' + seed[11:]) #change this slice depending on your file names
                colour = mask_classes_colours[int(key)]
                canvas, canvas_mask, x = add_seed(key, colour, canvas, canvas_mask, seed, seed_mask, bboxes,
                                                  weed_dir, allow_overlays)
                bboxes.append(x)

        num = random.randint(N_weed[0], N_weed[1])
        for i in tqdm(range(num)):
            if (list(set(mask_classes_colours.keys()) - set(int(dummy) for dummy in N_spec))):
                key = random.choice(list(set(mask_classes_colours.keys()) - set(int(dummy) for dummy in N_spec)))
                seed_imgs = os.listdir(weed_dir + '/' + str(key) + '/fimg')
                seed = random.choice(seed_imgs)
                #print(weed_dir + '/' + str(key) + '/mask' + '/seed_mask' + seed[11:])
                seed_mask = cv2.imread(weed_dir + '/' + str(key) + '/mask' + '/seed_mask' + seed[11:]) #change this slice depending on your file names
                colour = mask_classes_colours[key]
                canvas, canvas_mask, x = add_seed(key, colour, canvas, canvas_mask, seed, seed_mask, bboxes, weed_dir,
                                                  allow_overlays)
                bboxes.append(x)

        canvas = np.array(canvas)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas_mask = np.array(canvas_mask)
        canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_RGB2BGR)
        height, width, _ = canvas.shape
        bboxes = convert_bboxes(bboxes, height, width)
        with open(f'D:/my_dir/ULTRAL/yolo_data/labels/test/{s}.txt', 'w') as f:
            f.write('\n'.join(f'{tup[0]} {tup[1]} {tup[2]} {tup[3]} {tup[4]}' for tup in bboxes))
        cv2.imwrite(f'D:/my_dir/ULTRAL/yolo_data/images/test/{s}.png', canvas)
        cv2.imwrite(f'D:/my_dir/ULTRAL/yolo_data/masks/test/{s}.png', canvas_mask)
        s += 1

if __name__ == "__main__":
    mask_classes_colours = {1:(153,0,153), 2:(51, 255, 255), 3:(204,0,0), 4:(75, 176, 136), 5:(201, 202, 247),
                            7:(81, 82, 149), 8:(167, 101, 181), 9:(119, 155, 0),
                            10:(36, 65, 221), 11:(118, 80, 214), 12:(80, 192, 239)} # num of class: (colour bgr)
    img_num = 100
    background = "D:/my_dir/Background.png"
    colorchecker = "D:/my_dir/Color_checker"
    weed_dir = "D:/my_dir/Weeds_original" #names of dirs of classes should be like 1, 2 ... 0 class is reserved for color_checker
    allow_overlays = False #currently supports only non overlaying seeds
    N_weed = (30, 40) #range of all weeds on img, not including N_spec
    N_spec = {}
    #N_spec = {"1":(1,1), "2":(1,1), "3":(1,1), "4":(1,1), "5":(1,1), "7":(1,1), "8":(1,1), "9":(1,1), "10":(1,1),
    #          "11":(1,1), "12":(1,1)} #dict, looks like "weed_class" : (tuple of range), like N_weed. Used to specify certaint class amount
    generate_image(mask_classes_colours, img_num, background, colorchecker, weed_dir, allow_overlays, N_weed, N_spec)