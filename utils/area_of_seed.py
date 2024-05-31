import cv2
import os


dirs = ["Acroptilon repens", "Ambrosia artemisiifolia", "Ambrosia tridida", "Convolvulus arvensis", "Delphinium consolida",
        "Galium aparine", "Onopordum acanthium", "Polygonum convolvulus", "Silene latifolia", "Sinapis arvensis",
        "Turgenia latifolia", "Wheat"]

def count_all_areas():
    for dir in dirs:
        for file in os.listdir(dir + "/mask"):
            photo = cv2.imread(dir + "/mask/" + file, cv2.IMREAD_GRAYSCALE)
            non_black = cv2.countNonZero(photo)

            path = dir + "/areas/"
            if not os.path.exists(path):
                os.mkdir(path)
            with open(path + file[:-4] + ".txt", 'w') as f:
                f.write(f"in pixels {non_black}\n")
                f.write(f"in mm^2 {(non_black) / 784}\n")

def count_mean_areas():
    for dir in dirs:
        n = 0
        sum = 0
        for file in os.listdir(dir + "/areas"):
            n += 1
            mm = 0
            with open(dir + "/areas/" + file) as f:
                for i, line in enumerate(f):
                    if i == 1:
                        mm = float(line[8:])
            sum += mm
        mean = sum/n
        with open(dir + "/mean_area.txt", 'w') as f:
            f.write(f"in mm^2 {mean}\n")

count_mean_areas()
