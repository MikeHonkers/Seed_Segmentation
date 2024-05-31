from sklearn.metrics import recall_score
from tqdm import tqdm
import cv2
for n in range(100):
   y_true = cv2.imread(f"/scratch/mpavlov-U-Net/data/Masks/test/{n}.png", 0)
   y_pred = cv2.imread(f"/scratch/mpavlov-U-Net/pred_masks/{n}.png", 0)
   labels = [100, 132, 193, 108, 101, 63, 152, 215, 124, 23, 104, 231]
   true_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   false_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   for i in tqdm(range (1728)):
      for j in range (2592):
         if ((y_pred[i][j] in labels) and (y_true[i][j] == y_pred[i][j])):
            true_pos[labels.index(y_pred[i][j])] += 1
         elif ((y_true[i][j] in labels) and (y_true[i][j] != y_pred[i][j])):
            false_neg[labels.index(y_true[i][j])] += 1
   print(f"done{n+1}/100")
for i in range(12):
   print(true_pos[i]/(true_pos[i]+false_neg[i]))
