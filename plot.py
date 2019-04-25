import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


path = r'VGG16\with_augmentation'
filename = 'history_128.pkl'

with open(os.path.join(path,filename),"rb") as f:
    history_dict = pickle.load(f)

print (history_dict)
# #Loss curve

fig,ax = plt.subplots()
plt.plot(list(range(len(history_dict['loss']))), history_dict['loss'], color = 'b', label = 'Training Loss')
plt.plot(list(range(len(history_dict['val_loss']))), history_dict['val_loss'], color = 'r', label = 'Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Loss Curve for Pretrained Inception V3 without data augmentation')
fig.savefig(os.path.join(path,'loss_curve_wa_vgg16.png'))
plt.close()

#Acc curve

fig,ax = plt.subplots()
plt.plot(list(range(len(history_dict['f_score']))), history_dict['f_score'], color = 'b', label = 'Training F1 Score')
plt.plot(list(range(len(history_dict['val_f_score']))), history_dict['val_f_score'], color = 'r', label = 'Validation F1 Score')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('F1 Score (with threshold 0.2)')
plt.title('F1 Score Curve for Pretrained Inception V3 without data augmentation')
fig.savefig(os.path.join(path,'f1_curve_wa_vgg16.png'))
plt.close()
