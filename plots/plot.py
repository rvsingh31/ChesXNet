import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


with_path = r'IV3\with_augmentation'
filename = 'history_128.pkl'

without_path = r'IV3\without_augmentation'
with open(os.path.join(with_path,filename),"rb") as f:
    with_history_dict = pickle.load(f)

with open(os.path.join(without_path,filename),"rb") as f:
    without_history_dict = pickle.load(f)

#Loss curve

fig,ax = plt.subplots()
plt.plot(list(range(len(with_history_dict['loss']))), with_history_dict['loss'], linestyle = '-', color = 'b', label = 'Training Loss (With Augmentation)')
plt.plot(list(range(len(with_history_dict['val_loss']))), with_history_dict['val_loss'], linestyle = '-', color = 'r', label = 'Validation Loss (With Augmentation)')

plt.plot(list(range(len(without_history_dict['loss']))), without_history_dict['loss'], linestyle = ':', color = 'b', label = 'Training Loss (Without Augmentation)')
plt.plot(list(range(len(without_history_dict['val_loss']))), without_history_dict['val_loss'], linestyle = ':', color = 'r', label = 'Validation Loss (Without Augmentation)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Loss Curve for Pretrained Inception V3')
fig.savefig('loss_curve_iv3.png')
plt.close()

#Acc curve

fig,ax = plt.subplots()
plt.plot(list(range(len(with_history_dict['f_score']))), with_history_dict['f_score'],  linestyle = '-', color = 'b', label = 'Training F1 Score (With Augmentation)')
plt.plot(list(range(len(with_history_dict['val_f_score']))), with_history_dict['val_f_score'],  linestyle = '-', color = 'r', label = 'Validation F1 Score (With Augmentation)')

plt.plot(list(range(len(without_history_dict['f_score']))), without_history_dict['f_score'],  linestyle = ':', color = 'b', label = 'Training F1 Score (Without Augmentation)')
plt.plot(list(range(len(without_history_dict['val_f_score']))), without_history_dict['val_f_score'],  linestyle = ':', color = 'r', label = 'Validation F1 Score (Without Augmentation)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('F1 Score (with threshold 0.2)')
plt.title('F1 Score Curve for Pretrained Inception V3')
fig.savefig('f1_curve_iv3.png')
plt.close()
