import matplotlib.pyplot as plt
from sklearn import metrics


def report(model_path,fold):
  model = tf.keras.models.load_model(model_path)
  y_true = []
  outputs = []
  for x in fold['test']:
    y_true.append(labels[str(x)])
    output = model([np.load('/content/drive/MyDrive/knowledge_g1020_best/'+str(x[:-4])+'.npy'),np.load('/content/drive/MyDrive/knowledge2_g1020_best/'+str(x[:-4])+'.npy')])
    outputs.append(np.argmax(output,axis = -1))
  print(classification_report(y_true,outputs))



plt.figure(0).clf()
plt.rcParams['figure.figsize'] = [7, 5]
plt.rcParams['figure.dpi'] = 100
for i,mod_type in enumerate(model_list):
  model = tf.keras.models.load_model('/content/drive/MyDrive/'+mod_type)
  y_true = []
  outputs = []
  know = []
  know2 = []
  for x in folds[4]['validation']:
    y_true.append(labels[str(x)])
    k = np.load('/content/drive/MyDrive/knowledge_g1020_best/'+str(x[:-4])+'.npy',allow_pickle= True)
    k2= np.load('/content/drive/MyDrive/knowledge2_g1020_best/'+str(x[:-4])+'.npy',allow_pickle= True)
    outputs.append(np.argmax(model([k,k2]),axis = -1))
  fpr, tpr, _ = metrics.roc_curve(y_true, outputs)
  auc = round(metrics.roc_auc_score(y_true, outputs), 4)
  plt.plot(fpr,tpr,label=lab[i]+", AUC="+str(auc),color='black',linewidth=3)


for x in folds[4]['validation']:
    k = np.load('/content/drive/MyDrive/knowledge_g1020_best/'+str(x[:-4])+'.npy',allow_pickle= True)
    k2= np.load('/content/drive/MyDrive/knowledge2_g1020_best/'+str(x[:-4])+'.npy',allow_pickle= True)
    know.append(k)
    know2.append(k2)
fpr, tpr, _ = metrics.roc_curve(y_true, np.argmax(know,axis = -1))
auc = round(metrics.roc_auc_score(y_true, np.argmax(know,axis = -1)), 4)
plt.plot(fpr,tpr,label="Inceptionv3"+", AUC="+str(auc),color='red')  
fpr, tpr, _ = metrics.roc_curve(y_true, np.argmax(know2,axis = -1))
auc = round(metrics.roc_auc_score(y_true, np.argmax(know2,axis = -1)), 4)
plt.plot(fpr,tpr,label="Densenet"+", AUC="+str(auc)) 

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc= "best")



import numpy as np
import matplotlib.pyplot as plt
data = [[0.76,0.36],
[0.36, 0.47],
[0.43,0.45]]

X = np.arange(2)
fig = plt.figure()
plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams['figure.dpi'] = 100
ax = fig.add_axes([0,0,1,1])
ax.set_ylim([0, 1])
ax.bar(X + 0.00, data[0], color = 'r', width = 0.25, label = "Densenet (Teacher)")
ax.bar(X + 0.25, data[1], color = 'b', width = 0.25, label = "Inceptionv3 (Student)")
ax.bar(X + 0.50, data[2], color = 'lawngreen', width = 0.25, label = "Knowledge Incorporated Model")
ax.axes.xaxis.set_ticklabels([])
plt.tick_params(bottom=False)
plt.xlabel('Class0                                                 class1')
plt.ylabel('F1 Score')
plt.legend(loc= "best")

report('drive/MyDrive/G1020_knowledge_incorporated_fold4(dense+inception)_logits_denseteacher',folds[4])