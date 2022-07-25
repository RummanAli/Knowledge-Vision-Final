def get_loader():
    df = pd.read_csv('/content/drive/MyDrive/g1020-polygons/G1020.csv')
    partition = {'train': [],'validation' : []}
    for i in range(900):
        partition['train'].append(df['imageID'][i])
    for i in range(900,1020):
        partition['validation'].append(df['imageID'][i])
    labels = dict((val, out ) for val,out in zip(df['imageID'],df['binaryLabels']))

    labels2 = dict((val, out ) for val,out in zip(df['patientID'],df['binaryLabels']))

    patient_labels = []
    for x in df['patientID'].unique():
        patient_labels.append(labels2[x])

    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    skf = StratifiedKFold(n_splits=6, random_state=42, shuffle=True)
    train_folds = []
    val_folds = []
    for train_index, val_index in skf.split(df['patientID'].unique(), patient_labels):
        train_folds.append(train_index)
        val_folds.append(val_index)
    

    folds = []
    for train_fold,val_fold in zip(train_folds,val_folds):
        fold = {'train': [],'validation' : []}
        for x in train_fold:
            a = df['patientID'].unique()[x]
            index = df.index[df['patientID']== a].tolist()
            for img in index:
                fold['train'].append(df['imageID'][img])

    for y in val_fold:
        a = df['patientID'].unique()[y]
        index = df.index[df['patientID']== a].tolist()
        for img in index:
            fold['validation'].append(df['imageID'][img])
    folds.append(fold)
    return folds,labels
