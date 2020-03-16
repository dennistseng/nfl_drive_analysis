# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:10:33 2020

@author: halfc
"""


from sklearn.model_selection import KFold
from imblearn.pipeline import make_pipeline as make_pipeline_imb
'''


'''
# Cross Validation function
def cv_model(data_np, target_np, imb = NearMiss(version=1), classifier = DecisionTreeClassifier(), k = 5):
    kf = KFold(n_splits= k)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for train, test in kf.split(data_np, target_np):
        pipeline = make_pipeline_imb(imb, classifier)
        model = pipeline.fit(data_np.iloc[train], target_np.iloc[train])
        prediction = model.predict(data_np.iloc[test])
    
        accuracy.append(pipeline.score(data_np.iloc[test], target_np.iloc[test]))
        precision.append(precision_score(target_np.iloc[test], prediction, average='micro'))
        recall.append(recall_score(target_np.iloc[test], prediction, average='micro'))
        f1.append(f1_score(target_np.iloc[test], prediction, average='micro'))
    
    print()
    print("Training Scores 5-fold:")
    print("accuracy: %0.2f (+/- %0.3f)" % (np.mean(accuracy), np.std(accuracy)*2))
    print("precision: %0.2f (+/- %0.3f)" % (np.mean(precision), np.std(precision)*2))
    print("recall: %0.2f (+/- %0.3f)" % (np.mean(recall), np.std(recall)*2))
    print("f1: %0.2f (+/- %0.3f)" % (np.mean(f1), np.std(f1)*2))


print('Decision Tree Results')
start_ts=time.time()
cv_model(data_np, target_np)
print("CV Runtime:", time.time()-start_ts)

print()
print('Random Forest Results')
start_ts=time.time()
cv_model(data_np, target_np, imb = NearMiss(version=1), classifier = RandomForestClassifier(), k = 5)
print("CV Runtime:", time.time()-start_ts)

'''