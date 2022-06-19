import pandas as pd
import numpy as np
from collections import Counter
import statistics
import sklearn.naive_bayes as nb
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def data_mean_fn(features, class_instances):
    feature_vector_length=len(features[0])
    instances = len(features)
    attribute = 0
    mean_dict = []
    class_instances=[value[1] for value in class_instances]
    while attribute < feature_vector_length:
        class_dict={}
        count_instances = 0
        for classes in set(class_instances):
            class_dict[classes]=[]
        while count_instances < instances:
            class_dict[class_instances[count_instances]].append(int(features[count_instances][attribute]))
            count_instances +=1
        for classes in set(class_instances):
            class_dict[classes]=statistics.mean(class_dict[classes])
        mean_dict.append(class_dict)
        attribute +=1
    return(mean_dict)

def data_variance_fn(features, class_instances):
    feature_vector_length=len(features[0])
    instances = len(features)
    attribute = 0
    variance_dict = []
    class_instances=[value[1] for value in class_instances]
    while attribute < feature_vector_length:
        class_dict={}
        count_instances = 0
        for classes in set(class_instances):
            class_dict[classes]=[]
        while count_instances < instances:
            class_dict[class_instances[count_instances]].append(int(features[count_instances][attribute]))
            count_instances +=1
        for classes in set(class_instances):
            class_dict[classes]=statistics.variance(class_dict[classes])
        variance_dict.append(class_dict)
        attribute +=1
    return(variance_dict)

def prior(class_instances):
    class_instances = [value[1] for value in class_instances]
    class_prior = {}
    class_counter = Counter(class_instances)
    for value in class_counter.most_common():
        class_prior[value[0]]=value[1]/len(class_instances)
    return(class_prior)

def preprocess(filename):
    features=[]
    class_instances=[]
    with open(filename,'r') as f:
        headers=f.readline().strip().split(',')
        header_attributes = headers[2:]
        class_attributes=headers[:2]
        for instance in f:
            data=instance.strip().split(',')
            class_instances.append(data[:2])
            data_int=[float(i) for i in data[2:]]
            features.append(data_int)
        class_counter=Counter([i[1] for i in class_instances])
        '''
        attributes = [[] for index in range(len(features[0]))]
        attribute_counter = 0
        while attribute_counter < len(features[0]):
            data_instances = []
            instance_counter = 0
            while instance_counter < len(features):
                data_instances.append(features[instance_counter][attribute_counter])
                instance_counter+=1
            attributes[attribute_counter]=data_instances
            attribute_counter+=1
        '''
    return(features,class_instances)

def train(features, class_instances):
    data_mean = data_mean_fn(features, class_instances)
    data_variance = data_variance_fn(features,class_instances)
    class_prior = prior(class_instances)
    return(data_mean,data_variance,class_prior)

def predict(test_features,mean,variance,prior,total_length):
    if any(isinstance(x, list) for x in test_features):
        print("predict function only takes single instance, A list of multiple instances received ")
        return
    prediction_temp = {}
    for classes in prior.items():
        prediction_temp[classes[0]]=[np.log(classes[1])]
        for col_index, attributes in enumerate(test_features):
            mean_attribute = mean[col_index][classes[0]]
            variance_attribute=variance[col_index][classes[0]]
            gaussian_pdf = (np.e**(-0.5*(((int(attributes)-mean_attribute)**2)/variance_attribute))) / (np.sqrt(2 * np.pi*variance_attribute))
            if gaussian_pdf != 0:
                prediction_temp[classes[0]].append(np.log(gaussian_pdf))
            else:
                gaussian_pdf = 1/(10*total_length)
                prediction_temp[classes[0]].append(gaussian_pdf)
        prediction_temp[classes[0]]=sum(prediction_temp[classes[0]])
    prediction = max(prediction_temp, key=prediction_temp.get)
    return(prediction)


def evaluate(actual_classes, prediction):
    score = 0
    for i, j in zip(actual_classes, prediction):
        if i == j:
            score += 1
    accuracy_score = score/len(actual_classes)
    actual_classes= np.array(actual_classes)
    prediction=np.array(prediction)
    report = f1_score(actual_classes,prediction,average='macro')
    return(accuracy_score,report)

def Accuracy(y, prediction):
    y = list(y)
    prediction = list(prediction)
    score = 0
    for i, j in zip(y, prediction):
        if i == j:
            score += 1
    return score / len(y)

features,class_instances = preprocess('../2022S1-a2-data/2022S1-a2-data/adult.csv')

data_mean, data_variance, class_prior = train(features, class_instances)
predicted_classes =[]
for values in features:
    predicted_classes.append(predict(values,data_mean, data_variance, class_prior, len(features)))
actual_classes = [value[1] for value in class_instances]
print('my model is', evaluate(actual_classes, predicted_classes))


gnb = GaussianNB()
actual_classes = [value[1] for value in class_instances]

set_actual_classes=list(set(actual_classes))
actual_classes = [set_actual_classes.index(value[1]) for value in class_instances]
gnb.fit(np.array(features),np.array(actual_classes))
print("sklean implementation score is", gnb.score(np.array(features),np.array(actual_classes)))

exit()
print('class_instances are', class_instances[-10:])

print('mean is ', data_mean[0], 'and the variance is ', data_variance[0])
print('mean is ', data_mean[1], 'and the variance is ', data_variance[1])
print('mean is ', data_mean[2], 'and the variance is ', data_variance[2])


print("Feature vectors of instances [0, 1, 2]: ", features[0:2])

print("\nNumber of instances (N): ", len(features))
print("Number of features (F): ", len(features[0]))
print("Number of labels (L): ", len(Counter([i[1] for i in class_instances])))

print("\n\nPredicted class probabilities for instance N-3: ", )
print("Predicted class ID for instance N-3: ", )
print("\nPredicted class probabilities for instance N-2: ", )
print("Predicted class ID for instance N-2: ", )
print("\nPredicted class probabilities for instance N-1: ", )
print("Predicted class ID for instance N-1: ", )