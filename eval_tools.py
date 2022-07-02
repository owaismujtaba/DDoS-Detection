from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch
from tqdm import tqdm
import warnings
import sys


def evaluate_model(pred, labels):
   
    accuracy = metrics.accuracy_score(pred, labels)
    precision = metrics.precision_score(pred, labels)
    recall = metrics.recall_score(pred, labels)
    f1 = metrics.f1_score(pred, labels)
    roc_auc = metrics.roc_auc_score(pred, labels)
    
    print("Accuracy {}, Precision {}, Recall {}, f1 {}, roc_auc {}".format(accuracy, precision, recall, f1, roc_auc))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_report(outputs, labels):

    '''
    report is in batches inorder to save memory usage.
    The function gives various matrics performance of the model
    :param predictions: predictions by the model
    :param labels: original labels
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        predictions = torch.max(outputs, 1)[1].to(device)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = accuracy_score(predictions, labels)
        recall = recall_score(predictions, labels)
        f1 = f1_score(predictions, labels)
        cm = confusion_matrix(predictions, labels)

        print("Accuracy : {}, Recall : {}, F1 Score : {}".format(accuracy, recall, f1))
        print("************************* Confusion Matrix ***********************")
        print(cm)


def evaluate_proposed_model(model, test_loader):
    '''
    :param cnn_model: trained model
    :param test_loader: test dataset loader
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        count = 1
        with torch.no_grad():
            correct = 0
            total_samples = 0
            
            test_bar = tqdm(test_loader, file=sys.stdout)

            for step, (samples, labels) in enumerate(test_bar):
                samples, labels = samples.to(device), labels.to(device)

                output = model(samples)

                print("Report on BATCH: ", count)
                classification_report(output, labels)
                count +=1
            
            
def accuracy(outputs, labels):
    '''
    :param outputs: predictions from the model
    :param labels: original labels
    :return: correct: total number of correct classifications
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        predictions = torch.max(outputs, 1)[1].to(device)
        correct = (predictions == labels).sum().to(device)

        acc = correct/len(predictions)
        acc = acc.cpu().numpy()
        return acc
