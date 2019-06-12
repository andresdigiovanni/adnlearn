import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(X_test, y_test, model, labels):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(cm, index = [i for i in labels],
                      columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_feature_importances(clf, columns, top_n=10, figsize=(8,8), print_table=True, title="Feature Importances"):

    try:
        feat_imp = pd.DataFrame([clf.feature_importances_],
                                columns = columns)
        feat_imp = feat_imp.sort_values(by=0, ascending=False, axis=1)
        feat_imp = feat_imp.iloc[:top_n]
    
        feat_imp.plot.barh(title=title, figsize=figsize)
        plt.xlabel('Feature Importance Score')
        plt.draw()
    
        if print_table:
            print("Top {} features in descending order of importance".format(top_n))
            print(feat_imp.T.to_string(header=False))
    except:
        print("This model not supports feature importance")
