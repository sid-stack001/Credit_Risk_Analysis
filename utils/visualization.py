import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    return fig

def plot_feature_importances(importances, feature_names):
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title('Feature Importances')
    return fig
