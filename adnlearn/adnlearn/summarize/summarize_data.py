import matplotlib.pyplot as plt
import seaborn


def descriptive_statistics(dataset, target):
    print("Descriptive statistics:")

    # shape
    print("**** Shape ****")
    print(dataset.shape)
    
    # types
    print("**** Types ****")
    print(dataset.dtypes)
    
    # descriptions
    print("**** Descriptions ****")
    description = dataset.describe()
    print(description)
    
    # correlation
    print("**** Correlations ****")
    correlations = dataset.corr(method='pearson')
    print(correlations)
    
    # class distribution
    print("**** Class distribution ****")
    error_counts = dataset.groupby(target).size()
    print(error_counts)
    
    # skew
    print("**** Skew ****")
    skew = dataset.skew()
    print(skew)


def data_visualizations(dataset, target):
    # histograms
    dataset.hist()
    plt.show()
    
    # density
    dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
    plt.show()
    
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
    plt.show()

    # scatter plot matrix
    seaborn.pairplot(dataset, hue=target, size=3, diag_kind="kde")
    plt.show()
