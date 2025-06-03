import matplotlib.pyplot as plt
import numpy as np
from LundTagger import LundTagger

def comparing_plotter(tagger1,tagger2,is_loss=False):
    """
    A function for comparing the accuracy or loss on validation data
    for two distinct taggers
    
    Attributes:
        tagger1:    a LundTagger

        tagger2:    a LundTagger

        is_loss:    a bool that indicates wheter you wish to plot the
                    loss metric. When set to False, it will plot the  
                    accuracy
    """
    if(not hasattr(tagger1,'validation_loader')): tagger1.set_loaders()
    if(not hasattr(tagger2,'validation_loader')): tagger2.set_loaders()
    least = min(tagger1.epochs,tagger2.epochs)
    loss1,acc1 = tagger1.epochwise_performance(tagger1.validation_loader)
    loss2,acc2 = tagger2.epochwise_performance(tagger2.validation_loader)
    if (is_loss): q1,q2 = loss1,loss2
    else: q1,q2 = acc1,acc2
    q1 = q1[:least]
    q2 = q2[:least]
    plt.figure()
    plt.plot(np.linspace(0,least,num=least),q1,color="orange")
    plt.plot(np.linspace(0,least,num=least),q2,color="blue")
    title = f"{'Loss' if is_loss else 'Accuracy'} Comparison"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(f"{'CE Loss' if is_loss else 'Accuracy'}")
    plt.legend([f"{tagger1.modelname}_{tagger1.n}tagger_{'with_pdg' if tagger1.pdg else 'no_pdg'}{tagger1.suffix}",f"{tagger2.modelname}_{tagger2.n}tagger_{'with_pdg' if tagger2.pdg else 'no_pdg'}{tagger2.suffix}"],loc = f"{'upper right' if is_loss else 'lower right'}")
    plt.savefig(f"./images/compared_{'loss' if is_loss else 'accuracy'}_{tagger1.modelname}_{tagger1.n}tagger_{'with_pdg' if tagger1.pdg else 'no_pdg'}{tagger1.suffix}_vs_{tagger2.modelname}_{tagger2.n}tagger_{'with_pdg' if tagger2.pdg else 'no_pdg'}{tagger2.suffix}.png")
    plt.close()

def compare_ROC_flavor(tagger1,tagger2):
    """
    A function to compare flavor vs flavor ROCs
    for two different taggers

    Attributes:
        tagger1:    a LundTagger object

        tagger2:    another LundTagger object
    """
    assert tagger1.n == tagger2.n, f"Impossible to compare the two taggers! one is a {tagger1.n}tagger the other a {tagger2.n}"
    for i in range(tagger1.n):
        for j in range(tagger2.n):
            if i==j: continue
            else:
                fpr1, tpr1,auc1 = tagger1.roc_flavor_vs_flavor(i,j)
                fpr2,tpr2,auc2 = tagger2.roc_flavor_vs_flavor(i,j)
                roc,ax = plt.subplots()
                ax.plot(fpr1,tpr1,lw=2,color='blue',label=f"{tagger1.modelname}_{tagger1.n}tagger_{'with_pdg' if tagger1.pdg else 'no_pdg'}{tagger1.suffix} (AUC: {auc1:.2f})")
                ax.plot(fpr2,tpr2,lw=2,color='orange',label=f"{tagger2.modelname}_{tagger2.n}tagger_{'with_pdg' if tagger2.pdg else 'no_pdg'}{tagger2.suffix} (AUC: {auc2:.2f})")
                ax.plot([0,1],[0,1],color="gray",lw=2,linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"{LundTagger.flavor_map[i]} VS {LundTagger.flavor_map[j]} ROC comparison")
                ax.legend(loc="lower right")
                ax.grid(alpha=0.5)
                roc.savefig(f"./images/ROC_compared_{LundTagger.flavor_map[i]}_vs_{LundTagger.flavor_map[j]}-{tagger1.modelname}_{tagger1.n}tagger_{'with_pdg' if tagger1.pdg else 'no_pdg'}{tagger1.suffix}_vs_{tagger2.modelname}_{tagger2.n}tagger_{'with_pdg' if tagger2.pdg else 'no_pdg'}{tagger2.suffix}")
                plt.close(roc)