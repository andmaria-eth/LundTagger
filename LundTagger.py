import os
import importlib.util
from typing import Self
from utils.JetGraphProducer import JetGraphProducer
import torch
from data_loading.graphDataLoader import GraphDataLoader
from sklearn.metrics import roc_curve, auc
from utils.LundTreeUtilities import OnTheFlyNormalizer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np

class LundTagger():

    samples_folder = "./samples"
    images_folder = "./images"

    flavor_map ={
        0: "s",
        1: "c",
        2: "b",
        3: "u",
        4: "d",
        5: "g"
    }

    xsec_map = {
        0: 5e-5,
        1: 0.006,
        2: 0.116,
        3: 3.8e-8,
        4: 1.5e-7,
        5: 0.001
    }

    color_map = {
        4: ["red","blue","purple","orange"],
        5: ["red","blue","purple","orange","pink"],
        6: ["red","blue","purple","orange","pink","yellow"]
    }

    fraction_map = {
        4: [1/4,1/4,1/4,1/8,1/8],
        5: [1/5,1/5,1/5,1/10,1/10,1/5],
        6: [1/6,1/6,1/6,1/6,1/6,1/6]
    }

    labels_map = {
        4: ["s","c","b","lq"],
        5: ["s","c","b","lq","g"],
        6: ["s","c","b","u","d","g"]
    }

    cap = 0.80
    train_batchsize = 2056
    val_batchsize = 1028

    def __init__(self,n_flavors,pdg = False,suffix="",epochs = 150,modelname = "arch"):
        """
        Intializes the object.

        Attributes:
            n_flavors:  how many flavors the tagger should discriminate.
                        This should be either 4 [(lq,s,c,b) with lq indicating
                        light quarks up and down jointly], 5 [(lq,s,c,b,g) with
                        g gluon initiated showers] or 6 (u,d,s,c,b,g)
            
            pdg:        a bool indicating whether you would like to use the 
                        PDG information 
            
            epochs:     how many epochs you would like to use by default

            modelname:  a string with the name of the script where the model   
                        you want to use is defined (you do not need to specify
                        the absolute path: just insert the script name where
                        your LundNetTagger architecture is defined and the program
                        will look for your script in \'./architectures/\'). The
                        default name is \'arch\'
        """
        assert n_flavors in {4,5,6},"n_flavors should be chosen among {4,5,6}"
        self.n = n_flavors
        self.pdg = pdg
        self.suffix =  suffix
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modelname.endswith(".py"): self.modelname = os.path.splitext(os.path.basename(self.modelname))[0]
        else: self.modelname = modelname
        print(f"Running on device: {self.device}")

    @staticmethod
    def normalization_params(jgp_list):
        """
        A static method for evaluating mean and standard
        deviation within a list of JetGraphProducers.

        Attributes:
            jgp_list:   a python list of JetGraphProducers
        
        Returns:
            means:  a NumPy array containing the mean values of
                    the different features in the JetGraphProducers

            stds:   a NumPy array containing the standard deviations
                    of the different features in the JetGraphProducers
        """
        means, stds= 0.,0.
        counts = []
        for producer in jgp_list:
            for graph in producer:
                means+=graph.x.sum(dim=0)
                counts.append(graph.num_nodes)
        means /= sum(torch.tensor(counts))
        for producer in jgp_list:
            for graph in producer:
                stds += ((graph.x-means)**2).sum(dim=0)
        stds /= sum(torch.tensor(counts))
        stds = torch.sqrt(stds)
        stds = np.where(np.isnan(stds) | (stds == 0), 1, stds)
        return means,stds

    def set_jgps(self):
        """
        A method to set up to represent the dataset as 
        JetGraphProducers. This is the basic class for 
        using the SVJ_GNN framework
        """
        jgps = []
        for i in (range(5) if self.n==4 else range(6)):
            jgps.append(JetGraphProducer(
                    f"{LundTagger.samples_folder}/p8_ee_ZH{LundTagger.flavor_map[i]}{LundTagger.flavor_map[i]}_ecm240.root",
                    f"{LundTagger.flavor_map[i]}_jgp_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}",
                    n_store_jets = 2,
                    use_lund_decomp=True,
                    n_lund_vars=5,
                    save_4vectors_in_lund_tree=False,
                    save_energy_fractions=True,
                    save_pdg = self.pdg,
                    save_n_constituents=False,
                    weights=None,
                    verbose=True,
                    label=i,
                    use_relative_angles=False,
                    use_dummy_values=False,
                    use_displacement=False
                )
            )
        self.jgps = jgps

    def set_loaders(self):
        """
        A method to set up the loader we will use during training.
        It normalizes the datasets and then creates the loaders,
        saving the permutation that's being used
        """
        if not hasattr(self,'jgps'): self.set_jgps()
        norm_file = f"./norm/{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.pth"
        if os.path.isfile(norm_file):
            data = torch.load(norm_file)
            mean,std = data["means"],data["stds"]
            normalizer = OnTheFlyNormalizer(["x"],mean,std)
        else:
            mean, std = self.normalization_params(self.jgps)
            normalizer = OnTheFlyNormalizer(["x"],mean,std)
            torch.save({"means": mean, "stds": std},norm_file)

        perm_file = f"./perms/{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.pth"
        if os.path.isfile(perm_file):
            perm = torch.load(perm_file)
        else: perm = {}
        self.training_loader = GraphDataLoader(
            [[LundTagger.samples_folder,f"{LundTagger.flavor_map[i]}_jgp_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}"] for i in range(len(self.jgps))],
            window_to_load = [[0.,LundTagger.cap] for i in range(len(self.jgps))],
            normalizer=normalizer,
            batch_size=LundTagger.train_batchsize,
            shuffle = True,
            permutation = perm,
            dataset_fraction = LundTagger.fraction_map[self.n],
        )
        perm = (self.training_loader).get_permutation()
        if(not os.path.isfile(perm_file)): 
            torch.save(perm,perm_file)
        self.perm = perm 
        self.validation_loader = GraphDataLoader(
            [[LundTagger.samples_folder,f"{LundTagger.flavor_map[i]}_jgp_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}"] for i in range(len(self.jgps))],
            window_to_load = [[LundTagger.cap,1.] for i in range(len(self.jgps))],
            normalizer=normalizer,
            batch_size=LundTagger.val_batchsize,
            shuffle = True,
            permutation = self.perm,
        )

    def set_model(self):
        """
        A method to intially set up the method
        """
        if not hasattr(self,'jgps'): self.set_jgps()
        filepath = f"./architectures/{self.modelname}.py"
        spec = importlib.util.spec_from_file_location(self.modelname, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.LundNetTagger(self.jgps[0][0].x.shape[1], self.n)
        model.to(device=self.device)
        self.model = model

    def evaluate_batch(self, data, criterion):
        """
        A method to obtain the accuracy and loss of 
        a certain batch

        Attributes:
            data:       the batch we are evaluating
            
            criterion:  An Instance of possibly any loss class
                        (here we are mostly using CE Loss)

        Returns:
            float acc_value:    The accuracy on the batch that's being evaluated

            torch.tensor loss:  The mean loss across the batch as a one-dimensional
                                torch tensor
        """
        out = self.model(data)  
        indices = data.y.type(torch.int64)
        if(self.n==5): indices = torch.where(indices==4,3,indices)
        indices = torch.clamp(indices,max=self.n-1)
        loss = criterion(out,indices)
        pred = out.argmax(dim=1)
        acc_value = (pred == data.y).sum().item()
        return acc_value,loss

    def train(self):
        """
        A method to train the model that's currently loaded

        """
        if not hasattr(self,'model'): self.set_model()
        if not hasattr(self,'training_loader'): self.set_loaders()
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-2,weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer,max_lr=1e-2,div_factor=200,epochs=self.epochs, steps_per_epoch=len(self.training_loader))
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            self.model.train()
            for data1 in self.training_loader:
                data = data1.to(self.device)
                optimizer.zero_grad()  
                acc_value,loss = self.evaluate_batch(data, criterion)
                loss.backward()
                optimizer.step()
                scheduler.step()
            model_file = f"./model_instances/{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}_epoch{epoch}.pt"
            torch.save(self.model.state_dict(),model_file)
        final_model_file = f"./models/{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.pt"
        torch.save(self.model.state_dict(),final_model_file)

    def load(self,epoch=None):
        """
        A method for loading a particular model

        Attributes:
            epoch:  an int that specifies which epoch of the model
                    to load. If set to None it will load the 
                    \'final\' version of the model saved
                    in \'./models/\'. If epoch is set None and 
                    the final model is not found, the training
                    will start
        """
        if not hasattr(self,'model'): self.set_model()
        if not hasattr(self,'validation_loader'): self.set_loaders()
        if epoch is None: 
            model_file = f"./models/{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.pt"
            if os.path.isfile(model_file):
                self.model.load_state_dict(torch.load(model_file))
            else:
                print(f"File {model_file} not found. Training the model")
                self.train()
        else:
            model_file = f"./model_instances/{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}_epoch{epoch}.pt"
            if os.path.isfile(model_file):
                self.model.load_state_dict(torch.load(model_file))
            else:
                print(f"File {model_file} not found. No model was loaded or trained")

    def evaluate_model(self, loader, epoch=None):
        """
        A method for obtaining the accuracy and loss of the model
        saved

        Attributes:
            loader: The loader containg the data on whose performances
                    you are evaluating
            
            epoch:  An int indicating which epoch of the model you want 
                    to evalute. If set to None, it will evaluate the
                    model that's considered as \'final\' (the one in the
                    \'./models\' folder)
                
        Returns:
            float acc:  the accuracy of the model on the specified loader
            
            float loss_value:   the CE Loss of the model on the loader 
        """
        if not hasattr(self,'model'): self.set_model()
        self.load(epoch)
        loss_value = 0
        acc = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss() 
        self.model.eval()
        with torch.no_grad():
            for data1 in loader:
                data = data1.to(self.device)
                acc_value,loss = self.evaluate_batch(data,criterion)
                loss_value += loss.item()
                acc += acc_value
                total += data.y.size(0)
            acc = acc / total  
            loss_value = loss_value/len(loader)
        return acc, loss_value
    
    def epochwise_performance(self,loader):
        """
        A function for evaluating accuracy and loss of the currently 
        loaded model over epochs

        Attributes:
            loader: the loader on which you want to evaluate loss and 
                    accuracy

        Returns:
            loss:   a list of floats containg the value of the loss
                    at each epoch
            
            acc:    a list of floats containg the value of the accuracy
                    at each epoch
        """
        loss = []
        acc = []
        for i in (range(self.epochs)):
            temp_acc,temp_loss = self.evaluate_model(loader,i)
            loss.append(temp_loss)
            acc.append(temp_acc)
        return loss,acc

    def plot_confusion(self,loader):
        """
        Plots the confusion matrix for the model that's saved

        Attributes:
            loader: The loader containing the data whose 
                    confusion you want to check
        """
        matrix = torch.zeros((self.n,self.n),dtype=torch.float64,device=self.device)
        tot=0
        self.model.eval()
        for data1 in loader:
            data = data1.to(self.device)
            out = self.model(data)
            true_indices = data.y.type(torch.int64)
            if (self.n==5): true_indices = torch.where(true_indices==4,3,true_indices) 
            true_indices = torch.clamp(true_indices,max=self.n-1)
            pred_indices = out.argmax(dim=1)
            for t,p in zip(true_indices,pred_indices):
                matrix[p,t]+=1
        matrix = matrix/matrix.sum(dim=0,keepdim=True)
        matrix = matrix.cpu().numpy()
        labels = LundTagger.labels_map[self.n]
        plt.figure()
        plt.imshow(matrix, cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Fraction")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color="black")
        plt.xticks(np.arange(self.n), labels)
        plt.yticks(np.arange(self.n), labels)
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion Matrix")
        plt.savefig(f"{LundTagger.images_folder}/confusion_{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.png")
        plt.close()
    
    def plot_train_vs_val(self,q_training,q_validation,is_loss = False):
        """
        A static method for plotting loss or accuracy on both training and 
        validation dataset over the epochs for comparison

        Attributes:
            q_training:     the quantity (loss or accuracy) 
                            related to training. It should be
                            an one-dimensional array with (at least)
                            as many entries as epochs in the class
            
            q_validation:   the quantity (loss or accuracy) 
                            related to validation. It should be
                            an one-dimensional array with (at least)
                            as many entries as epochs in the class
            
            is_loss:        a bool indicating whether you are trying to
                            plot loss (if is_loss=False the accuracy will
                            be plotted)
        """
        plt.figure()
        plt.plot(np.linspace(0,self.epochs,num=self.epochs),q_training,color="orange")
        plt.plot(np.linspace(0,self.epochs,num=self.epochs),q_validation,color="blue")
        title = f"Training vs Validation {'Loss' if is_loss else 'Accuracy'} {'with PDG' if self.pdg else 'without'} ({self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix})"
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(f"{'CE Loss' if is_loss else 'Accuracy'}")
        plt.legend(["Training","Validation"],loc = f"{'upper right' if is_loss else 'lower right'}")
        plt.savefig(f"{LundTagger.images_folder}/{'loss' if is_loss else 'accuracy'}_{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.png")
        plt.close()

    def roc_params(self,use_weights=False):
        """
        A method that evaluates the ROC parameters for a model for
        each flavor in a one vs all fashion

        Attributes:
            use_weights:    defines whether using the physical weights
                            using the different cross-sections

        Returns:
            fpr:    a list of list. Specifying the first index we obtain
                    the false positive rates for the flavor of that index 
                    as in flavor_map

            tpr:    a list of list. Specifying the first index we obtain
                    the true positive rates for the flavor of that index 
                    as in flavor_map
            
            model_auc:  Area under the Curve (ROC)
        """
        self.model.eval()
        if (not hasattr(self,'validation_loader')): self.set_loaders()
        loader = self.validation_loader
        fpr = []
        tpr = []
        model_auc = []
        with torch.no_grad(): 
            for i in range(self.n):
                label = []
                prob = []
                weights = []
                for data1 in loader:
                    data = data1.to(self.device)
                    out = self.model(data)
                    sm = nn.Softmax(dim=1)
                    out = sm(out)
                    out = out.to(self.device)
                    keys = torch.tensor(list(LundTagger.xsec_map.keys()),dtype=torch.int64,device=self.device)
                    values = torch.tensor(list(LundTagger.xsec_map.values()), device=self.device)  
                    values.to(self.device)
                    offset = -keys.min().item() if keys.min() < 0 else 0
                    keys_shifted = keys + offset
                    keys_shifted.to(self.device)
                    lookup_table = torch.zeros(keys_shifted.max() + 1, dtype=torch.float32,device=self.device)
                    lookup_table[keys_shifted] = values  
                    data_y_shifted = data.y + offset 
                    data_y_shifted = data_y_shifted.type(torch.int64)
                    weights.append(lookup_table[data_y_shifted].cpu().numpy())
                    target = torch.zeros((len(data),self.n),dtype=torch.float32,device=self.device)
                    indices = data.y.type(torch.int64)
                    if (self.n==5): torch.where(indices==4,3,indices)
                    indices = torch.clamp(indices,max=self.n-1)
                    target.scatter_(1,indices.unsqueeze(1),1)
                    prob.append(out[:,i].detach().cpu().numpy())
                    label.append(target[:,i].detach().cpu().numpy())
                weights=np.concatenate(weights)
                prob = np.concatenate(prob)
                label = np.concatenate(label)
                if use_weights: temp_fpr,temp_tpr,_ = roc_curve(label,prob,sample_weight=weights) 
                else: temp_fpr,temp_tpr,_ = roc_curve(label,prob)
                temp_auc = auc(temp_fpr,temp_tpr)
                fpr.append(temp_fpr)
                tpr.append(temp_tpr)
                model_auc.append(temp_auc)
        return fpr,tpr,model_auc

    def plot_rocs(self,use_weights=False):
        """
        A method for plotting the ROCs of each flavor
        in a class vs all-others manner

        Attr:
            use_weights:    a bool. If True will apply physical
                            weights to each flavor
        """
        fpr, tpr,auc = self.roc_params(use_weights)
        plt.figure()
        for i in range(self.n):
            plt.plot(fpr[i], tpr[i], lw=2 ,color=f"{LundTagger.color_map[self.n][i]}", label=f"{LundTagger.labels_map[self.n][i]}-tagging (AUC = {auc[i]:.2f})") 
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC for {self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)
        fname = f"{LundTagger.images_folder}/ROC_{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.png"
        plt.savefig(fname)
        plt.close()

    def roc_flavor_vs_flavor(self, pos_idx,neg_idx):
        """
        A method for finding the roc parameters of a ROC curve that exclusively compares
        two flavors performances

        Attributes:
            pos_idx:    the idx of the flavor that's considered as positive as in flavor_map

            neg_idx:    the idx of the flavor that's considered as negative as in flavor_map

        Returns:
            fpr:    false positive rate

            tpr:    true positive rate
            
            model_auc:  Area under the Curve (ROC)
        """
        if(not hasattr(self,'validation_loader')): self.set_loaders()
        loader = self.validation_loader
        if(not hasattr(self,'model')): self.load()
        self.model.eval()
        with torch.no_grad():
            label = []
            prob =  []
            for data1 in loader:
                data = data1.to(self.device)
                out = self.model(data).to(self.device)
                out.to(self.device)
                sm = nn.Softmax(dim=1)
                out = sm(out)
                target = torch.zeros((len(data),6),dtype=torch.float32,device=self.device)
                indices = data.y.type(torch.int64)
                indices = torch.clamp(indices,min=0)
                target.scatter_(1,indices.unsqueeze(1),1)
                mask = (target[:,pos_idx]==1) | (target[:,neg_idx]==1)
                filtered_label = target[mask]
                temp_prob = out[:,pos_idx]/(out[:,pos_idx]+out[:,neg_idx])
                filtered_prob = temp_prob[mask]
                prob.append(filtered_prob.detach().cpu().numpy())
                label.append(filtered_label[:,pos_idx].detach().cpu().numpy())
            prob=np.concatenate(prob)
            label = np.concatenate(label)
            fpr,tpr,_ = roc_curve(label,prob)
            model_auc = auc(fpr,tpr)
        return fpr,tpr,model_auc
    
    def plot_flavored_rocs(self):
        """
        A method to plot the ROCs for each flavor 
        in a class vs class manner
        """
        for i in range(self.n):
            fig,ax = plt.subplots()
            for j in range(self.n):
                if i==j: continue
                else:
                    temp_fpr_no_pdg, temp_tpr_no_pdg,temp_auc_no_pdg = self.roc_flavor_vs_flavor(i,j)
                    ax.plot(temp_fpr_no_pdg,temp_tpr_no_pdg, lw=2,color=LundTagger.color_map[self.n][j],label=f"{LundTagger.labels_map[self.n][i]} VS {LundTagger.labels_map[self.n][j]} (AUC = {temp_auc_no_pdg:.2f})")
                    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title(f"{LundTagger.labels_map[self.n][i]} VS other flavors ROCs ({self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix})")
                    ax.legend(loc="lower right")
                    ax.grid(alpha=0.5)
            fig.savefig(f"{LundTagger.images_folder}/ROC_{LundTagger.labels_map[self.n][i]}_{self.modelname}_{self.n}tagger_{'with_pdg' if self.pdg else 'no_pdg'}{self.suffix}.png")
            plt.close(fig)
