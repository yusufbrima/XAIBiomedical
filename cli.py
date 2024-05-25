import click
import logging
import os
from pathlib import Path
from data import DataLoad
from utils import Utils
from model import Models
import numpy as np
import matplotlib.pyplot as plt
from Download import downloader
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import random
from saliencyAnalysis import Saliency,Main

# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42) 
np.seterr(divide='ignore', invalid='ignore')

CUSTOM_DATASETS = ["./Data/brainTumorDataPublic", "./Data/COVID-19_Radiography_Dataset"]

@click.command()
@click.option("--name",default="Download", prompt="Enter the action to perform [Download, Process, Train, Evaluate, Saliency]", help="This cli helps access the functionality of the program.")
@click.option("--batch_size",default=32, help="The batch size to use during training.")
@click.option("--epochs",default=10, help="The number of epochs to train the model.")
@click.option("--ds",default="BrainTumor", help="The dataset to use for training the model.")

def main(name, batch_size, epochs, ds):
    """Simple script allows to class the necessary routine to execute"""
    actions = ["Download", "Process", "Train", "Saliency"]
    if(name.capitalize() in actions):
        click.echo(f"Run, {name.capitalize()}!")
        if(name.capitalize() == actions[0]):
            """
                This script executes the download routine
            """
            if ds == "BrainTumor":
                dloader =  downloader()
                dloader.get()
            else:
                click.echo("Invalid dataset selected, please try again")
                # dloader =  downloader()
                # dloader.get()
        
        elif(name.capitalize() == actions[1]):
            """
            This section of the code extracts the mri images, bounding boxes, and class labels 
            """
            # Next We call the dataloader method to build the dataset and load it into memory 
            if ds == "BrainTumor":
                dl =  DataLoad(segmented=True)
                dl.build(flag=True)
            elif ds == "Covid":
                dpath = Path(f"{CUSTOM_DATASETS[1]}")
                dl =  DataLoad(datapath = dpath,outdir = f'{CUSTOM_DATASETS[1]}.npz')
                dl.build(flag=True, covid=True)
            else:
                click.echo("Invalid dataset selected, please try again") 
        elif(name.capitalize() == actions[2]):
            """
            This section splits the dataset into train, test, and validation sets. We plot sample figures and save them to the Figures directory
            """
            if ds == "BrainTumor":
                dl =  DataLoad()
                # dl.build(flag=True)
                dl.load()
                idx = np.random.randint(0, dl.X.shape[0],16)

                Utils.plot_samples(dl.A,dl.y,dl.Z,dl.CLASSES,idx,save=True, show=False, segmented=True, masked=True, filename=f"{ds}_Sample", fmt="tiff")

                #@title Splitting the dataset into train/test
                X_train,X_test,y_train,y_test = train_test_split(dl.X_,dl.y, test_size=0.1, shuffle=True)
                Utils.project2D(X_test, y_test,dl.CLASSES,figname=f'{ds}_Test_XMRI_TSNE_Before',save=True, show=False)
                
                #@title Instantiating the Models method
                models = Models()
                #@title Training different CNN model
                dft = models.train(X_train= X_train,X_test = X_test,y_train = y_train,y_test = y_test,input_shape = X_train.shape[1:],output_nums = len(dl.CLASSES),CLASSES = dl.CLASSES,epochs=epochs, batch_size=batch_size, dataset=ds)
                
                # We are plotting the evaluation result 
                
                dft =  pd.read_csv(f'./Data/{ds}_Evaluation_Results.csv')
                # Utils.plot_evaluation(dft)
                top_n_dft = dft.nlargest(3,'f1_score')
                for j in  tqdm(range(top_n_dft.shape[0])):
                    # print(j, top_n_dft.iloc[j,1])
                    model_name = top_n_dft.iloc[j,1]
                    X_hat = Utils.create_embedding(f"{ds}_{model_name}", X_test)
                    Utils.project2D(X_hat, y_test,dl.CLASSES,figname=f'{ds}_Test_set_{model_name}',save=True, show=False)
            elif ds == "Covid":
                dpath = Path(f"{CUSTOM_DATASETS[1]}")
                dl =  DataLoad(datapath = dpath,outdir = f'{CUSTOM_DATASETS[1]}.npz')
                dl.load(masked=False)

                idx = np.random.randint(0, dl.X.shape[0],16)

                Utils.plot_samples(dl.X,dl.y,dl.X,dl.CLASSES,idx,save=True, show=False, segmented=False, masked=False, filename=f"{ds}_Sample", fmt="tiff")

                #@title Splitting the dataset into train/test
                X_train,X_test,y_train,y_test = train_test_split(dl.X_,dl.y, test_size=0.1, shuffle=True)
                Utils.project2D(X_test, y_test,dl.CLASSES,figname=f'{ds}_Test_XMRI_TSNE_Before',save=True, show=False)

                #@title Instantiating the Models method
                models = Models()

                #@title Training different CNN model
                dft = models.train(X_train= X_train,X_test = X_test,y_train = y_train,y_test = y_test,input_shape = X_train.shape[1:],output_nums = len(dl.CLASSES),CLASSES = dl.CLASSES,epochs=epochs, batch_size=batch_size, dataset=ds)

                # We are plotting the evaluation result
                dft =  pd.read_csv(f'./Data/{ds}_Evaluation_Results.csv')

                # Utils.plot_evaluation(dft)
                top_n_dft = dft.nlargest(3,'f1_score')
                for j in  tqdm(range(top_n_dft.shape[0])):
                    # print(j, top_n_dft.iloc[j,1])
                    model_name = top_n_dft.iloc[j,1]
                    X_hat = Utils.create_embedding(f"{ds}_{model_name}", X_test)
                    Utils.project2D(X_hat, y_test,dl.CLASSES,figname=f'{ds}_Test_set_{model_name}',save=True, show=False)


            else:
                click.echo("Invalid dataset selected, please try again")

        elif(name.capitalize() == actions[3]):
            """
              Here we perform saliency analysis for the dataset
            """
            click.echo("We are executing the saliency command, hurray!") 
            
            if ds == "BrainTumor":
                dl =  DataLoad()

                dl.load()

                idx = Saliency.pick_indices(dl, n = 3)
                models = Main.load_models(n =3, file_path=f"./Data/{ds}_Evaluation_Results.csv")
                for i in tqdm(range(len(models))):
                    
                    logging.info(f"Loading {models[i]}")
                    model = tf.keras.models.load_model(f"./Models/{ds}_{models[i]}.keras")

                    logging.info("Computing saliency masks, this may take a while...")
                    sa =  Saliency(model, dl, idx, masked=True, dataset=ds)
                    mycollection, data, vizresult =  sa.analyze()

                    logging.info("Plotting heatmaps of relevant features")
                    sa.plot_results(mycollection, data)
                    logging.info(f"Saliency analysis completed for {model.name} model")
            elif ds == "Covid":
                dpath = Path(f"{CUSTOM_DATASETS[1]}")
                dl =  DataLoad(datapath = dpath,outdir = f'{CUSTOM_DATASETS[1]}.npz', segmented=False)
                dl.load(masked=False)
                idx = Saliency.pick_indices(dl, n = 3)
                models = Main.load_models(n =3, file_path=f"./Data/{ds}_Evaluation_Results.csv")
                for i in tqdm(range(len(models))):
                    
                    logging.info(f"Loading {models[i]}")
                    model = tf.keras.models.load_model(f"./Models/{ds}_{models[i]}.keras")

                    logging.info("Computing saliency masks, this may take a while...")
                    sa =  Saliency(model, dl, idx, masked=False, dataset=ds)
                    mycollection, data, vizresult =  sa.analyze()

                    logging.info("Plotting heatmaps of relevant features")
                    sa.plot_results(mycollection, data, mri=False)
                    logging.info(f"Saliency analysis completed for {model.name} model")
            else:
                click.echo("Invalid dataset selected, please try again")
        else:
            click.echo("Invalid option selected, please try again")
    else:
        click.echo("Invalid command select, please try again")
if __name__ == '__main__':
    """
      Here is where all stuff runs
    """
    logging.basicConfig(level=logging.INFO)
    main()

