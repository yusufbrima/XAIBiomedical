#Importing the saliency module. If not installed use the above pip command to install it in your environment
import PIL.Image
from matplotlib import pylab as P
from matplotlib import pylab as plt
import saliency.core as saliency
from pathlib import Path
from data import DataLoad
import tensorflow as tf
from utils import PackageManager
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
logging.basicConfig(level=logging.INFO)
PackageManager.install_and_import('saliency')
logging.info("Package installed successfully")
import saliency.core as saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus
from tf_keras_vis.scorecam import Scorecam
logging.info("Package import completed successfully")


class Saliency:

    def __init__(self, model,dl,idx, masked = True, dataset = "dataset") -> None:
        self.model =  model
        self.dl =  dl
        self.class_idx_str = 'class_idx_str'
        self.idx = idx
        self.masked =  masked 
        self.dataset = dataset
    
    #@title Snippet to compute prediction_feature gradients
    def call_model_function(self,images, call_model_args=None, expected_keys=None):
        target_class_idx =  call_model_args[self.class_idx_str]
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
                tape.watch(images)
                output_layer = self.model(images)
                output_layer = output_layer[:,target_class_idx]
                gradients = np.array(tape.gradient(output_layer, images))
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                logging.info("Computing gradients for convolutional layers")
                conv_layer, output_layer = self.model(images)
                gradients = np.array(tape.gradient(output_layer, conv_layer))
                return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

    @staticmethod
    def PreprocessImage(im):
        return tf.keras.applications.xception.preprocess_input(im)
    
    #@title We are select unique sample indices from each class 
    @staticmethod
    def pick_indices(dl, n=3):
        sidx =  []
        labels = []

        while(len(labels) < n):
            idx = np.random.randint(0,dl.X_.shape[0])
            label = dl.y[idx]
            if( label not in labels):
                sidx.append(idx)
                labels.append(label)
        return sidx

    def predict(self, im):
        im = np.expand_dims(im, axis=0)
        predictions = self.model(im)
        prediction_class = np.argmax(predictions[0])
        call_model_args = {self.class_idx_str: prediction_class}
        pred_prob = np.round(predictions[0,prediction_class].numpy(),2)
        return predictions,prediction_class,call_model_args,pred_prob
    #@title Function to perform model inference
    def inference(self, idx):
        im  = self.dl.X_[idx]
        # add batch dimension
        im = np.expand_dims(im, axis=0)
        predictions = self.model(im)
        prediction_class = np.argmax(predictions[0])
        call_model_args = {self.class_idx_str: prediction_class}
        pred_prob = np.round(predictions[0,prediction_class].numpy(),2)
        return predictions,prediction_class,call_model_args,pred_prob

    def analyze(self):
        #@title Saliency Computation Code updated 2
        if(self.masked):
            data =  {'img': [], 'tumorBorder': [], 'original': [],'predictions': [], 'prediction_class': [],'call_model_args':[],'pred_prob':[],'actual': [],'idx': []}
            vizresult = {'Image': [],'tumorBorder': [], 'VG': [] ,'SmoothGrad': [] ,'IG': [] ,'SmoothGrad': [] ,'XRAI_Full': [] ,'Fast_XRAI' :[] ,'VIG': [], 'GIG': [] ,'Blur_IG':[],'GradCAM': [], 'GradCAM++':[],'ScoreCAM':[] } #'GradCAM': []
        else:
            data =  {'img': [],'predictions': [], 'prediction_class': [],'call_model_args':[],'pred_prob':[],'actual': [],'idx': []}
            vizresult = {'Image': [], 'VG': [] ,'SmoothGrad': [] ,'IG': [] ,'SmoothGrad': [] ,'XRAI_Full': [] ,'Fast_XRAI' :[] ,'VIG': [], 'GIG': [] ,'Blur_IG':[], 'GradCAM': [], 'GradCAM++':[],'ScoreCAM':[]} #
        mycollection = []
        for idx in self.idx:
            predictions,prediction_class,call_model_args,pred_prob = self.inference(idx)
            data['img'].append(self.dl.X_[idx])
            if(self.masked):
                data['tumorBorder'].append(self.dl.Z[idx])
                data['original'].append(self.dl.A[idx])
            data['predictions'].append(predictions)
            data['prediction_class'].append(prediction_class)
            data['call_model_args'].append(call_model_args)
            data['pred_prob'].append(pred_prob)
            data['actual'].append(self.dl.y[idx])
            data['idx'].append(idx)

            im =  self.dl.X_[idx]
            if(self.masked):
                vizresult['Image'].append(self.dl.A[idx])
                vizresult['tumorBorder'].append(self.dl.Z[idx])


            # Construct the saliency object. This alone doesn't do anthing.
            gradient_saliency = saliency.GradientSaliency()

            # Compute the vanilla mask and the smoothed mask.
            vanilla_mask_3d = gradient_saliency.GetMask(im, self.call_model_function, call_model_args)
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, self.call_model_function, call_model_args)

            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
            vizresult['VG'].append(vanilla_mask_grayscale)
            vizresult['SmoothGrad'].append(smoothgrad_mask_grayscale)

            if(self.masked):
                mycollection.append(self.dl.A[idx])
            else:
                mycollection.append(self.dl.X[idx])
            mycollection.append(vanilla_mask_grayscale)
            mycollection.append(smoothgrad_mask_grayscale)


            # Construct the saliency object. This alone doesn't do anthing.
            integrated_gradients = saliency.IntegratedGradients()

            # Baseline is a black image.
            baseline = np.zeros(im.shape)

            # Compute the vanilla mask and the smoothed mask.
            vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
            # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
            smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

            mycollection.append(vanilla_mask_grayscale)
            mycollection.append(smoothgrad_mask_grayscale)


            # Construct the saliency object. This alone doesn't do anthing.
            xrai_object = saliency.XRAI()

            # Compute XRAI attributions with default parameters
            xrai_attributions = xrai_object.GetMask(im, self.call_model_function, call_model_args, batch_size=20)

            # Show most salient 30% of the image
            mask = xrai_attributions > np.percentile(xrai_attributions, 70)
            im_mask =  np.array(im)
            im_mask[~mask] = 0

            mycollection.append(xrai_attributions)
            mycollection.append(im_mask)

            # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
            integrated_gradients = saliency.IntegratedGradients()
            guided_ig = saliency.GuidedIG()

            # Baseline is a black image for vanilla integrated gradients.
            baseline = np.zeros(im.shape)

            # Compute the vanilla mask and the Guided IG mask.
            vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
            guided_ig_mask_3d = guided_ig.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
            guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)

            mycollection.append(vanilla_mask_grayscale)
            mycollection.append(guided_ig_mask_grayscale)


            # Compare BlurIG and Smoothgrad with BlurIG. Note: This will take a long time to run.
            blur_ig = saliency.BlurIG()

            # Compute the Blur IG mask and Smoothgrad+BlurIG mask.
            blur_ig_mask_3d = blur_ig.GetMask(im, self.call_model_function, call_model_args, batch_size=20)
            # Smoothed mask for BlurIG will take a while since we are doing nsamples * nsamples computations.
            smooth_blur_ig_mask_3d = blur_ig.GetSmoothedMask(im, self.call_model_function, call_model_args, batch_size=20)

            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
            smooth_blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(smooth_blur_ig_mask_3d)

            mycollection.append(blur_ig_mask_grayscale)
            mycollection.append(smooth_blur_ig_mask_grayscale)

            # print("The shape of big blur ig mask is ", blur_ig_mask_grayscale.shape)

            # Compute the Grad-CAM mask 

            replace2linear = ReplaceToLinear()
            score = CategoricalScore([self.dl.y[idx]])

            # Create Gradcam object
            gradcam = Gradcam(self.model,
                            model_modifier=replace2linear,
                            clone=True)

            # Generate heatmap with GradCAM
            cam = gradcam(score,im,penultimate_layer=-1)

            cam = normalize(cam)

            mycollection.append(cam)

            # Create GradCAM++ object
            gradcam = GradcamPlusPlus(self.model,
                                    model_modifier=replace2linear,
                                    clone=True)
            
            # Generate heatmap with GradCAM++
            gradcamplusplus = gradcam(score,
                        im,
                        penultimate_layer=-1)

            cam = normalize(gradcamplusplus)

            mycollection.append(cam)

            # Create ScoreCAM object
            scorecam = Scorecam(self.model)

            # Generate heatmap with ScoreCAM
            scorecam = scorecam(score, im, penultimate_layer=-1)

            cam = normalize(scorecam)

            mycollection.append(cam)
                    

        return mycollection, data, vizresult
    
    def plot_results(self,mycollection, data, mri = True, fmt="tiff", show=False):
        #Visualizating the saliency maps
        titles = ['Input Image', 'Vanilla Gradient','SmoothGrad','Integrated Gradients','SmoothGrad','XRAI Full','Fast XRAI 30%','VIG', 'Guided IG','Blur IG','Smooth Blur IG', 'GradCAM', 'GradCAM++','ScoreCAM']
        # r,c =  3,len(mycollection)//3 #r is the  number of rows and c is the number of columns
        r =  3
        c =  len(titles)
        img_idx =  list(np.arange(0,int(r*c),c))
        fig, axs = plt.subplots(r,c, figsize=(22,8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.125,bottom=0.2,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
        axs = axs.ravel()
        exclude = [0, 14,28]

        # grad_cams  = [11,12,13, 25, 26, 27, 39, 40, 41]
        grad_cams_first = [11, 12, 13]
        grad_cams_second = [25, 26, 27]
        grad_cams_third = [39, 40, 41]
        if(self.masked and   mri ):
            for i in tqdm(range(int(r*c))):
                if( i in list(range(c))):
                    axs[i].set_title(titles[i],fontsize=12)
                if i not in exclude:
                    if i in grad_cams_first:
                        axs[i].imshow(np.squeeze(self.dl.X_[self.idx[0]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(mycollection[exclude[0]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(data['img'][exclude[0]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_second:
                        axs[i].imshow(np.squeeze(self.dl.X_[self.idx[1]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(mycollection[exclude[1]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(data['img'][exclude[1]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_third:
                        axs[i].imshow(np.squeeze(self.dl.X_[self.idx[2]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(mycollection[exclude[2]]), cmap='gray')
                        # axs[i].imshow(np.squeeze(data['img'][exclude[2]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    else:
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                else:
                    axs[i].imshow(np.squeeze(mycollection[i]), cmap='gray')
                # axs[i].axis('off')
                axs[i].xaxis.set_ticklabels([])
                axs[i].yaxis.set_ticklabels([])
                if(i  in img_idx):
                    if(i == img_idx[0]):
                        tumorBorder = data['tumorBorder'][0]
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][0]]]).title()}',fontsize=12)
                    elif(i == img_idx[1]):
                        tumorBorder = data['tumorBorder'][1]
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][1]]]).title()}',fontsize=12)
                    elif(i == img_idx[2]):
                        tumorBorder = data['tumorBorder'][2]
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][2]]]).title()}',fontsize=12)
                    for j in range(tumorBorder.shape[0]-1):
                        axs[i].scatter(tumorBorder[j],  tumorBorder[j+1],marker=".", color="red", s=200, alpha=0.6,zorder=2)
                fig.tight_layout()
        elif(self.masked and not mri):
            logging.info("We're working with non-mri")
            for i in tqdm(range(int(r*c))):
                if( i in list(range(c))):
                    axs[i].set_title(titles[i],fontsize=12)
                
                # if i not in exclude:
                #     axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                # else:
                #     axs[i].imshow(np.squeeze(mycollection[i]), cmap='gray')
                if i not in exclude:
                    if i in grad_cams_first:
                        axs[i].imshow(np.squeeze(mycollection[exclude[0]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_second:
                        axs[i].imshow(np.squeeze(mycollection[exclude[1]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_third:
                        axs[i].imshow(np.squeeze(mycollection[exclude[2]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    else:
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                else:
                    axs[i].imshow(np.squeeze(mycollection[i]), cmap='gray')
                # axs[i].axis('off')
                axs[i].xaxis.set_ticklabels([])
                axs[i].yaxis.set_ticklabels([])
                if(i  in img_idx):
                    if(i == img_idx[0]):
                        tumorBorder = data['tumorBorder'][0]
                        axs[i].imshow(np.squeeze(tumorBorder), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][0]]]).title()}',fontsize=12)
                    elif(i == img_idx[1]):
                        tumorBorder = data['tumorBorder'][1]
                        axs[i].imshow(np.squeeze(tumorBorder), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][1]]]).title()}',fontsize=12)
                    elif(i == img_idx[2]):
                        tumorBorder = data['tumorBorder'][2]
                        axs[i].imshow(np.squeeze(tumorBorder), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][2]]]).title()}',fontsize=12)
                    
                fig.tight_layout()
        else:
            logging.info("We're working with COVID-19")
            for i in tqdm(range(int(r*c))):
                if( i in list(range(c))):
                    axs[i].set_title(titles[i],fontsize=12)

                # if i not in exclude:
                #     axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                # else:
                #     axs[i].imshow(np.squeeze(mycollection[i]), cmap='gray')
                if i not in exclude:
                    if i in grad_cams_first:
                        axs[i].imshow(np.squeeze(mycollection[exclude[0]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_second:
                        axs[i].imshow(np.squeeze(mycollection[exclude[1]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    elif i in grad_cams_third:
                        axs[i].imshow(np.squeeze(mycollection[exclude[2]]), cmap='gray')
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                    else:
                        axs[i].imshow(np.squeeze(mycollection[i]), cmap='jet', alpha=0.5)
                else:
                    axs[i].imshow(np.squeeze(mycollection[i]), cmap='gray')
                # axs[i].axis('off')
                axs[i].xaxis.set_ticklabels([])
                axs[i].yaxis.set_ticklabels([])
                if(i  in img_idx):
                    if(i == img_idx[0]):
                        img = data['img'][0]
                        # axs[i].imshow(np.squeeze(img), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][0]]]).title()}',fontsize=12)
                    elif(i == img_idx[1]):
                        img = data['img'][0]
                        # axs[i].imshow(np.squeeze(img), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][1]]]).title()}',fontsize=12)
                    elif(i == img_idx[2]):
                        img = data['img'][0]
                        # axs[i].imshow(np.squeeze(img), cmap='jet', alpha=0.4,interpolation='none')
                        axs[i].set_ylabel(f'{str(self.dl.CLASSES[self.dl.y[data["idx"][2]]]).title()}',fontsize=12)
                    
                fig.tight_layout()
        plt.savefig(f'./Figures/{self.dataset}_Saliency_Maps_{self.model.name}.{fmt}', bbox_inches ="tight", dpi=300)
        if show:
            plt.show()
        plt.close(fig)


class Main():
    def __init__(self)-> None:
        pass 
    @staticmethod
    def load_models(file_path = "./Data/Evaluation_Results.csv", n=3 ):
        df =  pd.read_csv(file_path).nlargest(n, 'f1_score')
        models = [] 

        for i in tqdm(range(df.shape[0])):
            models.append(df.iloc[i,1])
        return models
    
if __name__ == "__main__":
    """
    This routine performs saliency analysis using the trained models saved in the ./Models directory and the dataset
    which is loaded using the DataLoad class
    """
    pass 
    # logging.basicConfig(level=logging.INFO)
    
    # dl =  DataLoad()
    # # dl.build(flag=True)
    # dl.load()

    

    # idx = Saliency.pick_indices(dl, n = 1)

    # This loop runs the saliency analyses for the top n performing models

    # models = Main.load_models(n =2)
    # for i in tqdm(range(len(models))):
        
    #     logging.info(f"Loading {models[i]}")
    #     model = tf.keras.models.load_model(f"./Models/{models[i]}")

    #     logging.info("Computing saliency masks, this may take a while...")
    #     sa =  Saliency(model, dl, idx)
    #     mycollection, data, vizresult =  sa.analyze()

    #     logging.info("Plotting heatmaps of relevant features")
    #     sa.plot_results(mycollection, data)
    #     logging.info(f"Saliency analysis completed for {model.name} model")








