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
import tensorflow as tf
import random
from saliencyAnalysis import Saliency,Main
import saliency.core as saliency
from saliency.metrics import pic
from pic import compute_pic_metric, estimate_image_entropy, create_blurred_image,estimate_entropy, estimate_image_avg_entropy,compute_pic_metric_flag  
import cv2
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.scorecam import Scorecam 

# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42) 
np.seterr(divide='ignore', invalid='ignore')

CUSTOM_DATASETS = ["./Data/brainTumorDataPublic", "./Data/COVID-19_Radiography_Dataset"]
# ds = "Covid"
# experiments = "kapis"
FONT_SIZE = 14


def show_image(im, title='', ax=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(12, 6))
  ax.axis('off')
  ax.imshow(im, cmap='gray')
  ax.set_title(title,fontsize=FONT_SIZE)


def show_grayscale_image(im, title='', ax=None):
  if ax is None:
    plt.figure()
  plt.axis('off')

  plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
  plt.title(title, fontsize=FONT_SIZE)


def show_curve_xy(x, y, title='PIC', label=None, color='blue',
    ax=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(12, 6))
  auc = np.trapz(y) / y.size
  label = f'{label}, AUC={auc:.3f}'
  ax.plot(x, y, label=label, color=color)
  ax.set_title(title, fontsize=FONT_SIZE)
  ax.set_xlim([0.0, 1.1])
  ax.set_ylim([-0.1, 1.1])
  ax.set_xlabel('Normalized estimation of entropy', fontsize=FONT_SIZE)
  ax.set_ylabel('Predicted score', fontsize=FONT_SIZE)
  # set the legend including fontsize
  ax.legend(fontsize=FONT_SIZE)
  # add grid
  ax.grid(True)


def show_curve(compute_pic_metric_result, title='PIC', label=None, color='blue',
    ax=None):
  show_curve_xy(compute_pic_metric_result.curve_x,
                compute_pic_metric_result.curve_y, title=title, label=label,
                color=color,
                ax=ax)


def show_blurred_images_with_scores(compute_pic_metric_result):
  # Get model prediction scores.
  images_to_display = compute_pic_metric_result.blurred_images
  scores = compute_pic_metric_result.predictions
  thresholds = compute_pic_metric_result.thresholds

  # Visualize blurred images.
  nrows = (len(images_to_display) - 1) // 5 + 1
  fig, ax = plt.subplots(nrows=nrows, ncols=5,
                         figsize=(20, 20 / 5 * nrows))
  for i in range(len(images_to_display)):
    row = i // 5
    col = i % 5
    title = f'Score: {scores[i]:.3f}\nThreshold: {thresholds[i]:.3f}'
    show_image(images_to_display[i], title=title, ax=ax[row, col])


# Define prediction function.
def create_predict_function_softmax(class_idx, model):
  """Creates the model prediction function that can be passed to compute_pic_metcompute_picric method.

    The function returns the softmax value for the Softmax Information Curve.
  Args:
    class_idx: the index of the class for which the model prediction should
      be returned.
  """

  def predict(image_batch):
    """Returns model prediction for a batch of images.

    The method receives a batch of images in uint8 format. The method is responsible to
    convert the batch to whatever format required by the model. In this particular
    implementation the conversion is achieved by calling preprocess_input().

    Args:
      image_batch: batch of images of dimension [B, H, W, C].

    Returns:
      Predictions of dimension [B].
    """
    # image_batch = tf.keras.applications.vgg16.preprocess_input(image_batch)
    score = model(image_batch)[:, class_idx]
    return score.numpy()

  return predict



# Define prediction function.
def create_predict_function_accuracy(class_idx, model):
  """Creates the model prediction function that can be passed to compute_pic_metric method.

    The function returns the accuracy for the Accuracy Information Curve.

  Args:
    class_idx: the index of the class for which the model prediction score should
      be returned.
  """

  def predict(image_batch):
    """Returns model accuracy for a batch of images.

    The method receives a batch of images in uint8 format. The method is responsible to
    convert the batch to whatever format required by the model. In this particular
    implementation the conversion is achieved by calling preprocess_input().

    Args:
      image_batch: batch of images of dimension [B, H, W, C].

    Returns:
      Predictions of dimension [B], where every element is either 1.0 for correct
      prediction or 0.0 for incorrect prediction.
    """
    # image_batch = tf.keras.applications.vgg16.preprocess_input(image_batch)
    scores = model(image_batch)
    arg_max = np.argmax(scores, axis=1)
    accuracy = arg_max == class_idx
    return np.ones_like(arg_max) * accuracy

  return predict


def compute_pic_score(dl,model_indx, masked = True, ds = "BrainTumor", experiments = "base", n_samples = 1000):
    models = Main.load_models(n =3, file_path=f"./Data/{ds}_Evaluation_Results.csv")
    model = tf.keras.models.load_model(f"./Models/{ds}_{models[model_indx]}.keras")

    logging.info("Computing saliency masks, this may take a while...")
    idxs = Saliency.pick_indices(dl, n = 3)
    sa =  Saliency(model, dl, idxs, masked=masked, dataset=ds)

    
    # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    # We use Guided Integrated Gradients as an example saliency (see https://arxiv.org/abs/2106.09788).
    guided_ig = saliency.GuidedIG()
    
    # Define saliency thresholds
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]
    gig_aic_individual_results = []
    rnd_aic_individual_results = []
    gig_sic_individual_results = []
    rnd_sic_individual_results = []
    vanilla_aic_individual_results = []
    smoothgrad_aic_individual_results = []
    vanilla_sic_individual_results = []
    smoothgrad_sic_individual_results = []
    xrai_aic_individual_results = []
    xrai_sic_individual_results = []
    gradcam_aic_individual_results = []
    gradcam_sic_individual_results = []
    gradcamplusplus_aic_individual_results = []
    gradcamplusplus_sic_individual_results = []
    scorecam_aic_individual_results = []
    scorecam_sic_individual_results = []

    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()
 

    for idx in range(len(dl.X_)):
        im_orig = dl.X_[idx]
        
        replace2linear = ReplaceToLinear()
        score = CategoricalScore([dl.y[idx]])
        
        predictions,prediction_class,call_model_args,pred_prob = sa.predict(im_orig)

        # Create a random mask for the initial fully blurred image.
        random_mask = pic.generate_random_mask(image_height=im_orig.shape[0],
                                        image_width=im_orig.shape[1],
                                        fraction=0.01)
        
        # Estimate entropy of the completely blurred image.
        fully_blurred_img = create_blurred_image(full_img=im_orig, pixel_mask=random_mask)
        # Estimate entropy of the original image.
        if experiments == "base":
            original_img_entropy = estimate_image_entropy(im_orig)
            fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)
        elif experiments == "kapis":
          original_img_entropy = estimate_entropy(im_orig)
          fully_blurred_img_entropy = estimate_entropy(fully_blurred_img)
        else:
            original_img_entropy = estimate_image_avg_entropy(im_orig)
            fully_blurred_img_entropy = estimate_image_avg_entropy(fully_blurred_img)

        pred_func_sic = create_predict_function_softmax(class_idx=prediction_class, model=model)

        pred = pred_func_sic(im_orig[np.newaxis, ...])
        pred_blurred = pred_func_sic(fully_blurred_img[np.newaxis, ...])

        # We want to compute PIC only if the prediction of the original image is higher than the prediction of the blurred image.
        # And the entropy of the original image is higher than the entropy of the fully blurred image.
        if pred > pred_blurred:
           if original_img_entropy > fully_blurred_img_entropy:
                baseline = np.zeros(im_orig.shape)



                # Compute the Guided IG saliency.
                guided_ig_mask_3d = guided_ig.GetMask(
                    im_orig, sa.call_model_function, call_model_args, x_steps=25, x_baseline=baseline,
                    max_dist=1.0, fraction=0.5)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)


                # Construct the saliency object. This alone doesn't do anthing.
                integrated_gradients = saliency.IntegratedGradients()
                
                # Compute the vanilla mask and the smoothed mask.
                vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                        im_orig, sa.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
                # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
                smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
                        im_orig, sa.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
                smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

                # Construct the saliency object. This alone doesn't do anthing.
                xrai_object = saliency.XRAI()

                # Compute XRAI attributions with default parameters
                xrai_attributions = xrai_object.GetMask(im_orig, sa.call_model_function, call_model_args, batch_size=20)

                # Show most salient 30% of the image
                mask = xrai_attributions > np.percentile(xrai_attributions, 70)
                im_mask =  np.array(im_orig)
                im_mask[~mask] = 0
                

                # Create Gradcam object
                gradcam = Gradcam(model,
                                model_modifier=replace2linear,
                                clone=True)

                # Generate heatmap with GradCAM
                cam = gradcam(score,im_orig,penultimate_layer=-1)

                gradcam_saliency_map = normalize(cam)

                # Create GradCAM++ object
                gradcam = GradcamPlusPlus(model,
                                        model_modifier=replace2linear,
                                        clone=True)
                
                # Generate heatmap with GradCAM++
                gradcamplusplus = gradcam(score,
                            im_orig,
                            penultimate_layer=-1)

                cam = normalize(gradcamplusplus)

                gradcamplusplus_saliency_map = np.expand_dims(np.squeeze(cam), axis=2)
                
                # squeeze and expand the channel dimension
                gradcam_saliency_map = np.expand_dims(np.squeeze(gradcam_saliency_map), axis=2)

                
                # Create ScoreCAM object
                scorecam = Scorecam(model)

                # Generate heatmap with ScoreCAM
                scorecam = scorecam(score, im_orig, penultimate_layer=-1)

                cam = normalize(scorecam)

                scorecam_saliency_map = np.expand_dims(np.squeeze(cam), axis=2)
           

                # print(guided_ig_mask_3d.shape, vanilla_integrated_gradients_mask_3d.shape, smoothgrad_integrated_gradients_mask_3d.shape, xrai_attributions.shape)
                # check if color channel is missing in xrai attributions add it
                if len(xrai_attributions.shape) == 2:
                    xrai_attributions = np.expand_dims(xrai_attributions, axis=2)
                # Softmax Information Curve (SIC)
                gig_saliency_map = np.abs(np.sum(guided_ig_mask_3d, axis=2))
                vanilla_saliency_map = np.abs(np.sum(vanilla_integrated_gradients_mask_3d, axis=2))
                smoothgrad_saliency_map = np.abs(np.sum(smoothgrad_integrated_gradients_mask_3d, axis=2))
                xrai_saliency_map = np.abs(np.sum(xrai_attributions, axis=2))
                gradcam_saliency_map = np.abs(np.sum(gradcam_saliency_map, axis=2))
                gradcamplusplus_saliency_map = np.abs(np.sum(gradcamplusplus_saliency_map, axis=2))
                scorecam_saliency_map = np.abs(np.sum(scorecam_saliency_map, axis=2))

                sic_flag =  compute_pic_metric_flag(
                    img=im_orig,
                    random_mask=random_mask,
                    pred_func=pred_func_sic,
                    experiment=experiments
                )

                print("Current condition is ", sic_flag)

                # # Softmax Information Curve (SIC)
                gig_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=gig_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                vanilla_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=vanilla_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                smoothgrad_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=smoothgrad_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                
                xrai_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=xrai_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                
                gradcam_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=gradcam_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                gradcamplusplus_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=gradcamplusplus_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                scorecam_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=scorecam_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)

                # # For comparison, compute PIC for random saliency map.
                rnd_saliency_map = np.random.random(size=(im_orig.shape[0], im_orig.shape[1]))
                rnd_result_sic = compute_pic_metric(img=im_orig,
                                                    saliency_map=rnd_saliency_map,
                                                    random_mask=random_mask,
                                                    pred_func=pred_func_sic,
                                                    min_pred_value=0.5,
                                                    saliency_thresholds=saliency_thresholds,
                                                    keep_monotonous=True,
                                                    num_data_points=1000,
                                                    experiment=experiments)
                gig_sic_individual_results.append(gig_result_sic)
                rnd_sic_individual_results.append(rnd_result_sic)
                vanilla_sic_individual_results.append(vanilla_result_sic)
                smoothgrad_sic_individual_results.append(smoothgrad_result_sic)
                xrai_sic_individual_results.append(xrai_result_sic)
                gradcam_sic_individual_results.append(gradcam_result_sic)
                gradcamplusplus_sic_individual_results.append(gradcamplusplus_result_sic)
                scorecam_sic_individual_results.append(scorecam_result_sic)
                

                # # Accuracy Information Curve (AIC)
                pred_func_accuracy = create_predict_function_accuracy(prediction_class, model)

                aic_flag =  compute_pic_metric_flag(
                    img=im_orig,
                    random_mask=random_mask,
                    pred_func=pred_func_accuracy,
                    experiment=experiments
                )

                print("Current condition is ", aic_flag)
                if aic_flag:

                  gig_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=gig_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  vanilla_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=vanilla_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000)
                  
                  smoothgrad_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=smoothgrad_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)

                  # # For comparison, compute PIC for random saliency map.
                  rnd_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=rnd_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  xrai_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=xrai_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  gradcam_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=gradcam_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  gradcamplusplus_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=gradcamplusplus_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  scorecam_result_aic = compute_pic_metric(img=im_orig,
                                                      saliency_map=scorecam_saliency_map,
                                                      random_mask=random_mask,
                                                      pred_func=pred_func_accuracy,
                                                      min_pred_value=0.5,
                                                      saliency_thresholds=saliency_thresholds,
                                                      keep_monotonous=True,
                                                      num_data_points=1000,
                                                      experiment=experiments)
                  gig_aic_individual_results.append(gig_result_aic)
                  rnd_aic_individual_results.append(rnd_result_aic)
                  vanilla_aic_individual_results.append(vanilla_result_aic)
                  xrai_aic_individual_results.append(xrai_result_aic)
                  smoothgrad_aic_individual_results.append(smoothgrad_result_aic)
                  gradcam_aic_individual_results.append(gradcam_result_aic)
                  gradcamplusplus_aic_individual_results.append(gradcamplusplus_result_aic)
                  scorecam_aic_individual_results.append(scorecam_result_aic)



                print(len(gig_aic_individual_results), len(rnd_aic_individual_results))
        #1300
        if len(gig_aic_individual_results) >= n_samples:
            break
    return gig_aic_individual_results, rnd_aic_individual_results, gig_sic_individual_results, rnd_sic_individual_results, \
        vanilla_aic_individual_results, smoothgrad_aic_individual_results, vanilla_sic_individual_results, \
        smoothgrad_sic_individual_results, xrai_aic_individual_results, xrai_sic_individual_results, gradcam_aic_individual_results, \
        gradcam_sic_individual_results, gradcamplusplus_aic_individual_results, gradcamplusplus_sic_individual_results, scorecam_aic_individual_results, scorecam_sic_individual_results
                


@click.command()
@click.option("--ds",default="BrainTumor", help="The dataset to use for training the model.")
@click.option("--experiments",default="base", help="The experiment to run.")
@click.option("--n_samples",default=1000, help="The number of datapoints to run the experiment on.")
def main(ds, experiments, n_samples):
  if ds == "BrainTumor":
    dl =  DataLoad()

    dl.load()
    model_indx = 0

    gig_aic_individual_results, rnd_aic_individual_results, gig_sic_individual_results, rnd_sic_individual_results, \
    vanilla_aic_individual_results, smoothgrad_aic_individual_results, vanilla_sic_individual_results, \
    smoothgrad_sic_individual_results, xrai_aic_individual_results, xrai_sic_individual_results, gradcam_aic_individual_results, \
    gradcam_sic_individual_results, gradcamplusplus_aic_individual_results, gradcamplusplus_sic_individual_results, scorecam_aic_individual_results, scorecam_sic_individual_results = compute_pic_score(dl,model_indx, masked = True, ds = "BrainTumor", experiments = experiments, n_samples = n_samples)
    print(len(gig_aic_individual_results), len(rnd_aic_individual_results))
    gig_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gig_aic_individual_results, method='median')
    rnd_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=rnd_aic_individual_results, method='median')
    vanilla_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=vanilla_aic_individual_results, method='median')
    smoothgrad_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=smoothgrad_aic_individual_results, method='median')
    xrai_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=xrai_aic_individual_results, method='median')
    gradcam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcam_aic_individual_results, method='median')
    gradcamplusplus_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcamplusplus_aic_individual_results, method='median')
    scorecam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=scorecam_aic_individual_results, method='median')

    # save these aggregated results as pandas csv
    data = {'Guided IG': gig_agg_result, 'Random': rnd_agg_result, 'Vanilla IG': vanilla_agg_result, 'SmoothGrad IG': smoothgrad_agg_result, 'XRAI': xrai_agg_result, 'GradCAM': gradcam_agg_result, 'GradCAM++': gradcamplusplus_agg_result, 'ScoreCAM': scorecam_agg_result}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_AIC_Evaluation_Results_TEMP.csv")
    # Save the original results as well
    data = {'Guided IG': gig_aic_individual_results, 'Random': rnd_aic_individual_results, 'Vanilla IG': vanilla_aic_individual_results, 'SmoothGrad IG': smoothgrad_aic_individual_results, 'XRAI': xrai_aic_individual_results, 'GradCAM': gradcam_aic_individual_results, 'GradCAM++': gradcamplusplus_aic_individual_results, 'ScoreCAM': scorecam_aic_individual_results}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_AIC_Individual_Results.csv")


    # Plot the aggregated results.
    fig, ax = plt.subplots(figsize=(12, 6))
    title = "Aggregated Accuracy Information Curve (AIC)"
    show_curve(gig_agg_result, title=f'{title}', label='Guided IG', color='blue', ax=ax)
    show_curve(rnd_agg_result, title=f'{title}', label='Random', color='red', ax=ax)
    show_curve(vanilla_agg_result, title=f'{title}', label='Vanilla IG', color='green', ax=ax)
    show_curve(smoothgrad_agg_result, title=f'{title}', label='SmoothGrad IG', color='orange', ax=ax)
    show_curve(xrai_agg_result, title=f'{title}', label='XRAI', color='purple', ax=ax)
    show_curve(gradcam_agg_result, title=f'{title}', label='GradCAM', color='black', ax=ax)
    show_curve(gradcamplusplus_agg_result, title=f'{title}', label='GradCAM++', color='brown', ax=ax)
    show_curve(scorecam_agg_result, title=f'{title}', label='ScoreCAM', color='pink', ax=ax)
    plt.savefig(f'./Figures/{ds}_{experiments}_AIC_Aggregated.tiff', format='tiff', dpi=300)
    plt.close(fig)

    gig_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gig_sic_individual_results, method='median')
    rnd_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=rnd_sic_individual_results, method='median')
    vanilla_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=vanilla_sic_individual_results, method='median')
    smoothgrad_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=smoothgrad_sic_individual_results, method='median')
    xrai_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=xrai_sic_individual_results, method='median')
    gradcam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcam_sic_individual_results, method='median')
    gradcamplusplus_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcamplusplus_sic_individual_results, method='median')
    scorecam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=scorecam_sic_individual_results, method='median')

    # save these aggregated results as pandas csv
    data = {'Guided IG': gig_agg_result, 'Random': rnd_agg_result, 'Vanilla IG': vanilla_agg_result, 'SmoothGrad IG': smoothgrad_agg_result, 'XRAI': xrai_agg_result, 'GradCAM': gradcam_agg_result, 'GradCAM++': gradcamplusplus_agg_result, 'ScoreCAM': scorecam_agg_result}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_{experiments}_SIC_Evaluation_Results.csv")
    # Save the original results as well
    data = {'Guided IG': gig_sic_individual_results, 'Random': rnd_sic_individual_results, 'Vanilla IG': vanilla_sic_individual_results, 'SmoothGrad IG': smoothgrad_sic_individual_results, 'XRAI': xrai_sic_individual_results, 'GradCAM': gradcam_sic_individual_results, 'GradCAM++': gradcamplusplus_sic_individual_results, 'ScoreCAM': scorecam_sic_individual_results}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_{experiments}_SIC_Individual_Results.csv")

    # want to plot the mean and deviation from it for each method



    # Plot the aggregated results.
    fig, ax = plt.subplots(figsize=(12, 6))
    title = "Aggregated Softmax Information Curve (SIC)"
    show_curve(gig_agg_result, title=f'{title}', label='Guided IG', color='blue', ax=ax)
    show_curve(rnd_agg_result, title=f'{title}', label='Random', color='red', ax=ax)
    show_curve(vanilla_agg_result, title=f'{title}', label='Vanilla IG', color='green', ax=ax)
    show_curve(smoothgrad_agg_result, title=f'{title}', label='SmoothGrad IG', color='orange', ax=ax)
    show_curve(xrai_agg_result, title=f'{title}', label='XRAI', color='purple', ax=ax)
    show_curve(gradcam_agg_result, title=f'{title}', label='GradCAM', color='black', ax=ax)
    show_curve(gradcamplusplus_agg_result, title=f'{title}', label='GradCAM++', color='brown', ax=ax)
    show_curve(scorecam_agg_result, title=f'{title}', label='ScoreCAM', color='pink', ax=ax)
    plt.savefig(f'./Figures/{ds}_{experiments}_SIC_Aggregated.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gig_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GIG_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gig_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GIG_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    # Do the same for the others 
    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(vanilla_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_Vanilla_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(vanilla_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_Vanilla_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(smoothgrad_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_SmoothGrad_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(smoothgrad_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_SmoothGrad_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(xrai_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_XRAI_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(xrai_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_XRAI_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcam_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAM_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcam_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAM_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcamplusplus_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAMPlusPlus_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcamplusplus_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAMPlusPlus_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(scorecam_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_ScoreCAM_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)
  
  elif ds == "Covid":
    dl =  DataLoad(datapath = Path(f"{CUSTOM_DATASETS[1]}"),outdir = f'{CUSTOM_DATASETS[1]}.npz', segmented=False)
    dl.load(masked=False)

    model_indx = 0

    gig_aic_individual_results, rnd_aic_individual_results, gig_sic_individual_results, rnd_sic_individual_results, \
    vanilla_aic_individual_results, smoothgrad_aic_individual_results, vanilla_sic_individual_results, \
    smoothgrad_sic_individual_results, xrai_aic_individual_results, xrai_sic_individual_results, gradcam_aic_individual_results, \
    gradcam_sic_individual_results, gradcamplusplus_aic_individual_results, gradcamplusplus_sic_individual_results, scorecam_aic_individual_results, scorecam_sic_individual_results = compute_pic_score(dl,model_indx, masked = True, ds = "BrainTumor")
    print(len(gig_aic_individual_results), len(rnd_aic_individual_results))
    gig_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gig_aic_individual_results, method='median')
    rnd_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=rnd_aic_individual_results, method='median')
    vanilla_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=vanilla_aic_individual_results, method='median')
    smoothgrad_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=smoothgrad_aic_individual_results, method='median')
    xrai_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=xrai_aic_individual_results, method='median')
    gradcam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcam_aic_individual_results, method='median')
    gradcamplusplus_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcamplusplus_aic_individual_results, method='median')
    scorecam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=scorecam_aic_individual_results, method='median')

    # save these aggregated results as pandas csv
    data = {'Guided IG': gig_agg_result, 'Random': rnd_agg_result, 'Vanilla IG': vanilla_agg_result, 'SmoothGrad IG': smoothgrad_agg_result, 'XRAI': xrai_agg_result, 'GradCAM': gradcam_agg_result, 'GradCAM++': gradcamplusplus_agg_result, 'ScoreCAM': scorecam_agg_result}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_{experiments}_AIC_Evaluation_Results.csv")
    

    # Plot the aggregated results.
    fig, ax = plt.subplots(figsize=(12, 6))
    title = "Aggregated Accuracy Information Curve (AIC)"
    show_curve(gig_agg_result, title=f'{title}', label='Guided IG', color='blue', ax=ax)
    show_curve(rnd_agg_result, title=f'{title}', label='Random', color='red', ax=ax)
    show_curve(vanilla_agg_result, title=f'{title}', label='Vanilla IG', color='green', ax=ax)
    show_curve(smoothgrad_agg_result, title=f'{title}', label='SmoothGrad IG', color='orange', ax=ax)
    show_curve(xrai_agg_result, title=f'{title}', label='XRAI', color='purple', ax=ax)
    show_curve(gradcam_agg_result, title=f'{title}', label='GradCAM', color='black', ax=ax)
    show_curve(gradcamplusplus_agg_result, title=f'{title}', label='GradCAM++', color='brown', ax=ax)
    show_curve(scorecam_agg_result, title=f'{title}', label='ScoreCAM', color='pink', ax=ax)
    plt.savefig(f'./Figures/{ds}_{experiments}_AIC_Aggregated.tiff', format='tiff', dpi=300)
    plt.close(fig)

    gig_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gig_sic_individual_results, method='median')
    rnd_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=rnd_sic_individual_results, method='median')
    vanilla_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=vanilla_sic_individual_results, method='median')
    smoothgrad_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=smoothgrad_sic_individual_results, method='median')
    xrai_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=xrai_sic_individual_results, method='median')
    gradcam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcam_sic_individual_results, method='median')
    gradcamplusplus_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=gradcamplusplus_sic_individual_results, method='median')
    scorecam_agg_result = pic.aggregate_individual_pic_results(compute_pic_metrics_results=scorecam_sic_individual_results, method='median')

    # save these aggregated results as pandas csv
    data = {'Guided IG': gig_agg_result, 'Random': rnd_agg_result, 'Vanilla IG': vanilla_agg_result, 'SmoothGrad IG': smoothgrad_agg_result, 'XRAI': xrai_agg_result, 'GradCAM': gradcam_agg_result, 'GradCAM++': gradcamplusplus_agg_result, 'ScoreCAM': scorecam_agg_result}
    df = pd.DataFrame(data)
    df.to_csv(f"./Data/{ds}_{experiments}_SIC_Evaluation_Results.csv")


    # Plot the aggregated results.
    fig, ax = plt.subplots(figsize=(12, 6))
    title = "Aggregated Softmax Information Curve (SIC)"
    show_curve(gig_agg_result, title=f'{title}', label='Guided IG', color='blue', ax=ax)
    show_curve(rnd_agg_result, title=f'{title}', label='Random', color='red', ax=ax)
    show_curve(vanilla_agg_result, title=f'{title}', label='Vanilla IG', color='green', ax=ax)
    show_curve(smoothgrad_agg_result, title=f'{title}', label='SmoothGrad IG', color='orange', ax=ax)
    show_curve(xrai_agg_result, title=f'{title}', label='XRAI', color='purple', ax=ax)
    show_curve(gradcam_agg_result, title=f'{title}', label='GradCAM', color='black', ax=ax)
    show_curve(gradcamplusplus_agg_result, title=f'{title}', label='GradCAM++', color='brown', ax=ax)
    show_curve(scorecam_agg_result, title=f'{title}', label='ScoreCAM', color='pink', ax=ax)
    plt.savefig(f'./Figures/{ds}_{experiments}_SIC_Aggregated.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gig_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GIG_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gig_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GIG_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    # Do the same for the others 
    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(vanilla_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_Vanilla_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(vanilla_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_Vanilla_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(smoothgrad_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_SmoothGrad_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(smoothgrad_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_SmoothGrad_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(xrai_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_XRAI_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(xrai_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_XRAI_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcam_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAM_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcam_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAM_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcamplusplus_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAMPlusPlus_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(gradcamplusplus_sic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_GradCAMPlusPlus_Blurred_SIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_blurred_images_with_scores(scorecam_aic_individual_results[0])
    plt.savefig(f'./Figures/{ds}_{experiments}_ScoreCAM_Blurred_AIC_Images.tiff', format='tiff', dpi=300)
    plt.close(fig)
  
  else:
    logging.info("Invalid dataset selected, please try again")
if __name__ == "__main__":
    main()