import io
import numpy as np
import os
from PIL import Image
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
import skimage.measure
from scipy import stats    

def estimate_image_avg_entropy(image: np.ndarray) -> float:
    entropy = (estimate_entropy(image) + estimate_image_entropy(image))/2.0
    return entropy


def estimate_image_entropy(im_orig: np.ndarray) -> float:
    p = np.histogram(im_orig, bins=256, range=(im_orig.min(), im_orig.max()), density=True)[0]
    im_entropy = stats.entropy(p, base=2)
    return im_entropy

def estimate_entropy(image: np.ndarray) -> float:
    """Estimates the amount of information in a given image.

    Args:
      image: an image, which entropy should be estimated. The dimensions of the
        array should be [H, W] or [H, W, C] of type uint8 or float.
    Returns:
      The estimated amount of information in the image.
    """
    # Normalize the image to range [0, 255] if it's in float format
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2:  # Single-channel image
        pil_image = Image.fromarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 1:  # Single-channel image as [H, W, 1]
        pil_image = Image.fromarray(image[:, :, 0])
    else:  # Multi-channel image
        pil_image = Image.fromarray(image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format='webp', lossless=True, quality=100)
    buffer.seek(0, os.SEEK_END)
    length = buffer.tell()
    buffer.close()
    
    return length

def create_blurred_image(full_img: np.ndarray, pixel_mask: np.ndarray,
    method: str = 'linear') -> np.ndarray:
  """ Creates a blurred (interpolated) image.

  Args:
    full_img: an original input image that should be used as the source for
      interpolation. The image should be represented by a numpy array with
      dimensions [H, W, C] or [H, W].
    pixel_mask: a binary mask, where 'True' values represent pixels that should
      be retrieved from the original image as the source for the interpolation
      and 'False' values represent pixels, which values should be found. The
      method always sets the corner pixels of the mask to True. The mask
      dimensions should be [H, W].
    method: the method to use for the interpolation. The 'linear' method is
      recommended. The alternative value is 'nearest'.

    Returns:
      A numpy array that encodes the blurred image with exactly the same
      dimensions and type as `full_img`.
  """
  data_type = full_img.dtype
  has_color_channel = full_img.ndim > 2
  if not has_color_channel:
    full_img = np.expand_dims(full_img, axis=2)
  channels = full_img.shape[2]

  # Always include corners.
  pixel_mask = pixel_mask.copy()
  height = pixel_mask.shape[0]
  width = pixel_mask.shape[1]
  pixel_mask[
    [0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]] = True

  mean_color = np.mean(full_img, axis=(0, 1))

  # If the mask consists of all pixels set to True then return the original
  # image.
  if np.all(pixel_mask):
    return full_img

  blurred_img = full_img * np.expand_dims(pixel_mask, axis=2).astype(
      np.float32)

  # Interpolate the unmasked values of the image pixels.
  for channel in range(channels):
    data_points = np.argwhere(pixel_mask > 0)
    data_values = full_img[:, :, channel][tuple(data_points.T)]
    unknown_points = np.argwhere(pixel_mask == 0)
    interpolated_values = interpolate.griddata(np.array(data_points),
                                               np.array(data_values),
                                               np.array(unknown_points),
                                               method=method,
                                               fill_value=mean_color[channel])
    blurred_img[:, :, channel][tuple(unknown_points.T)] = interpolated_values

  if not has_color_channel:
    blurred_img = blurred_img[:, :, 0]

  if issubclass(data_type.type, np.integer):
    blurred_img = np.round(blurred_img)

  return blurred_img.astype(data_type)


class PicMetricResult(NamedTuple):
  """Holds results of compute_pic_metric(...) method."""
  # x-axis coordinates of PIC curve data points.
  curve_x: Sequence[float]
  # y-axis coordinates of PIC curve data points.
  curve_y: Sequence[float]
  # A sequence of intermediate blurred images used for PIC computation with
  # the fully blurred image in front and the original image at the end.
  blurred_images: Sequence[np.ndarray]
  # Model predictions for images in the `blurred_images` sequence.
  predictions: Sequence[float]
  # Saliency thresholds that were used to generate corresponding
  # `blurred_images`.
  thresholds: Sequence[float]
  # Area under the curve.
  auc: float

def compute_pic_metric_flag(
    img: np.ndarray,
    random_mask: np.ndarray,
    pred_func: Callable[[np.ndarray], Sequence[float]],
    experiment: Optional[str] = "base",
):
  blurred_images = []
  predictions = []


  # Estimate entropy of the completely blurred image.
  fully_blurred_img = create_blurred_image(full_img=img, pixel_mask=random_mask)

  # Estimate entropy of the original image.
  if experiment == "base":
    original_img_entropy = estimate_image_entropy(img)
    fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)
  
  elif experiment == "kapis":
    original_img_entropy = estimate_entropy(img)
    fully_blurred_img_entropy = estimate_entropy(fully_blurred_img)
  else:
    original_img_entropy = estimate_image_avg_entropy(img)
    fully_blurred_img_entropy = estimate_image_avg_entropy(fully_blurred_img)
  

  # Compute model prediction for the original image.
  original_img_pred = pred_func(img[np.newaxis, ...])[0]
  
  # print(f'Original image entropy: {original_img_entropy} Entropy of the completely blurred image: {fully_blurred_img_entropy}')

  # if original_img_pred < min_pred_value:
  #   print(f'Original image prediction is below the threshold: {original_img_pred}')

  # Compute model prediction for the completely blurred image.
  fully_blurred_img_pred = pred_func(fully_blurred_img[np.newaxis, ...])[0]
  
  blurred_images.append(fully_blurred_img)
  predictions.append(fully_blurred_img_pred)

    # If the entropy of the completely blurred image is higher or equal to the
  # entropy of the original image then the metric cannot be used for this
  # image. Don't include this image in the aggregated result.
  if original_img_entropy >= fully_blurred_img_entropy and original_img_pred >= fully_blurred_img_pred:
    return True
  else:
    print(f'Fully blurred image entropy is higher or equal to the original image entropy: {fully_blurred_img_entropy}')
    print(
        'The model prediction score on the completely blurred image is not'
        ' lower than the score on the original image. Catch the error and'
        ' exclude this image from the evaluation. Blurred score: {}, original'
        ' score {}'.format(fully_blurred_img_pred, original_img_pred))
    return False
    

def compute_pic_metric(
    img: np.ndarray,
    saliency_map: np.ndarray,
    random_mask: np.ndarray,
    pred_func: Callable[[np.ndarray], Sequence[float]],
    saliency_thresholds: Sequence[float],
    min_pred_value: float = 0.8,
    keep_monotonous: bool = True,
    num_data_points: int = 1000,
    experiment: Optional[str] = "base",
):
  blurred_images = []
  predictions = []

  # This list will contain mapping of image entropy for a given saliency
  # threshold to model prediction.
  entropy_pred_tuples = []

  # Estimate entropy of the completely blurred image.
  fully_blurred_img = create_blurred_image(full_img=img, pixel_mask=random_mask)

  # Estimate entropy of the original image.
  if experiment == "base":
    original_img_entropy = estimate_image_entropy(img)
    fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)
  
  elif experiment == "kapis":
    original_img_entropy = estimate_entropy(img)
    fully_blurred_img_entropy = estimate_entropy(fully_blurred_img)
  else:
    original_img_entropy = estimate_image_avg_entropy(img)
    fully_blurred_img_entropy = estimate_image_avg_entropy(fully_blurred_img)
  

  # Compute model prediction for the original image.
  original_img_pred = pred_func(img[np.newaxis, ...])[0]
  
  # print(f'Original image entropy: {original_img_entropy} Entropy of the completely blurred image: {fully_blurred_img_entropy}')

  # if original_img_pred < min_pred_value:
  #   print(f'Original image prediction is below the threshold: {original_img_pred}')

  # Compute model prediction for the completely blurred image.
  fully_blurred_img_pred = pred_func(fully_blurred_img[np.newaxis, ...])[0]
  
  blurred_images.append(fully_blurred_img)
  predictions.append(fully_blurred_img_pred)

    # If the entropy of the completely blurred image is higher or equal to the
  # entropy of the original image then the metric cannot be used for this
  # image. Don't include this image in the aggregated result.
  if fully_blurred_img_entropy >= original_img_entropy:
    print(f'Fully blurred image entropy is higher or equal to the original image entropy: {fully_blurred_img_entropy}')

  # If the score of the model on completely blurred image is higher or equal to
  # the score of the model on the original image then the metric cannot be used
  # for this image. Don't include this image in the aggregated result.
  if fully_blurred_img_pred >= original_img_pred:
    print(
        'The model prediction score on the completely blurred image is not'
        ' lower than the score on the original image. Catch the error and'
        ' exclude this image from the evaluation. Blurred score: {}, original'
        ' score {}'.format(fully_blurred_img_pred, original_img_pred))
  # Iterate through saliency thresholds and compute prediction of the model
  # for the corresponding blurred images with the saliency pixels revealed.
  max_normalized_pred = 0.0
  for threshold in saliency_thresholds:
    quantile = np.quantile(saliency_map, 1 - threshold)
    pixel_mask = saliency_map >= quantile
    pixel_mask = np.logical_or(pixel_mask, random_mask)
    blurred_image = create_blurred_image(full_img=img, pixel_mask=pixel_mask)
    if experiment == "base":
      entropy = estimate_image_entropy(blurred_image)
    elif experiment == "kapis":
      entropy = estimate_entropy(blurred_image)
    else:
      entropy = estimate_image_avg_entropy(blurred_image)
    # entropy = estimate_image_entropy(blurred_image)
    # entropy = estimate_entropy(blurred_image)
    pred = pred_func(blurred_image[np.newaxis, ...])[0]
    # Normalize the values, so they lie in [0, 1] interval.
    normalized_entropy = (entropy - fully_blurred_img_entropy) / (
        original_img_entropy - fully_blurred_img_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    normalized_pred = (pred - fully_blurred_img_pred) / (
        original_img_pred - fully_blurred_img_pred)
    normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
    max_normalized_pred = max(max_normalized_pred, normalized_pred)

    # Make normalized_pred only grow if keep_monotonous is true.
    if keep_monotonous:
      entropy_pred_tuples.append((normalized_entropy, max_normalized_pred))
    else:
      entropy_pred_tuples.append((normalized_entropy, normalized_pred))

    blurred_images.append(blurred_image)
    predictions.append(pred)

  # Interpolate the PIC curve.
  entropy_pred_tuples.append((0.0, 0.0))
  entropy_pred_tuples.append((1.0, 1.0))

  entropy_data, pred_data = zip(*entropy_pred_tuples)
  interp_func = interpolate.interp1d(x=entropy_data, y=pred_data)

  curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points,
                        endpoint=False)
  curve_y = np.asarray([interp_func(x) for x in curve_x])

  curve_x = np.append(curve_x, 1.0)
  curve_y = np.append(curve_y, 1.0)

  auc = np.trapz(curve_y, curve_x)

  blurred_images.append(img)
  predictions.append(original_img_pred)

  thresholds = [0.0] + list(saliency_thresholds) + [1.0]
  
#   return entropy_pred_tuples, fully_blurred_img
  return PicMetricResult(curve_x=curve_x, curve_y=curve_y,
                         blurred_images=blurred_images,
                         predictions=predictions, thresholds=thresholds,
                         auc=auc)

