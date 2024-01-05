import SimpleITK as sitk
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import torchio as tio
import skimage
import skimage.measure
import warnings

def rescale_intensity_to_reference(reference, input):
    # Get the intensity range of the reference image
    min_ref, max_ref = reference.min(), reference.max()

    # Rescale the input image to match the intensity range of the reference image
    rescaled_array = (input - input.min()) / (input.max() - input.min())
    rescaled_array = rescaled_array * (max_ref - min_ref) + min_ref
    return rescaled_array

def toTemplateRegistration(image, template):
    try:
        RigidElastix = sitk.ElastixImageFilter()

        RigidElastix.SetFixedImage(template)
        RigidElastix.SetMovingImage(image)
        RigidElastix.LogToConsoleOff()

        rigid_map = RigidElastix.ReadParameterFile("Parameters_Rigid.txt")
        RigidElastix.SetParameterMap(rigid_map)

        RigidElastix.Execute()
        TFM_image = RigidElastix.GetResultImage()

        TFM_image_array = sitk.GetArrayFromImage(TFM_image)
        Template_Image_array = sitk.GetArrayFromImage(template)

        Final_Image_array_Rescaled = rescale_intensity_to_reference(Template_Image_array, TFM_image_array)
        Final_Image_Rescaled = sitk.GetImageFromArray(Final_Image_array_Rescaled)

        #spacing = np.array(image.GetSpacing()) * np.array(image.GetSize()) / np.array(template.GetSize())
        #Final_Image_Rescaled.SetSpacing(spacing)

        Final_Image_Rescaled.SetOrigin(template.GetOrigin())
        Final_Image_Rescaled.SetDirection(template.GetDirection())

        sitk.WriteImage(Final_Image_Rescaled, 'CTtoTemplateImage.nii.gz')
        return Final_Image_Rescaled

    except RuntimeError as e:
        warnings.warn(str(e))


def AlignAndRescale(ctImage, template):
    CT_to_template = toTemplateRegistration(ctImage, template)
    return CT_to_template

def FloodFillHull(image):
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img

def CreateHeadMask(ctImage, hounsfieldThreshold = -200):
    """
    Returns a binary image mask of the head from an input CT image

    """

    headMask = sitk.GetArrayFromImage(ctImage)

    # Getting the head
    headMask = (headMask > hounsfieldThreshold).astype(np.uint8)

    headMask = skimage.measure.label(headMask)
    largestLabel = np.argmax(np.bincount(headMask.flat)[1:])+1
    headMask = (headMask == largestLabel).astype(np.uint8)

    headMask = sitk.GetImageFromArray(headMask)
    headMask.SetOrigin(ctImage.GetOrigin())
    headMask.SetSpacing(ctImage.GetSpacing())
    headMask.SetDirection(ctImage.GetDirection())

    return headMask

def CreateBoneMask(ctImage, headMaskImage=None, minimumThreshold=100, maximumThreshold=200, verbose=False):
    """
    Uses adapting thresholding to create a binary mask of the cranial bones from an input CT image.
    [Dangi et al., Robust head CT image registration pipeline for craniosynostosis skull correction surgery, Healthcare Technology Letters, 2017]

    """

    # If a head mask is not provided
    if headMaskImage is None:

        if verbose:
            print('Creating head mask.')

        headMaskImage = CreateHeadMask(ctImage)


    ctImageArray = sitk.GetArrayFromImage(ctImage)
    headMaskImageArray = sitk.GetArrayViewFromImage(headMaskImage)

    # Applying the mask to the CT image
    ctImageArray[headMaskImageArray == 0] = 0

    # Extracting the bones
    minObjects = np.inf
    optimalThreshold = 0
    for threshold in range(minimumThreshold, maximumThreshold+1, 10):

        if verbose:
            print('Optimizing skull segmentation. Threshold {:03d}.'.format(threshold), end='\r')

        labels = skimage.measure.label(ctImageArray >= threshold)
        nObjects = np.max(labels)

        if nObjects < minObjects:
            minObjects = nObjects
            optimalThreshold = threshold
    if verbose:
        print('The optimal threshold for skull segmentation is {:03d}.'.format(optimalThreshold))
    
    ctImageArray = ctImageArray >= optimalThreshold

    ctImageArray = skimage.measure.label(ctImageArray)
    largestLabel = np.argmax(np.bincount(ctImageArray.flat)[1:])+1
    ctImageArray = (ctImageArray == largestLabel).astype(np.uint)
    
    ctImageArray = sitk.GetImageFromArray(ctImageArray)
    ctImageArray.SetOrigin(ctImage.GetOrigin())
    ctImageArray.SetSpacing(ctImage.GetSpacing())
    ctImageArray.SetDirection(ctImage.GetDirection())

    return ctImageArray

def ResampleAndMaskImage(ctImage, binaryImage, outputImageSize = np.array([96, 96, 96], dtype=np.int16)):
    binary = sitk.GetArrayFromImage(binaryImage)
    convexMask = FloodFillHull(binary)
    convexMaskImage = sitk.GetImageFromArray(convexMask)
    convexMaskImage.CopyInformation(binaryImage)
    convexMaskImage = sitk.Cast(convexMaskImage, sitk.sitkUInt32)
    
    normalize = tio.RescaleIntensity(out_min_max = (0, 1), p = 1)
    filter = sitk.MaskImageFilter()
    ctImage = normalize(ctImage)
    ctImage = filter.Execute(ctImage, convexMaskImage)

    templateImageArray = np.zeros(outputImageSize, dtype=np.float32)
    templateImage = sitk.GetImageFromArray(templateImageArray)

    templateImage.SetOrigin(ctImage.GetOrigin())
    spacing = np.array(ctImage.GetSpacing())*np.array(ctImage.GetSize())/outputImageSize
    templateImage.SetSpacing(spacing)

    transform = sitk.AffineTransform(3)
    transform.SetIdentity()
    resampledCTImage = sitk.Resample(ctImage, templateImage, transform.GetInverse(), sitk.sitkLinear)

    return(resampledCTImage)
