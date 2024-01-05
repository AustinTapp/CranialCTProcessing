import vtk
import os
import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

# process example CT image

def Segment(image_path, filename, save_directory):
    isExist = os.path.exists(save_directory)
    if not isExist:
        os.makedirs(save_directory)

    image_name = os.path.join(save_directory, filename.split("_")[0])
    ctImage = sitk.ReadImage(image_path)
    # ctImage.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)) #for latent diffusion code outputs only
    ctImage = DataProcessing.AlignAndRescale(ctImage, sitk.ReadImage('ct.nii.gz'))
    sitk.WriteImage(ctImage, image_name + "_toTemplate.nii.gz")

    binaryImage = DataProcessing.CreateBoneMask(ctImage)
    ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)

    # model
    modelPath = './Model.dat'
    device = ModelConfiguration.getDevice()
    model = ModelConfiguration.adaptModel(modelPath, device)
    imageData = ModelConfiguration.adaptData(ctImage, device)

    landmarks, five_boneLabels, seven_boneLabels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

    sitk.WriteImage(five_boneLabels, image_name + "_5Labeled.nii.gz")
    sitk.WriteImage(seven_boneLabels, image_name + "_7Labeled.nii.gz")

    point_writer = vtk.vtkXMLPolyDataWriter()
    point_writer.SetFileName(image_name + "_landmarks.vtp")
    point_writer.SetInputData(landmarks)
    point_writer.Write()


if __name__ == '__main__':
    data_dir = "E:\\Data\\CNH_Paired\\NoBedCTs"
    save_dir = "E:\\Data\\CNH_Paired\\nbCTsegsV2"
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        try:
            Segment(filepath, filename, save_dir)
        except Exception as e:
            print(f"For case {filename}, an error occurred:", e)
            continue
