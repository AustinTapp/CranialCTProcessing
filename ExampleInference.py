import DataProcessing
import SimpleITK as sitk
import ModelConfiguration
import vtk

### process example CT image

ctImage = sitk.ReadImage('ct.nii.gz')
#ctImage.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)) #for latent diffusion code outputs only
ctImage = DataProcessing.AlignAndRescale(ctImage, sitk.ReadImage('ct.nii.gz'))
binaryImage = DataProcessing.CreateBoneMask(ctImage)
ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)

### model
modelPath = './Model.dat'
device = ModelConfiguration.getDevice()
model = ModelConfiguration.adaptModel(modelPath, device)
imageData = ModelConfiguration.adaptData(ctImage, device)

landmarks, five_boneLabels, seven_boneLabels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)

point_writer = vtk.vtkXMLPolyDataWriter()
point_writer.SetFileName("landmarks.vtp")
point_writer.SetInputData(landmarks)
point_writer.Write()

sitk.WriteImage(five_boneLabels, 'labels5.nii.gz')
sitk.WriteImage(seven_boneLabels, 'labels7.nii.gz')
