# 3DDeconvolutionGAN

AI framework for perform deconvolution on 3D-volumes:

  - create a pre-trained generative adversaraial network (GAN)
  - in detail this GAN is an *vol2vol* (Volume-to-Volume) neuronal network 
  - train a generator which is able to perform deconvolution directly 
  - use generator and tailer-made discriminator for an iterative deconvolution model implemented with tensorflow

# Creaete vol2vol-GAN

  - Import volumes and specify hyperparameter via the json
  - Perform Deconvolution in corresponding jupyter notebook
