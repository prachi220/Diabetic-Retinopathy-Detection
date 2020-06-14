# Diabetic-Retinopathy-Detection

Trained CNN for automated detection of diabetic retinopa- thy and diabetic macular edema in retinal fundus photographs.
The dataset could be downloaded from https://www.kaggle.com/c/diabetic-retinopathy-detection/data. The dataset required compressing due to large image sizes.

The instructions to run the model on high processing computer are:

Change the following code in your shell script to your email id-
### Specify email address to use for notification.
  `#PBS -M <your webmail id>`

Running the code on high processing computer-

  `cd (the folder where training data and model scripts are present)`
  
  `sed -i -e 's/\r$//' job_model_binary.sh`
  
  `qsub job_model_binary.sh`

To check error after the job ends:

   `cat stderr_model_binary`

To check output after the job ends:

  `cat stdout_model_binary`
