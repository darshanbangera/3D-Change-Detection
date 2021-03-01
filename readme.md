# 3D Change Detection
Environment : environment.yml file provided to create conda environment
Dataset File Structure
-Project folder
    |------Data
    	     |
    	     |----model.pth
    	     |----train.dat(upon creation)
    	     |----test.dat(upon creation)
    |------Shrec_change_detection_dataset_public
    	     |
    	     |----2016
    	     |
    	     |----2020
    	     |
    	     |----labeled_point_lists_train

Data Preparation:

To create the data loader for the model use generate.py file. The data loader is is created and can be saved into Train, Test, Validation pickle files based on the requirements.

To create Train, validation loader run : python generate_dataset.py --Mode "train_val"
To create Test loader run: python generate_dataset.py --Mode "val"

Model Training:
The model can be trained on the data set using train.py. Update the complete path in the python file if model needs to be saved or loaded.

Model Evaluation:
To evaluate the model on test data use evaluate.py file. The evaluation can be done once the pickle files for test/validation loader is generated. 

To evaluate on Test set run: python evaluate.py --Mode "test"
To evaluate on Validation set run: python evaluate.py --Mode "train_eval"

 



