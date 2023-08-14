# TopicDiversifiedVAESumm

Description of the model : VAE with Dirichlet distribution for unsupervised Text Summarization

## Description

The model present a multi-task learning model with two autoencoders. One learning a topic model and the other learning to summarize documents. The topic model is then used to bias summaries generation towards specific topics.
The details of the model can be found in paper : TO BE ADDED

## How tu run model :

All parameters of the model can be modified in the "configs/config.py" file. The default configuration is set as presented in the article
To train the model : 
    a. Set the train_model variable to True 
    b. Input a Name (str) to record model -- default name is set to "default_config"
To generate summaries and metrics:
    a. the variable generate must be set to True
    b. the metrics compute_evals must be set to True
    c. you must enter a save_name for all files names
    If you used the generate only, you must provide a model name (Name variable as defined in Train)
    
Here is an example to train and generate the results: python TopicMainSum.py --train_model=True Name=Model_name --generate=True --compute_evals=True --save_name=Model_results

## Project Organization

├── README.md  --> Read me file for developpers.
├── TopicSumMain.py --> Runing code for defined configs.
├── configs --> Directory for configurations of model & application.
├── data --> data processing folder 
├── experiments --> Trained and serialized models, model predictions,
│                              run metrics, or model summaries.
    ├── evaluation --> Computing evaluation.
    ├── model_save --> Storing trained models.
    ├── results --> Storing json/csv results files.
├── model --> TopicVAE Summraizer code.
├── Processing_csv --> If you want to preprocess the data as in paper. 
├── runs --> Tensorboard runs.
├── utils --> All utility function for the model.