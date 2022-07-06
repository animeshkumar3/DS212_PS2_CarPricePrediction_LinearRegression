# MLOpsImage

Following is the source code structure

## src -- Source code direcory
      start.py                 -- Marks the start time of the train process
      train.py                 -- Performing the training with baseline Linear Regression model
      evaluate.py              -- Performing validation using the test split (20%)
      generate_captions.py     -- Generates the caption for the given image

## report -- Generated Reports
      score.json               -- Report containing the MAE and MSE scores
      
## template -- User Interface 
      index.html               -- HTML file defining the UI for Flask app

## models -- 
      mnist_model_func.h5     -- Model saved using the pickle utilitybuildtime

## images -- , static --
      test.jpg                -- Input image for prediction

## test --  
      data.json               -- Output file having generated caption

## appforhero.py  -- Entry point of the web application for Heroku
## app.py         -- Entry point of the web application for AWS

## requirements.txt, dvcrequirements.txt -- List of dependencies to be installed
## dvc.yaml       -- DVC configuration file containing the stages
## .gitub/workflows/cml.yaml -- CML configuration for the repo
