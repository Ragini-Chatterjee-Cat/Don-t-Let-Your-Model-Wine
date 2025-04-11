Don’t Let Your Model Wine

mini MLOps platform for training ML modles, and checking if they're being fair to different kinds of wines.

What This Does

Trains a model to predict wine qualitty (score from 0 to 10) based on chemical properties like alcohol, acidity, etc.

Packages the model, the metrics (accurracy and stuff), and the config all nice into the outputs folder

Checks if the model is being “fair” to low-alcohol vs high-alcohol wines. (equality matters)

Has a CLI tool (src/cli.py) that lets you run the whole thing.

Has a dashboard (Streamlit) so you can see how well (or not well) the model did.

Is Dockerized so you can run it anywhere. :)
