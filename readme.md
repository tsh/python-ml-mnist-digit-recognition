## Python digit recognition

Application attempts to recognize your handwritten digits, using different machine learning algorithms.


### Usage:
Run `docker-compose up` and navigate to `localhost:8000`.
  
### How it works:

If started for the first time app will download MNIST dataset, to `./data` directory and train classificators against it.
 Trained classifiers serialized to `classifiers` directory. Note: trained SVC classifier included in this repo. 
Depending on your machine, classification can take more than 20 minutes.