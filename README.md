# Good Netizens cyberbullying classification model API

This is the API for our machine learning model used to classify For FIT5120. 

It uses [FastAPI](https://fastapi.tiangolo.com/) to serve the API, a logistic 
regression model (using [scikit-learn](https://scikit-learn.org/)) to classify 
text as cyberbullying or not, and an RNN (using [Tensorflow](https://www.tensorflow.org/)) 
to further categorise the text as toxic, aggressive, or attacking.

## Usage

To run this, you must have Docker and Docker Compose installed. You can find 
instructions on how to install these in the [Docker docs](https://docs.docker.com/engine/install/).

First, clone the repository and `cd` into it:

```
$ git clone https://github.com/mattjamesdev/goodnetizensau-model-pipeline.git
$ cd goodnetizensau-model-pipeline
```

Now build and run the container using Docker Compose:

```
$ docker compose up

# Or if you want to run it in detached mode
$ docker compose up -d
```

If you make any changes to the files, make sure to add the `--build` flag to 
rebuild the image.

```
$ docker compose up --build
```

You can now access the API from `http://localhost:8000/`.

To see the FastAPI auto-generated docs, head to `http://localhost:8000/docs`.
