# AI for Actuarial Science
_Repository for the final project of the AI for Actuarial Science course (2nd Semester, final year, ENSAE Paris)._

## Contents

* The code used to train the machine learning models can be found in the `src/models/` folder (the parameters used for the grid search are stored in the `conf` folder, and the `main.py` script orchestrates it all.
* The code used to train the deep learning model can be found in the `Neural_Network.ipynb` notebook.
* Then, on the one hand, `📚_Presentation.py` and `pages/` contain the user interface code for the three pages of our `Streamlit` application. On the other, `src/app/` contains the code for our application's backend (as well as some useful frontend components). `static/` folder contains some content of the app and the `css` styles.
* Finally, in order to deploy the app, we built a `Docker` image (with entrypoint being the `run.sh`script). We automated the image delivery thanks to some configuration stuff (`deployment/`and `argocd` folders), hence a new image is being pushed to the `DockerHub` at every new version of the app.

<br>

_**NB1:** The data comes from a public [Kaggle Repository](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction), and can also be directly downloaded from this site. We have also added the data to a S3 bucket, accessible to the [SSP Cloud](https://datalab.sspcloud.fr/)'s solution ([`MinIO`](https://min.io/)). Hence, in our code, we directly use the data that is stored in our bucket._

_**NB2:** When our application runs, many static elements are displayed. However, we also load the machine learning models we've trained (we've trained 6 models in total). The trained models are stored in the same S3 bucket as our data, and when running the application, we store them in cache memory to avoid reloading / optimize code efficiency._

## Setup Instructions

From the command line, you will have to follow the following steps to set this project up:

1. Clone this repository:

```bash
git clone https://github.com/JulesBrable/ai_insurance.git
```

2. Go to the project folder:
```bash
cd ai_insurance
```

3. Create and activate conda environnement:
   
```bash
conda create -n ai_insurance python=3.9 -y
conda acitvate ai_insurance
```

4. Install the listed dependencies:
   
```bash
pip install -r requirements.txt
```

## Model Training

To train the model, you can run the following commands:

```bash
cd src
```

```bash
python main.py
```

Note that `main.py` can take multiple arguments : `--methods` and `--model`. See the script for more information about the values that can be entered.
By default, we are training a Random Forest or a Logistic Regression, by GridSearchCV & StratifiedKFold (K=5) cross-validation. you can change the parameters of the grid in the `conf/params.yaml` file.

## Web application

In this project, we also built a simple [`Streamlit`](https://streamlit.io/) web app.

To access the app, one can simply click [here](https://ai-insurance.kub.sspcloud.fr/). Indeed, the app is deployed on a `Kubernetes` cluster hosted by [SSP Cloud](https://datalab.sspcloud.fr/).

On the other hand, you can also run this app locally. To do so, after following the set-up instruction described above, you will have to run the following command:

```bash
streamlit run 📚_Presentation.py --server.port=5010
```

By default, we are using port 5010, so once you have run the last command, you will be able to access the app with the following link: [http://localhost:5010/](http://localhost:5010/).

## Contact

* [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
* [Eunice Koffi]() - eunice.koffi@ensae.fr
* [Berthe Magajie Wamsa](https://github.com/BertheMagella) - berthe.magajiewamsa@ensae.fr
* [Leela Thamaraikkannan]() - leela.thamaraikkannan@ensae.fr
