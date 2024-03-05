# AI for Actuarial Science
_Repository for the final project of the AI for Actuarial Science course (2nd Semester, final year, ENSAE Paris)._

## Contents


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

_**Note:** the data comes from a [Kaggle Repository](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction), and can also be directly downloaded from this site. We have also added the data to a S3 bucket, accessible to the [SSP Cloud](https://datalab.sspcloud.fr/)'s solution ([`MinIO`](https://min.io/)). Hence, in our code, we directly use the data that is stored in our bucket._

## Web application

In this project, we also built a simple [`Streamlit`](https://streamlit.io/) web app.

To access the app, one can simply click [here](https://ai-insurance.kub.sspcloud.fr/). Indeed, the app is deployed on a `Kubernetes` cluster hosted by [SSP Cloud](https://datalab.sspcloud.fr/).

On the other hand, you can also run this app locally. To do so, once you have followed the set-up instruction described above, you will have to run the following command:

```bash
streamlit run 📚_Presentation.py --server.port=5010
```

By default, we are using port 5010, so once you have run the last command, you will be able to access the app with the following link: [http://localhost:5010/](http://localhost:5010/).

## Contact

* [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
* [Eunice Koffi]() - eunice.koffi@ensae.fr
* [Berthe Magajie Wamsa](https://github.com/BertheMagella) - berthe.magajiewamsa@ensae.fr
* [Leela Thamaraikkannan]() - leela.thamaraikkannan@ensae.fr