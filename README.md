# RecsysAssociationPy
# Collaborative Filtering and Association Rule Mining App

This is a Python application that implements collaborative filtering for item recommendation and association rule mining based on the Apriori and FPGrowth algorithms. The app is built using Flask, a lightweight web framework for Python.

## Requirements

To run this app, you need to have the following installed:

- Python 3.7 or higher
- Flask
- pandas
- mlxtend

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python3.7 -m venv .env

source .env/bin/activate

pip3 install -r requirements.txt

python3 app.py 
```

## Dataset

The app uses the "OnlineRetail.csv" dataset for collaborative filtering and association rule mining. The dataset contains online retail transaction data. The app loads and preprocesses the data to create customer-item and item-item matrices for collaborative filtering.

## How to Use : Docker

1. Clone this repository to your local machine.

2. Navigate to the project directory containing the Dockerfile.

3. Build the Docker image by running the following command:

```bash
docker build -t collaborative-app .
```

Replace `collaborative-app` with the desired name for your Docker image.

4. Once the image is built, run the Docker container using the following command:

```bash
docker run -p 5000:5000 collaborative-app
```

5. The Flask app will now be running inside the Docker container. You can access it at [http://localhost:5000](http://localhost:5000) on your local machine.

## Endpoints

The app provides the following endpoints:

- `/item-recommendation`: Recommends similar items based on an input item ID.
- `/user-recommendation`: Recommends items to one user based on the items purchased by another user.
- `/apriori-recommendation`: Mines association rules using the Apriori algorithm and provides recommendations based on the rules.
- `/fpgrowth-recommendation`: Mines association rules using the FPGrowth algorithm and provides recommendations based on the rules.



## Example Usage

1. To get item recommendations based on item ID 23167:

```bash
curl http://localhost:5000/item-recommendation?item_id=23167
```

2. To get user-based item recommendations for users with IDs 12583 and 13047:

```bash
curl http://localhost:5000/user-recommendation?user_idA=12583&user_idB=13047
```

3. To mine association rules using apriori and get recommendations based on the rules:

```bash
curl http://localhost:5000/apriori-recommendation?min_support=0.01&min_threshold=0.5
```

4. To mine association rules using fpgrowth and get recommendations based on the rules:

```bash
curl http://127.0.0.1:5000/fpgrowth-recommendation?min_support=0.01&min_threshold=0.5
```

## Notes

- The app uses cosine similarity for collaborative filtering.
- The Apriori and FPGrowth algorithms are used for association rule mining.
- The dataset should be placed in the project directory and named "OnlineRetail.csv".

