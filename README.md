# RecsysAssociationPy

# Collaborative Filtering and Association Rule Mining App

This Python app combines Collaborative Filtering recommendation and Association Rule Mining to provide personalized item recommendations and discover interesting item associations in a transactional dataset. The app is built using Flask, a micro web framework, to create a RESTful API that exposes various endpoints for recommendation and association rule mining.

## Installation and Setup

1. Clone this repository to your local machine:

```bash
git clone https://github.com/imSrbh/RecsysAssociationPy.git
```

2. Install the required Python packages using pip:

```bash
pip install flask pandas mlxtend
```

3. Download the "OnlineRetail.csv" dataset and place it in the same directory as the "app.py" file.

## Usage

Start the Flask server by running the following command:

```bash
python app.py
```

The server will start running locally on your machine at `http://127.0.0.1:5000/`.

### Endpoints

The app provides the following endpoints:

1. `/item-recommendation`: Get item-based recommendations for a given item.

   Example usage: `http://127.0.0.1:5000/item-recommendation?item_id=23167`

2. `/user-recommendation`: Get user-to-user recommendation for two given users.

   Example usage: `http://127.0.0.1:5000/user-recommendation?user_idA=12583&user_idB=13047`

3. `/apriori-recommendation`: Discover association rules using Apriori algorithm.

   Example usage: `http://127.0.0.1:5000/apriori-recommendation?min_support=0.01&min_threshold=0.5`

4. `/fpgrowth-recommendation`: Discover association rules using FP-Growth algorithm.

   Example usage: `http://127.0.0.1:5000/fpgrowth-recommendation?min_support=0.01&min_threshold=0.5`

## Note

- The app preprocesses the "OnlineRetail.csv" dataset to create necessary matrices for collaborative filtering and association rule mining.
- The collaborative filtering part uses item-item similarity, while the association rule mining part uses Apriori and FP-Growth algorithms.
- The app provides flexible options to customize the minimum support and confidence thresholds for association rule mining.
- The responses are returned in JSON format for easy integration with other applications.

