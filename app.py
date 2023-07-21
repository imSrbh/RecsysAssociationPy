from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import association_rules, apriori, fpgrowth

from json import JSONEncoder

class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, frozenset):
            return list(o)
        return super().default(o)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder


# Load data and preprocess
df = pd.read_csv("OnlineRetail.csv", encoding='unicode_escape')
df1 = df.dropna(subset=['CustomerID'])
customer_item_matrix = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_to_user_sim_matrix.columns = customer_item_matrix.index
user_to_user_sim_matrix['CustomerID'] = customer_item_matrix.index
user_to_user_sim_matrix = user_to_user_sim_matrix.set_index('CustomerID')
item_item_sim_matrix = pd.DataFrame(cosine_similarity(df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T))
item_item_sim_matrix.columns = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T.index
item_item_sim_matrix['StockCode'] = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T.index
item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')

# Item-based recommendation
def ib_recommend_items(item_id, top_n=5):
    top_similar_items = list(
        item_item_sim_matrix
        .loc[item_id]
        .sort_values(ascending=False)
        .iloc[:top_n]
        .index
    )
    recommended_items = df1.loc[df1['StockCode'].isin(top_similar_items), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[top_similar_items]
    return recommended_items

# User-based recommendation
def ub_recommend_items(user_idA, user_idB, top_n=5):
    user_idA = float(user_idA)
    user_idB = float(user_idB)
    items_bought_by_A = set(customer_item_matrix.loc[user_idA].iloc[customer_item_matrix.loc[user_idA].to_numpy().nonzero()].index)
    items_bought_by_B = set(customer_item_matrix.loc[user_idB].iloc[customer_item_matrix.loc[user_idB].to_numpy().nonzero()].index)
    items_to_recommend_User_B = items_bought_by_A - items_bought_by_B
    items_to_recommend_User_B = df1.loc[df['StockCode'].isin(items_to_recommend_User_B), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[items_to_recommend_User_B]
    return items_to_recommend_User_B

def appriori_recommend_items(min_support=0.01, min_threshold=0.5):
    data = df.sort_values(by='InvoiceDate')
    data = data.set_index('InvoiceDate')
    print('Dataset Shape:', data.shape)
    data['sold'] = 1
    pivot = data.pivot_table(values='sold', index='InvoiceNo', columns='Description').fillna(0)
    # Convert to bool type
    pivot = pivot.astype(bool)
    # Limit the number of items
    pivot = pivot.iloc[:, :2000]  # Adjust the number of items as needed
    print('Pivot Table Shape:', pivot.shape)
    

    rules = apriori(pivot, min_support=min_support, use_colnames=True).sort_values(by='support', ascending=False)
    rules = association_rules(rules, metric='lift', min_threshold=min_threshold).sort_values(by='lift', ascending=False)
    
    # Convert the frozenset in 'antecedents' and 'consequents' columns to list
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules

def fpgrowth_recommend_items(min_support=0.01, min_threshold=0.5):
    data2 = df.sort_values(by='InvoiceDate')
    data2 = data2.set_index('InvoiceDate')
    print('Dataset Shape:', data2.shape)
    data2['sold'] = 1
    pivot = data2.pivot_table(values='sold', index='InvoiceNo', columns='Description').fillna(0)
    # Convert to bool type
    pivot = pivot.astype(bool)
    # Limit the number of items
    pivot = pivot.iloc[:, :2000]  # Adjust the number of items as needed
    print('Pivot Table Shape:', pivot.shape)
    freq_items1 = fpgrowth(pivot, min_support=0.02, use_colnames=True).sort_values(by='support', ascending=False)
    rules1 = association_rules(freq_items1, metric='lift', min_threshold=1).sort_values(by='lift', ascending=False)
    return rules1


@app.route('/item-recommendation', methods=['GET'])
def item_recommendation():
    item_id = request.args.get('item_id')
    recommended_items = ib_recommend_items(item_id)
    return jsonify(recommended_items.to_dict(orient='index'))


@app.route('/user-recommendation', methods=['GET'])
def user_recommendation():
    user_idA = request.args.get('user_idA')
    user_idB = request.args.get('user_idB')
    recommended_items = ub_recommend_items(user_idA, user_idB)
    # Reset the index to make it unique
    recommended_items.reset_index(drop=True, inplace=True)
    return jsonify(recommended_items.to_dict(orient='records'))


@app.route('/apriori-recommendation', methods=['GET'])
def apriori_recommendation():
    min_support = float(request.args.get('min_support', 0.01))
    min_threshold = float(request.args.get('min_threshold', 0.5))
    rules = appriori_recommend_items(min_support, min_threshold)
    return jsonify(rules.to_dict(orient='index'))

@app.route('/fpgrowth-recommendation', methods=['GET'])
def fpgrowth_recommendation():
    min_support = float(request.args.get('min_support', 0.01))
    min_threshold = float(request.args.get('min_threshold', 0.5))
    rules = fpgrowth_recommend_items(min_support, min_threshold)
    return jsonify(rules.to_dict(orient='index'))

if __name__ == '__main__':
    app.run(debug=True)
