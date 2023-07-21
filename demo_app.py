import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import association_rules, apriori, fpgrowth
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("OnlineRetail.csv", encoding='unicode_escape')
df1 = df.dropna(subset=['CustomerID'])

# customer_item_matrix = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
# customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

# user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
# user_to_user_sim_matrix.columns = customer_item_matrix.index
# user_to_user_sim_matrix['CustomerID'] = customer_item_matrix.index
# user_to_user_sim_matrix = user_to_user_sim_matrix.set_index('CustomerID')

def ub_recommend_items(user_idA, user_idB, top_n=5):
    user_idA = float(user_idA)
    user_idB = float(user_idB)
    print(user_idA,user_idB)
    customer_item_matrix = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
    customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

    user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
    user_to_user_sim_matrix.columns = customer_item_matrix.index
    user_to_user_sim_matrix['CustomerID'] = customer_item_matrix.index
    user_to_user_sim_matrix = user_to_user_sim_matrix.set_index('CustomerID')
    print(user_to_user_sim_matrix.loc[user_idA].sort_values(ascending = False))

    try:
        items_bought_by_A = set(customer_item_matrix.loc[user_idA].iloc[customer_item_matrix.loc[user_idA].to_numpy().nonzero()].index)
        items_bought_by_B = set(customer_item_matrix.loc[user_idB].iloc[customer_item_matrix.loc[user_idB].to_numpy().nonzero()].index)
    
    except KeyError:
        st.error("Invalid Customer ID. Please enter a valid Customer ID.")
        return []
    print("items_bought_by_A", items_bought_by_A)
    print("items_bought_by_B",items_bought_by_B)
    items_to_recommend_User_B = items_bought_by_A - items_bought_by_B
    items_to_recommend_User_B = df1.loc[df['StockCode'].isin(items_to_recommend_User_B), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[items_to_recommend_User_B]

    return items_to_recommend_User_B


def ib_recommend_items(item_id, top_n=5):
    item_item_sim_matrix = pd.DataFrame(cosine_similarity(df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T))
    item_item_sim_matrix.columns = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T.index
    item_item_sim_matrix['StockCode'] = df1.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0).T.index
    item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')

    top_similar_items = list(
        item_item_sim_matrix
        .loc[item_id]
        .sort_values(ascending=False)
        .iloc[:top_n]
        .index
    )

    top_similar_items = df1.loc[df1['StockCode'].isin(top_similar_items), ['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[top_similar_items]
    # print(top_similar_items)
    return top_similar_items


def mine_association_rules(min_support=0.02, min_threshold=0.5):
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    basket = basket.applymap(lambda x: 1 if x >= 1 else 0)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_threshold)

    return rules

    
def appriori_recommend_items(min_support=0.01, min_threshold=0.5):
    data = df.sort_values(by='InvoiceDate')
    data = data.set_index('InvoiceDate')
    print('Dataset Shape:', data.shape)
    data['sold'] = 1
    pivot = data.pivot_table(values='sold', index='InvoiceNo', columns='Description').fillna(0)
    # Limit the number of items
    pivot = pivot.iloc[:, :2000]  # Adjust the number of items as needed
    print('Pivot Table Shape:', pivot.shape)
    
    # Lower the min_support value
    freq_items = apriori(pivot, min_support=min_support, use_colnames=True).sort_values(by='support', ascending=False)
    rules = association_rules(freq_items, metric='lift', min_threshold=min_threshold).sort_values(by='lift', ascending=False)
    return rules
    # Adjust the min_threshold value if needed
    # return rules[(rules['lift'] > 5) & (rules['confidence'] > 0.5)]

def fpgrowth_recommend_items(min_support=0.01, min_threshold=0.5):
    data2 = df.sort_values(by='InvoiceDate')
    data2 = data2.set_index('InvoiceDate')
    print('Dataset Shape:', data2.shape)
    data2['sold'] = 1
    pivot = data2.pivot_table(values='sold', index='InvoiceNo', columns='Description').fillna(0)
    # Limit the number of items
    pivot = pivot.iloc[:, :2000]  # Adjust the number of items as needed
    print('Pivot Table Shape:', pivot.shape)
    freq_items1 = fpgrowth(pivot,min_support=0.02,use_colnames=True).sort_values(by='support',ascending=False)
    rules1 = association_rules(freq_items1,metric='lift',min_threshold=1).sort_values(by='lift',ascending=False)
    return rules1

def main():
    st.title("Application: Recommendation and Association Rule Mining")
    st.sidebar.title("Options")

    # Display options in the sidebar
    option = st.sidebar.selectbox("Select an option", ("ItemBased_Recommendation", "UserBased_Recommendation", "Apriori", "FPGrowth", "Simple Association Rules"))

    if option == "ItemBased_Recommendation":
        st.header("Item Based Recommendation")
        item_id = st.text_input("Enter Item ID")
        item_name = df1.loc[df1['StockCode']==item_id,['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
        st.write(item_name)
        if st.button("Recommend"):
            recommended_items = ib_recommend_items(str(item_id))
            st.success("Recommended Items:")
            st.write(recommended_items)
            # for item in recommended_items:
            #     st.write("- " + item)

    elif option == "UserBased_Recommendation":
        st.header("User2User Based Item Recommendation")
        user_idA = st.text_input("Enter User ID A")
        user_idB = st.text_input("Enter User ID B")
        if st.button("Recommend"):
            # recommended_items = ub_recommend_items(str(user_idA), str(user_idB))
            recommended_items = ub_recommend_items(user_idA, user_idB)
            st.success("Recommended Items:")
            st.write(recommended_items)
    
    elif option == "Apriori":
        st.header("Association Rule Mining: Apriori")
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.02, 0.01)
        min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
        if st.button("Mine"):
            rules = appriori_recommend_items(min_support, min_threshold)
            st.dataframe(rules)

    elif option == "FPGrowth":
        st.header("Association Rule Mining: FPGrowth")
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.02, 0.01)
        min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
        if st.button("Mine"):
            rules = fpgrowth_recommend_items(min_support, min_threshold)
            st.dataframe(rules)

    elif option == "Simple Association Rules":
        st.header("Association Rule Mining")
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.02, 0.01)
        min_threshold = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
        if st.button("Mine"):
            rules = mine_association_rules(min_support, min_threshold)
            st.dataframe(rules)

if __name__ == '__main__':
    main()
