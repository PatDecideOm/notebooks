import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Get Data
germany = pd.read_csv('./germany.csv')
print(germany.shape)

# Filter Data
germany = germany[germany['quantity'] >= 0]
germany = germany[germany['stockcode'] != 'POST']
print(germany.shape)

germany = germany.drop('description', 1)

germany.to_csv('./germany_filter.csv', index=False, sep=',', encoding='utf-8')

transaction_invoice = germany.groupby('invoiceno').apply(lambda x: ','.join(x.stockcode))
# df_transaction = pd.DataFrame({'invoiceno': transaction.index, 'stockcode': sorted(transaction.values)})

transaction_customer = germany.groupby('customerid').apply(lambda x: ','.join(x.stockcode))

df_transaction = pd.DataFrame({'customerid': transaction_customer.index, 'stockcode': sorted(transaction_customer.values)})

ls_sales = list(df_transaction['stockcode'].apply(lambda x: sorted(x.split(','))))

# instantiate transcation encoder
encoder = TransactionEncoder().fit(ls_sales)
onehot = encoder.transform(ls_sales)

# convert one-hot encode data to DataFrame
onehot = pd.DataFrame(onehot, columns=encoder.columns_)

print('Start of frequent itemsets')

# compute frequent items using the Apriori algorithm - Get up to three items
frequent_itemsets = apriori(onehot, min_support=0.075, max_len=3, use_colnames=True)

print('Start of rules')

# compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8)

rules['lhs items'] = rules['antecedents'].apply(lambda x: len(x))

print(rules[rules['lhs items']>1].sort_values('lift', ascending=False).head())

# print(rules.head())

# Replace frozen sets with strings
rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

# Write Rules to Output CSV
rules.to_csv('./rules_customer.csv', index=False, sep=';')

# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules[rules['lhs items']>1].pivot(index='antecedents_', columns='consequents_', values='lift')

# Generate a heatmap with annotations on and the colorbar off
sns.heatmap(pivot.iloc[1:15, 1:15], annot=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
