import pandas as pd
import numpy as np
from numpy import nan
from mlxtend.frequent_patterns import association_rules, apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2

# Here you want to change your database,
# username & password according to your own values
param_dic = {
    "host": "localhost",
    "database": "datascience",
    "user": "patrice",
    "password": "P@trice1969si",
    "port": "5432"
}

#----------------------------------------------------------------
# SqlAlchemy Only
#----------------------------------------------------------------
from sqlalchemy import create_engine
connect = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
    param_dic['user'],
    param_dic['password'],
    param_dic['host'],
    param_dic['port'],
    param_dic['database']
)

def to_alchemy(df, table):
    """
    Using a dummy table to test this call library
    """
    engine = create_engine(connect)
    df.to_sql(
        table,
        con=engine,
        index=False,
        if_exists='replace'
    )
    print("to_sql() done (sqlalchemy)")

print('Start of Recommender...')

print('Start of Get Data')

# Get Data
online_retail = pd.read_excel('./online_retail_II.xlsx')
print(online_retail.shape)

print('End of Get Data')

print('Start of Filter Data')

# Filter Data
online_retail = online_retail[online_retail['Quantity'] >= 0]
online_retail = online_retail[online_retail['StockCode'] != 'POST']
online_retail['Customer ID'] = online_retail['Customer ID'].replace(nan, 0)
online_retail = online_retail[online_retail['Customer ID'] != 0 ]
print(online_retail.shape)

online_trans = pd.DataFrame({'invoiceno': online_retail['Invoice'].apply(str),
                             'stockcode': online_retail['StockCode'].apply(str),
                             'quantity': online_retail['Quantity'],
                             'invoicedate': online_retail['InvoiceDate'],
                             'price': online_retail['Price'],
                             'customerid': online_retail['Customer ID'].astype(int).apply(str)
                            })

print('End of Filter Data')

print('Start of Write Data')

online_trans.to_csv('./online_trans.csv', index=False, sep=',', encoding='utf-8')

to_alchemy(online_trans, 'online_retail')

print('End of Write Data')

trans_invoice = online_trans.groupby('invoiceno').apply(lambda x: ','.join(x.stockcode))
trans_customer = online_trans.groupby('customerid').apply(lambda x: ','.join(x.stockcode))
df_transaction = pd.DataFrame({'invoiceno': trans_invoice.index, 'stockcode': sorted(trans_invoice.values)})
df_transaction = pd.DataFrame({'customerid': trans_customer.index, 'stockcode': sorted(trans_customer.values)})

ls_sales = list(df_transaction['stockcode'].apply(lambda x: sorted(x.split(','))))

# instantiate transcation encoder
encoder = TransactionEncoder().fit(ls_sales)
onehot = encoder.transform(ls_sales)

# convert one-hot encode data to DataFrame
onehot = pd.DataFrame(onehot, columns=encoder.columns_)

print('Start of Frequent Itemsets')

# compute frequent items using the Apriori algorithm - Get up to three items
frequent_itemsets = apriori(onehot, min_support=0.020, max_len=5, use_colnames=True)

print('End of Frequent Itemsets')

print('Start of Rules')

# compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8)

rules['lhs items'] = rules['antecedents'].apply(lambda x: len(x))

print(rules[rules['lhs items']>1].sort_values('lift', ascending=False).head())

# print(rules.head())

# Replace frozen sets with strings
rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

# Write Rules to Output CSV
rules.to_csv('./rules_invoice.csv', index=False, sep=';')

print('End of Rules')

print('End of Recommender...')
