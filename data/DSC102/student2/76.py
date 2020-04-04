import dask
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import json
import re
from distributed import Client

def Assignment1A(user_reviews_csv):
    client = Client('127.0.0.1:8786')
#     client = Client('172.31.6.146:8786')
    client = client.restart()    
    SPLIT_OUT = 16
    
    revs = user_reviews_csv.loc[:, ['reviewerID', 'asin', 'overall', 'helpful',\
                                    'reviewTime']]#dd.read_csv(rev_path)'/movie'/s//'

    revs_stg1 = revs.assign(converted_helpful = lambda x:\
                    x.helpful.str[1:-1].str.split(',').str[0].str.strip())\
                    .astype({'converted_helpful': 'int32'})

    revs_stg2 = revs_stg1.assign(converted_total = lambda x:\
                    x.helpful.str[1:-1].str.split(',').str[1].str.strip())\
                    .astype({'converted_total': 'int32'})

    revs_stg3 = revs_stg2.assign(year = lambda x: x.reviewTime.str[-4:]).astype({'year': 'int32'})

    revs_group = revs_stg3.groupby('reviewerID')
    #.loc[:, ['reviewerID', 'asin', 'overall','year', 'converted_helpful', 'converted_total']]
    
    final_df = revs_group.agg({
        'asin': 'count',
        'overall': 'mean',
        'year': 'min',
        'converted_helpful': 'sum',
        'converted_total' : 'sum'
    }, split_out = SPLIT_OUT)
    
    submit = final_df.describe().compute().round(2)
    submit = submit.rename({
        'asin': 'number_products_rated',
        'overall': 'avg_ratings',
        'year': 'reviewing_since',
        'converted_helpful': 'helpful_votes',
        'converted_total': 'total_votes'
    }, axis = 1)
    with open('results_1A.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)

def scale_dict(dictionary, factor = 100):
    for key in dictionary:
        dictionary[key] = round(dictionary[key] * factor, 2)
        
        
def Assignment1B(user_reviews_csv, products_csv):
    import re
    client = Client('127.0.0.1:8786')
#     client = Client('172.31.6.146:8786')
    client = client.restart()
    SPLIT_OUT = 16
    #===============  q1 return 2 dicts    
    res1_revs = user_reviews_csv.isna().mean().compute().to_dict()
    res1_prods = products_csv.isna().mean().compute().to_dict()
    
    #=============== q2 return num
    user_reviews_csv['revs'] = 1
    products_csv['prods'] = 1
    
    ddf_ratings = user_reviews_csv.loc[:, ['asin', 'overall', 'revs']]
    ddf_prices = products_csv.loc[:, ['asin', 'price', 'prods']]
    temp2 = ddf_ratings.merge(ddf_prices, how = 'outer', on = 'asin').compute()#.drop_duplicates()
    temp2_prices = temp2.price
    temp2_ratings = temp2.overall
    res2 = temp2_prices.corr(temp2_ratings)
    
    #=============== q3 return dict
#     res3 = temp2_prices.dropna().describe().compute().to_dict()
    res3 = products_csv.price.describe().compute().to_dict()
    res3.pop('count')
    res3.pop('25%')
    res3.pop('75%')
    
    #=============== q4 return dict
    res4 = products_csv.categories.apply(\
        lambda x: str(x)[1: -1].split(',')[0].split(']')[0].split('[')[-1][1 : -1]
                                        )\
            .value_counts().compute().to_dict()
    
    #=============== q5 return bool
    selector = (temp2.prods.isna()) & (temp2.revs.notnull())
    temp5 = len(temp2.loc[selector])
    res5 = temp5 > 0
    
    # q6 return num
    PATTERN = '\'([A-Za-z0-9]+)\''
    products_csv['lst'] = products_csv.related.apply(lambda x: re.findall(PATTERN, str(x)))

    asin_col = products_csv.asin.unique().compute()
    lst_col = products_csv.lst
    temp6 = lst_col.explode()
    res6 = 0
    for idx, elem in temp6.iteritems():
        if not (elem in asin_col):
            res6 = 1
            break
    
    # Write your results to "results_1B.json" here
    scale_dict(res1_prods)
    scale_dict(res1_revs)
    scale_dict(res3, factor = 1)
    
    submission = {
        "q1": {
            "products": res1_prods,
            "reviews": res1_revs
        },
        "q2": round(res2, 2),
        "q3": res3,
        "q4": res4,
        "q5": int(res5), 
        "q6": int(res6)
    }
    with open('results_1B.json', 'w') as outfile: json.dump((submission), outfile)
    