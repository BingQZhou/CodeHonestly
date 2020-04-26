import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
from pyspark.sql.functions import col, create_map, lit
from itertools import chain
from pyspark.sql.functions import arrays_zip, col,explode, explode_outer
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
import datetime
import numpy as np
import gc
from pyspark.ml.stat import Summarizer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics 
# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    review = review_data.alias('review')
    product = product_data.alias('product').select(asin_column)
    join_df=review.groupBy(review[asin_column])\
    .agg({overall_column: 'avg','*':'count'})\
    .withColumnRenamed("count(1)", count_rating_column)\
    .withColumnRenamed("avg(overall)", mean_rating_column)
    join_df=product.join(join_df,asin_column,'left_outer').cache()
    
    
    df_stats = join_df.select(
        F.mean(col(mean_rating_column)).alias('mean_mean_rating_column'),
        F.variance(col(mean_rating_column)).alias('variance_mean_rating_column'),
        F.mean(col(count_rating_column)).alias('mean_count_rating_column'),
        F.variance(col(count_rating_column)).alias('variance_count_rating_column')
    ).collect()
    
    mean_mean_rating_column = df_stats[0]['mean_mean_rating_column']
    variance_mean_rating_column = df_stats[0]['variance_mean_rating_column']
    mean_count_rating_column = df_stats[0]['mean_count_rating_column']
    variance_count_rating_column = df_stats[0]['variance_count_rating_column']
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': 0,
        'mean_meanRating': 0,
        'variance_meanRating': 0,
        'numNulls_meanRating': 0,
        'mean_countRating': 0,
        'variance_countRating': 0,
        'numNulls_countRating': 0
    }
    #Modify res:
    res = {
        'count_total': join_df.count(),
        'mean_meanRating': mean_mean_rating_column,
        'variance_meanRating': variance_mean_rating_column,
        'numNulls_meanRating': join_df.filter(join_df[mean_rating_column].isNull()).count(),
        'mean_countRating': mean_count_rating_column,
        'variance_countRating': variance_count_rating_column,
        'numNulls_countRating': join_df.filter(join_df[count_rating_column].isNull()).count()
    }

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------
    
def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    product_data = product_data.select(
        product_data.categories.getItem(0).getItem(0).alias(category_column),
        F.map_keys(product_data.salesRank).getItem(0).alias(bestSalesCategory_column),
        F.map_values(product_data.salesRank).getItem(0).alias(bestSalesRank_column)
    ).cache()
    
    df_stats = product_data.select(
        F.mean(col(bestSalesRank_column)).alias('mean_bestSalesRank_column'),
        F.variance(col(bestSalesRank_column)).alias('variance_bestSalesRank_column'),
    ).collect()

    mean_bestSalesRank_column = df_stats[0]['mean_bestSalesRank_column']
    variance_bestSalesRank_column = df_stats[0]['variance_bestSalesRank_column']
    
    

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': product_data.count(),
        'mean_bestSalesRank': mean_bestSalesRank_column,
        'variance_bestSalesRank': variance_bestSalesRank_column,
        'numNulls_category': product_data.filter((product_data[category_column].isNull())|\
                                                 (product_data[category_column]=='')).count(),
        'countDistinct_category': product_data.filter(product_data[category_column]!='')\
                                    .select(category_column).dropna().distinct().count(), 
        'numNulls_bestSalesCategory': product_data.filter((product_data[bestSalesCategory_column].isNull())).count(),
        'countDistinct_bestSalesCategory': product_data.select(bestSalesCategory_column).dropna().distinct().count()
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------

def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product = product_data.limit(10000)
    #product = product_data#.persist()
    
    def get_value(x):
        if x:
            if attribute in x.keys():
                return x[attribute]
        return None
    
    def count_num(x):
        if x:
            return len(x)
        else:
            return None
    get_value_udf = F.udf(get_value, ArrayType(StringType()))
    count_num_udf = F.udf(count_num, IntegerType())
    product = product_data.select(asin_column,get_value_udf(related_column).alias(meanPriceAlsoViewed_column))
    temp_product = product.select(product.asin,explode_outer(meanPriceAlsoViewed_column))
    temp_product=temp_product.withColumnRenamed('asin','pid')
    
    join_product=temp_product.join(product_data[[asin_column,price_column]],temp_product.col==product_data.asin,'leftouter')

    avg_product=join_product.groupBy('pid').avg(price_column)

    count_product = product.select(count_num_udf(meanPriceAlsoViewed_column).alias(countAlsoViewed_column))
    
    avg_results = avg_product.select(F.avg('avg(price)')\
                     ,F.variance('avg(price)')\
                     ,F.count(F.when(avg_product['avg(price)'].isNull(),'avg(price)'))).collect()
    count_results = count_product.select(F.avg(countAlsoViewed_column)\
                     ,F.variance(countAlsoViewed_column)\
                     ,F.count(F.when(count_product[countAlsoViewed_column].isNull(),countAlsoViewed_column))).collect()
    count_total = product.count()
    mean_meanPriceAlsoViewed = avg_results[0][0]
    variance_meanPriceAlsoViewed = avg_results[0][1] 
    numNulls_meanPriceAlsoViewed = avg_results[0][2] 
    mean_countAlsoViewed = count_results[0][0]
    variance_countAlsoViewed = count_results[0][1]
    numNulls_countAlsoViewed = count_results[0][2]  
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res['count_total'] = count_total
    res['mean_meanPriceAlsoViewed'] = mean_meanPriceAlsoViewed
    res['variance_meanPriceAlsoViewed'] = variance_meanPriceAlsoViewed
    res['numNulls_meanPriceAlsoViewed'] = numNulls_meanPriceAlsoViewed
    res['mean_countAlsoViewed'] = mean_countAlsoViewed
    res['variance_countAlsoViewed'] = variance_countAlsoViewed
    res['numNulls_countAlsoViewed'] = numNulls_countAlsoViewed



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------

def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    product_price = product_data.select(product_data.price)
    product_price = product_price.fillna(product_price.agg((F.avg(price_column).alias(price_column))).head()[0])
    product_price = product_price.withColumn(meanImputedPrice_column, product_price[price_column])
    
    
    def median(values_list):
        med = np.median(values_list)
        return float(med)
    udf_median = F.udf(median, FloatType())

    median = product_data.agg(udf_median(F.collect_list(col(price_column)))).head()[0]
    
    product_data = product_data.fillna({ price_column: median})
    product_data = product_data.withColumn(medianImputedPrice_column, product_data[price_column])
    product_data = product_data.withColumn(unknownImputedTitle_column, product_data[title_column])
    product_data = product_data.fillna({ unknownImputedTitle_column: 'unknown'})


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total'] = product_price.count()
    res['mean_meanImputedPrice']= product_price.select(F.avg(F.col(meanImputedPrice_column))).head()[0]
    res['variance_meanImputedPrice']=  product_price.agg({meanImputedPrice_column: 'variance'}).head()[0]
    res['numNulls_meanImputedPrice']= product_price.where(F.col(meanImputedPrice_column).isNull()).count()
    res['mean_medianImputedPrice']= product_data.select(F.avg(F.col(medianImputedPrice_column))).head()[0]
    res['variance_medianImputedPrice']= product_data.agg({medianImputedPrice_column: 'variance'}).head()[0]
    res['numNulls_medianImputedPrice']= product_data.where(F.col(medianImputedPrice_column).isNull()).count()
    res['numUnknowns_unknownImputedTitle']= product_data.where(F.col(unknownImputedTitle_column)=='unknown').count()
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------

def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    def get_titleArray(x):
        return x.lower().split(' ')
    
    get_titleArray_udf = F.udf(get_titleArray, ArrayType(StringType()))
    
    product_processed_data = product_processed_data.withColumn(titleArray_column,\
                                                               get_titleArray_udf(F.col(title_column))).cache()
    
    
    word2Vec = M.feature.Word2Vec(minCount= 100,vectorSize=16, numPartitions = 4,seed=SEED, \
                                  inputCol=titleArray_column, outputCol=titleVector_column)
    model = word2Vec.fit(product_processed_data)

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': None,
        'word_1_synonyms': None,
        'word_2_synonyms': None
    }
    # Modify res:
    res['count_total'] = product_processed_data.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    #product_processed_data = product_processed_data.limit(10000)
    stringIndexer = M.feature.StringIndexer(inputCol=category_column, outputCol=categoryIndex_column)
    model = stringIndexer.fit(product_processed_data)
    td = model.transform(product_processed_data)
    encoder = M.feature.OneHotEncoder(dropLast=False,inputCol=categoryIndex_column, outputCol=categoryOneHot_column)
    td= encoder.transform(td)
    pca = M.feature.PCA(k=15, inputCol=categoryOneHot_column, outputCol=categoryPCA_column)
    model = pca.fit(td)
    td= model.transform(td)
    meanVector_categoryOneHot = td.select(Summarizer.mean(F.col(categoryOneHot_column))).head()[0]
    meanVector_categoryPCA = td.select(Summarizer.mean(F.col(categoryPCA_column))).head()[0]
    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total']=td.count()
    res['meanVector_categoryOneHot']=meanVector_categoryOneHot
    res['meanVector_categoryPCA']=meanVector_categoryPCA

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    dt = DecisionTreeRegressor(maxDepth=5, labelCol="overall",featuresCol='features')
    model = dt.fit(train_data)
    predictions = model.transform(test_data)
    
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse']=rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    (training, validation) = train_data.randomSplit([0.75, 0.25])
    dt5 = DecisionTreeRegressor(maxDepth=5, labelCol="overall",featuresCol='features')
    dt7 = DecisionTreeRegressor(maxDepth=7, labelCol="overall",featuresCol='features')
    dt9 = DecisionTreeRegressor(maxDepth=9, labelCol="overall",featuresCol='features')
    dt12 = DecisionTreeRegressor(maxDepth=12, labelCol="overall",featuresCol='features')
    model5 = dt5.fit(training)
    model7 = dt7.fit(training)
    model9 = dt9.fit(training)
    model12 = dt12.fit(training)
    predictions5 = model5.transform(validation)
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    #rmse5 = evaluator.evaluate(predictions)
    predictions7 = model7.transform(validation)
    #rmse7 = evaluator.evaluate(predictions)
    predictions9 = model9.transform(validation)
    #rmse9 = evaluator.evaluate(predictions)
    predictions12 = model12.transform(validation)
    #rmse12 = evaluator.evaluate(predictions)
    result=[evaluator.evaluate(predictions5),evaluator.evaluate(predictions7)\
            ,evaluator.evaluate(predictions9),evaluator.evaluate(predictions12)]
    best_index=result.index(min(result))
    best_rmse=0
    if best_index==0:
        predictions = model5.transform(test_data)
        best_rmse=evaluator.evaluate(predictions)
    if best_index==1:
        predictions = model7.transform(test_data)
        best_rmse=evaluator.evaluate(predictions)
    if best_index==2:
        predictions = model9.transform(test_data)
        best_rmse=evaluator.evaluate(predictions)
    if best_index==3:
        predictions = model12.transform(test_data)
        best_rmse=evaluator.evaluate(predictions)
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res['test_rmse']=best_rmse
    res['valid_rmse_depth_5']=result[0]
    res['valid_rmse_depth_7']=result[1]
    res['valid_rmse_depth_9']=result[2]
    res['valid_rmse_depth_12']=result[3]


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------