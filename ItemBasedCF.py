import sys
import os
import time
import math
from pyspark import SparkContext

def pearson_similarity(bus1, bus2, id1, id2, bus_averages):
    '''
    formula: (sum of all [(score1 - mean1) × (score2 - mean2)]) / (sd1 * sd2)
    '''

    #users that rated both businesses
    overlap = set(bus1.keys()) & set(bus2.keys())

    #not enough data
    if len(overlap) < 3:
        diff = abs(bus_averages[id1] - bus_averages[id2])
        return (5 - diff) / 5       #normalize

    #average ratings for each business using overlap users
    bus1_sum = 0
    for u in overlap:
        bus1_sum += bus1[u]
    bus1_mean = bus1_sum / len(overlap)

    bus2_sum = 0
    for u in overlap:
        bus2_sum += bus2[u]
    bus2_mean = bus2_sum / len(overlap)

    #normalize by mean formula: sum of (rating1 - mean1) * (rating2 - mean2) for each common user
    top = sum((bus1[u] - bus1_mean) * (bus2[u] - bus2_mean)
                    for u in overlap)

    # standard deviation fo reach business
    # sum of squared diff for each overlapping user
    bottom1 = 0
    for u in overlap:
        bottom1 += (bus1[u] - bus1_mean) ** 2

    bottom2 = 0
    for u in overlap:
        bottom2 += (bus2[u] - bus2_mean) ** 2

    bottom = math.sqrt(bottom1 * bottom2)

    # check if dividing by 0
    if bottom == 0:
        return 0        #bottom

    similarity = top / bottom
    return similarity

def prediction(user_id, bus_id, user_to_businesses, business_to_users, business_avg_ratings, calc_sims, neighbors, default):
    '''
    methodology: item based cf
    finds all businesses that user has rated
    checks similarity between rated businesses and bus that needs prediction (pearson)
    uses weighted avg to calc rating using n nearest neighborsde

    edge cases:
    user has not rated any businesses yet (cold start)
    no similarities are found- user has not rated any similar businesses
    rating under 1 or over 5
    '''

    #cold start cases
    if user_id not in user_to_businesses or bus_id not in business_to_users:
        return float(round(default))

    if user_id not in user_to_businesses:
        return float(round(business_avg_ratings.get(bus_id, default)))       #return default if user id not found

    if bus_id not in business_to_users:
        user_ratings = user_to_businesses[user_id].values()
        user_avg = sum(user_ratings) / len(user_ratings)
        return float(round(0.8 * user_avg + 0.2 * default))    #return default if bus id not found

    similarities = []
    user_rated_businesses = user_to_businesses[user_id]     #all businesses prev rated by user {bus_id : rating, bus_id2, rating2...}

    for rated_business, rating in user_rated_businesses.items():
        sim_key = frozenset([bus_id, rated_business])          #key to look up if sim has alr been calculated (save time)
        if sim_key in calc_sims:
            similarity = calc_sims[sim_key]                 #return similarity if exists
        else:
            if bus_id in business_to_users and rated_business in business_to_users:  # check
                similarity = pearson_similarity(business_to_users[bus_id], business_to_users[rated_business], bus_id, rated_business, business_avg_ratings)      #calc similarity
                calc_sims[sim_key] = similarity         #add to cache
            else:
                similarity = 0          #if bus
        # only add pos similarities
        if similarity > 0:
            similarities.append((similarity, rating))

    '''
    # testing
    if user_id == 'wf1GqnKQuvH-V3QN80UOOQ' and bus_id == 'fThrN4tfupIGetkrz18JOg':
        print('check user id', user_id in user_to_businesses)
        print('check bus id', bus_id in business_to_users)
        print('check user_rated_businesses', user_rated_businesses)
        print('sim chec', not similarities)
    '''

    #user has not rated any similar businesses
    if not similarities:
        user_ratings = user_to_businesses[user_id].values()
        user_avg = sum(user_ratings) / len(user_ratings)
        return float(round(0.8 * user_avg + 0.2 * default))

    #sort desc
    similarities.sort(reverse=True)
    similarities = similarities[:neighbors]         #use only nearest neighbors

    #calc weighted sum with nearest neighbors
    weighted_sum = sum(sim * rating for sim, rating in similarities)
    sim_sum = sum(sim for sim, x in similarities)

    # extra check return avg is sum =0
    if sim_sum == 0:
        return float(round(business_avg_ratings.get(bus_id, default)))

    #predict
    predicted = round(weighted_sum / sim_sum)

    #constrain to range
    if predicted < 1.0:
        return float(1.0)
    elif predicted > 5.0:
        return float(5.0)
    else:
        return float(predicted)

def rsme(output_file_name, ground_truth):
    '''
    formula: sqrt [(Σ(Pi – Oi)²) / n]
    '''

    #get ratings from each file
    #format : (user_id, business_id) : rating
    predictions = sc.textFile(output_file_name) \
        .filter(lambda line: not line.startswith("user_id")) \
        .map(lambda line: line.split(',')) \
        .map(lambda x: ((x[0], x[1]), float(x[2])))

    actual = sc.textFile(ground_truth) \
        .filter(lambda line: not line.startswith("user_id")) \
        .map(lambda line: line.split(',')) \
        .map(lambda x: ((x[0], x[1]), float(x[2])))

    # calc squared differences
    squared_errors = predictions.join(actual) \
        .map(lambda x: (x[1][0] - x[1][1]) ** 2)

    # rmse
    n = squared_errors.count()
    if n == 0:
        return 0

    rmse = math.sqrt(squared_errors.sum() / n)

    return rmse

def main():

    # process train and test data
    train_rdd = sc.textFile(train_file)
    train_data = (
        train_rdd
            .filter(lambda line: not line.startswith("user_id"))  # no header
            .map(lambda line: line.split(','))
            .map(lambda x: (x[0], x[1], float(x[2])))  # format: (user_id, business_id, rating)
            .cache()
    )

    # test data
    test_rdd = sc.textFile(test_file)
    test_data = (
        test_rdd
            .filter(lambda line: not line.startswith("user_id"))  # no header
            .map(lambda line: line.split(','))
            .map(lambda x: (x[0], x[1]))  # format: (user_id, business_id)
    )

    #dictionaries for easy look up during prediciton

    # format (nested dictionary): {bus_id1 : {user_id1 : rating1, user2 : rating2}...}
    business_users = (train_data
        .map(lambda x: (x[1], {x[0]: x[2]}))
        .reduceByKey(lambda x, y: {**x, **y})      #combine the two dictionaries
        .collectAsMap())

    # format (nested dictionary): {user_id1 : {bus_id1 : rating1, bus_id2 : rating2}...}
    user_businesses = (train_data
        .map(lambda x: (x[0], {x[1]: x[2]}))
        .reduceByKey(lambda x, y: {**x, **y})       #combine the two dictionaries
        .collectAsMap())        #dicitonary

    #format: {bus_id : avg_rating, bus_id2 : avg_rating...}
    # use for default vals
    business_avg_ratings = (train_data
        .map(lambda x: (x[1], (x[2], 1)))               #map bus_id : (rating, 1)
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))           #add ratings and rating counts
        .mapValues(lambda x: x[0] / x[1])               #avg rating
        .collectAsMap())                            #dicitonary

    #store all calculated similarities- reduce run time
    calculated_sims = {}

    #tuned params
    neighbors = 18
    default = 3

    predictions = test_data.map(lambda x: (
            x[0],  #user id
            x[1],  #bus id
            prediction(x[0], x[1], user_businesses, business_users, business_avg_ratings, calculated_sims, neighbors, default) ) )\
        .collect()

    # write output
    with open(output_file, 'w') as f:
        f.write('user_id,business_id,prediction\n')         #header
        for user_id, business_id, p in predictions:
            f.write(f'{user_id},{business_id},{p}\n')

    '''
    error = rsme(output_file, test_file)
    print("RSME: ", error)
    '''

    sc.stop()

if __name__ == "__main__":
    start_time = time.time()

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    #env set up
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    sc = SparkContext('local[*]', 'hw3task2_1')
    sc.setLogLevel("ERROR")

    main()

    end = time.time()
    duration = end - start_time

    print("Duration: " + str(duration))
