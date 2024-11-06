from pyspark import SparkContext
import sys
import os
import time
import json
import numpy as np
from xgboost import XGBRegressor
from pyspark import StorageLevel

def load_business_data(sc, business_file):
    # business_id, stars, and review_count
    return sc.textFile(business_file) \
        .map(lambda line: json.loads(line)) \
        .map(lambda business: (
        business['business_id'],
        (float(business['stars']), float(business['review_count']))
    )) \
        .persist(StorageLevel.MEMORY_AND_DISK) \
        .collectAsMap()


def load_user_data(sc, user_file):
    # user_id, average_stars, review_count, and fans count
    return sc.textFile(user_file) \
        .map(lambda line: json.loads(line)) \
        .map(lambda user: (
        user['user_id'],
        (float(user['average_stars']), float(user['review_count']), float(user['fans']))
    )) \
        .persist(StorageLevel.MEMORY_AND_DISK) \
        .collectAsMap()


def extract_features(pair, user_data, business_data):
    user_id, bus_id = pair  # extract ids

    # user stats: (average_stars, review_count, fans)
    user_avg_stars, user_review_count, user_fans = user_data.get(user_id, (0, 0, 0))

    # bus stats: (stars, review_count)
    business_stars, business_review_count = business_data.get(bus_id, (0, 0))

    return [
        user_avg_stars,  # avg rating
        user_review_count,  # review count
        user_fans,  # fan count
        business_stars,  # bus avg rating
        business_review_count  # bus total review count
    ]


def main(sc, folder_, test_file_path, output_file_path):
    # set up file paths
    train_file_path = folder_ + "/yelp_train.csv"
    user_file = folder_ + "/user.json"
    business_file = folder_ + "/business.json"

    # load all data
    user_data = load_user_data(sc, user_file)
    business_data = load_business_data(sc, business_file)

    # parse and set up training data
    feature_rdd = (
        sc.textFile(train_file_path)
            .filter(lambda x: not x.startswith('user_id'))  # remove headers
            .map(lambda line: (line.split(',')[0], line.split(',')[1], float(line.split(',')[2])))
            .map(lambda x: (
            x[0],  # user_id
            x[1],  # business_id
            extract_features((x[0], x[1]), user_data, business_data),
            # format: (user_id, business_id, feature_list, rating)
            x[2]  # rating
        ))
    ).cache()

    # format: (feature_list, rating)
    feature_data = feature_rdd.map(lambda x: (x[2], x[3])).collect()
    X = np.array([x for x, y in feature_data])
    y = np.array([y for x, y in feature_data])

    # set up model

    model = XGBRegressor()
    model.fit(X, y)

    # test data
    test_data = (
        sc.textFile(test_file_path)
            .filter(lambda x: not x.startswith('user_id'))  # remove headers
            .map(lambda line: line.split(','))
            .map(lambda x: (
            x[0],  # user_id
            x[1],  # business_id
            extract_features((x[0], x[1]), user_data, business_data)
        ))
    ).cache()

    # user_id and business_id pairs
    test_pairs = test_data.map(lambda x: (x[0], x[1])).collect()

    # features for prediction
    X_test = np.array(test_data.map(lambda x: x[2]).collect())

    # Make predictions
    predictions = model.predict(X_test)

    # write to output file
    with open(output_file_path, 'w') as f:
        f.write('user_id,business_id,prediction\n')
        for (user_id, business_id), p in zip(test_pairs, predictions):
            f.write(f'{user_id},{business_id},{p}\n')


if __name__ == "__main__":
    start_time = time.time()

    folder_ = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # env set up
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    sc = SparkContext('local[*]', 'hw3task2_2')
    sc.setLogLevel("ERROR")

    main(sc, folder_, test_file, output_file)

    # duration
    end = time.time()
    duration = end - start_time
    print("Duration: " + str(duration))

    sc.stop()
