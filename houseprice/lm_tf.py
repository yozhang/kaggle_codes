import pandas as pd
import tensorflow as tf 
import shutil
import sys 
import argparse



_CSV_COLUMNS = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice']

#  Continus Columns 
continus_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF',
'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF2', 'Fireplaces','GarageArea', 'GarageCars',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','GarageYrBlt'
]

type_columns = ['LandContour', 'BsmtFinType1', 'GarageFinish', 
'MSZoning', 'SaleCondition', 'Fence', 'Condition1', 'BsmtExposure', 
'BsmtFinType2', 'Utilities', 'ExterCond', 'BldgType', 'RoofMatl', 
'LandSlope', 'Street', 'RoofStyle', 'YearBuilt', 'Exterior2nd', 'BsmtQual', 
'GarageType', 'Neighborhood', 'LotConfig', 'Electrical', 'PoolQC', 'ExterQual', 
'GarageQual', 'Alley', 'KitchenQual', 'Exterior1st', 'YrSold', 
'YearRemodAdd',  'MasVnrType', 'FireplaceQu', 'Condition2', 'LotShape', 
'PavedDrive', 'MiscFeature', 'HouseStyle', 'Foundation', 'HeatingQC', 
'MSSubClass', 'MoSold',  'OverallCond', 'OverallQual', 'TotRmsAbvGrd',
'CentralAir', 'SaleType', 'Heating', 'BsmtCond', 'GarageCond', 'Functional']


_DEFAULT_VALUES= [[''], [''], [''], [0.0], [0.0], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [0.0], [''], [''], [''], [''], [''], [''], [''], [0.0], [''], [0.0], [0.0], [0.0], [''], [''], [''], [''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [0.0], [''], [''], [0.0], [''], [0.0], [0.0], [''], [''], [''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [0.0], [''], [''], [''], [''], ['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/Users/zhangyong/Downloads/tmp',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/Users/zhangyong/projects/kaggle_codes/houseprice/data/train_data.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/Users/zhangyong/projects/kaggle_codes/houseprice/data/test_data.csv',
    help='Path to the test data.')


def columnValues(input_path):
    a_path = '/Users/zhangyong/projects/kaggle_codes/houseprice/data/train.csv'
    dat = pd.read_csv(a_path)
    column_values = {}
    for column in type_columns:
        column_values[column] = dat[column].dropna().apply(lambda x: str(x)).unique()
    return column_values



def build_model_columns(type_values):
    base_columns = []
    for column in continus_columns:
        feature = tf.feature_column.numeric_column(column)
        print('yonzhang:' + column)
        base_columns.append(feature)

    for column in type_columns:
        print('zhangyong:' + column)
        feature = tf.feature_column.categorical_column_with_vocabulary_list(
            column, type_values[column]
        )
        base_columns.append(feature)

    return base_columns


def build_estimator():
    t_values = columnValues(FLAGS.train_data)
    columns = build_model_columns(t_values)
    run_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU': 0})
    )
    return tf.estimator.LinearRegressor(model_dir=FLAGS.model_dir, feature_columns=columns)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    def parse_csv(value):
        print('parse csv')
        columns = tf.decode_csv(value, record_defaults=_DEFAULT_VALUES)
        features = dict(zip(_CSV_COLUMNS, columns))
        features.pop('Id')
        labels = features.pop('SalePrice')
        print(labels)
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_csv, num_parallel_calls=5).repeat(num_epochs).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def main(unused_argv):
    model = build_estimator()
    for n in range(FLAGS.train_epochs):
        model.train(input_fn=lambda: input_fn(FLAGS.train_data, 
        FLAGS.epochs_per_eval, True, FLAGS.batch_size))
        result = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
          print('%s: %s' % (key, results[key]))
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv = [sys.argv[0]] + unparsed)