import tensorflow as tf
import numpy as np
import pandas as pd

_CSV_COLUMNS = [
    'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]

_TEST_CSV_VOLUMNS = [
    'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
]

_CSV_COLUMN_DEFAULTS=[[1.1],[0],[0],[''],[''],[1.1],[0],[0], [''],[1.1],[''],['']]
TEST_CSV_COLUMN_DEFAULTS=[[1.1],[0],[''],[''],[1.1],[0],[0], [''],[1.1],[''],['']]

def test_input_fn(data_file, num_epochs, shuffle, batch_size):
    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=TEST_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_TEST_CSV_VOLUMNS, columns))
        features.pop('PassengerId')
        features.pop('Name')
        return features
  # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file, buffer_size=1024)

#  if shuffle:
#    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    dataset = dataset.shuffle(1000).map(parse_csv, num_parallel_calls=500)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    return features


def input_fn(data_file, num_epochs, shuffle, batch_size, predic=False):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    features.pop('PassengerId')
    features.pop('Name')
    if not predic:
        labels = features.pop('Survived')
    else:
        labels = None
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file, buffer_size=1024)

#  if shuffle:
#    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.map(parse_csv, num_parallel_calls=500)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()

  return features, labels


def build_model_columns():
    age = tf.feature_column.numeric_column('Age')
    fare = tf.feature_column.numeric_column('Fare',dtype=float)
    pclass = tf.feature_column.categorical_column_with_identity('Pclass', 6, default_value=5)
    sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])
    sibsp = tf.feature_column.categorical_column_with_identity('SibSp', 11, default_value=10)
    parch = tf.feature_column.categorical_column_with_identity('Parch', 11, default_value=10)
    ticket = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', ['A/5', 'PC', 'STON/O2.', 's', 'PP', 'A/5.', 'C.A.', 'A./5.',
       'SC/Paris', 'S.C./A.4.', 'A/4.', 'CA', 'S.P.', 'S.O.C.', 'SO/C',
       'W./C.', 'SOTON/OQ', 'W.E.P.', 'STON/O', 'A4.', 'C', 'SOTON/O.Q.',
       'SC/PARIS', 'S.O.P.', 'A.5.', 'Fa', 'CA.', 'F.C.C.', 'W/C',
       'SW/PP', 'SCO/W', 'P/PP', 'SC', 'SC/AH', 'A/S', 'A/4', 'WE/P',
       'S.W./PP', 'S.O./P.P.', 'F.C.', 'SOTON/O2', 'S.C./PARIS',
       'C.A./SOTON'])
#     fare = tf.feature_column.categorical_column_with_hash_bucket('Fare', hash_bucket_size=1000)
    cabin = tf.feature_column.categorical_column_with_hash_bucket('Cabin', hash_bucket_size=200)
    embarded = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
    
    age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[9, 19, 26, 30, 35, 42, 53, 60, 80])
    base_columns = [
        age_buckets, pclass, sibsp, parch, ticket, sex, fare, cabin, embarded
    ]
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['SibSp', 'Parch'], hash_bucket_size=100
        ),
        tf.feature_column.crossed_column([age_buckets, 'Sex'], hash_bucket_size=100),
        tf.feature_column.crossed_column(['Pclass', 'Sex'], hash_bucket_size=1000),
        # tf.feature_column.crossed_column(['Embarked', 'SibSp'], hash_bucket_size=100),
    ]
    wide_columns = base_columns + crossed_columns
    deep_columns = [
        age,
        tf.feature_column.indicator_column(pclass),
        tf.feature_column.indicator_column(sibsp),
        tf.feature_column.indicator_column(parch),
        tf.feature_column.indicator_column(cabin),
        tf.feature_column.indicator_column(age_buckets),
        tf.feature_column.indicator_column(ticket),
        tf.feature_column.indicator_column(sex),
        tf.feature_column.indicator_column(embarded)
    ]
    return wide_columns, deep_columns



def build_estimator(model_dir, model_type):
    print('build estimator')
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [200, 150, 100, 50]
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU':0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)
    
def main(unused_argv):   
    model_dir = '/Users/zhangyong/Downloads/model_dir/'
    model_type= ''
    model = build_estimator(model_dir, model_type)
    for n in range(40):
        model.train(input_fn=lambda: input_fn('/Users/zhangyong/dataset/titannic/train_data.csv', 20, True, 40))
    
    results = model.evaluate(input_fn=lambda: input_fn('/Users/zhangyong/dataset/titannic/test_data.csv', 1, False, 40))
    predicts = model.predict(input_fn=lambda: input_fn('/Users/zhangyong/dataset/titannic/test.csv', 2, True, 20, True))

    print('Results at epoch', (n + 1) * 2)
    print('-' * 60)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
        
tf.app.run(main=main, argv=['aa'])