import tensorflow as tf 
import argparse
import lm_tf
import sys
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/home/zhangyong/Downloads/house_model',
    help='Base directory for the model.')

parser.add_argument(
    '--test_dir', type=str, default='/home/zhangyong/projects/kaggle_codes/houseprice/data/test.csv'
)

parser.add_argument(
    '--output_file', type=str, default='/home/zhangyong/Downloads/predict_out.csv'
)

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

_DEFAULT_VALUES= [[''], [''], [''], [0.0], [0], [''], [''], [''], [''], [''], 
[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], 
[''], [''], [''], [''], [''], [''], [0.0], [''], [''], [''], 
[''], [''], [''], [''], [0.0], [''], [0.0], [0.0], [0.0], [''], 
[''], [''], [''], [0], [0], [0], [0], [''], [''], [''], 
[''], [0], [''], [''], [''], [''], [''], [''], [''], [0.0], 
[''], [''], [0.0], [''], [''], [''], [0], [0], [0], [''], 
[0], [''], [''], [''], [''], [0], [''], [''], [''], [''] ]

def input_fn(data_file, num_epochs, shuffle, batch_size):
    def parse_csv(value):
        print('parse csv')
        columns = tf.decode_csv(value, record_defaults=_DEFAULT_VALUES)
        features = dict(zip(lm_tf._CSV_COLUMNS[:-1], columns))
        features.pop('Id')
        return features

    dataset = tf.data.TextLineDataset(data_file).skip(1)
    dataset = dataset.map(parse_csv, num_parallel_calls=5).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

def main(unused_argv):
    run_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU': 0})
    )
    t_values = lm_tf.columnValues(FLAGS.test_dir)
    columns = lm_tf.build_model_columns(t_values)
    model = tf.estimator.LinearRegressor(model_dir=FLAGS.model_dir, feature_columns=columns, config=run_config)
    results = model.predict(input_fn=lambda: input_fn(FLAGS.test_dir, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
    pre_values = []
    for l in results:
        pre_values.append(l['predictions'][0])
    dat = pd.read_csv(FLAGS.test_dir)
    out = pd.DataFrame(dat, columns=['Id'])
    out['SalePrice'] = pre_values
    out.to_csv(FLAGS.output_file, index=False)
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv = [sys.argv[0]] + unparsed)
    