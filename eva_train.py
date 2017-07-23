"""A binary to train EVA using a single GPU.

Accuracy:
eva_train.py achieves % accuracy after K steps (256 epochs of
data) as judged by eva_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

"""

# Config
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime

import tensorflow as tf

import Utils.Eva_config_consts as config
import eva
from Storage.TFStorage import TFStorage, TFStorageOpenOptions

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/eva_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('save_checkpoint_secs', 600,
                            """The frequency, in seconds, that a checkpoint is saved.""")


def train():
    """Train EVA for a number of steps."""
    with tf.Graph().as_default(), TFStorage(config.DATESET_FILE_PATH(), TFStorageOpenOptions.READ) as storage:
        #global_step = tf.contrib.framework.get_or_create_global_step()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Get sound features and phonemes and speakers for EVA.
        sound_features, phonemes, speakers = storage.inputs(config.BATCH_SIZE, shuffle=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = eva.inference(sound_features, train=True)

        # Calculate loss.
        loss = eva.loss(logits, speakers)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = eva.train(loss, global_step)

        # Initialize variables
        init = tf.global_variables_initializer()

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:

            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Model was restored')
            else:
                mon_sess.run(init)

            # Required for getting an input batch from the storage
            tf.train.start_queue_runners(sess=mon_sess)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    #if tf.gfile.Exists(FLAGS.train_dir):
    #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
