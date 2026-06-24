#!/usr/bin/env python

import joblib
import os
import os.path as osp
import tensorflow as tf
from safe_rl.utils.logx import restore_tf_graph


def _save_dir_has_saved_model(save_dir):
    return osp.exists(osp.join(save_dir, 'saved_model.pb')) or osp.exists(osp.join(save_dir, 'saved_model.pbtxt'))


def _get_valid_save_iters(fpath):
    """Return sorted iteration numbers for simple_save dirs with a SavedModel file."""
    valid = []
    for name in os.listdir(fpath):
        if not name.startswith('simple_save'):
            continue
        suffix = name[11:]
        if not suffix.isdigit():
            continue
        save_dir = osp.join(fpath, name)
        if _save_dir_has_saved_model(save_dir):
            valid.append(int(suffix))
    return sorted(valid)


def _select_itr_to_load(fpath, itr):
    valid_iters = _get_valid_save_iters(fpath)

    if len(valid_iters) == 0:
        raise IOError('No valid simple_save* directories with saved_model.pb(.txt) found under %s' % fpath)

    # load most recent valid checkpoint
    if itr == 'last':
        return valid_iters[-1]

    requested = int(itr)
    if requested in valid_iters:
        return requested

    # prefer the nearest earlier checkpoint to avoid using future policy params
    earlier = [i for i in valid_iters if i < requested]
    if len(earlier) > 0:
        chosen = earlier[-1]
    else:
        chosen = valid_iters[0]

    print('Requested simple_save%d is unavailable or incomplete; falling back to simple_save%d.' % (requested, chosen))
    return chosen

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from (robust to incomplete simple_save dirs)
    selected_itr = _select_itr_to_load(fpath, itr)
    itr = '%d' % selected_itr

    # load the things!
    sess = tf.Session(graph=tf.Graph())
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, sess