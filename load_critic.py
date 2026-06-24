import tensorflow as tf
import numpy as np
import os

def load_critic(checkpoint_dir, hidden_sizes=(256, 256), obs_dim=None):
    """
    Load the cost critic (vc) from saved checkpoints.
    
    Args:
        checkpoint_dir: Path to the critic_checkpoints directory
        hidden_sizes: Must match what was used during training (256, 256)
        obs_dim: Observation space dimension (will be inferred if None)
    """
    
    # ── 1. Build the same MLP graph used during training ──────────────────────
    tf.reset_default_graph()
    
    # Placeholder for observations — shape inferred from checkpoint if obs_dim unknown
    obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim), name='obs_input')
    
    def mlp(x, hidden_sizes, activation=tf.tanh, output_activation=None):
        for size in hidden_sizes[:-1]:
            x = tf.layers.dense(x, size, activation=activation)
        return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)
    
    # Recreate critic scopes exactly as in mlp_actor_critic
    with tf.variable_scope('pi'):
        pass  # Needed so variable numbering matches the saved checkpoint
    
    with tf.variable_scope('vf'):
        v = tf.squeeze(
            mlp(obs_ph, list(hidden_sizes) + [1], tf.tanh, None),
            axis=1
        )
    
    with tf.variable_scope('vc'):
        vc = tf.squeeze(
            mlp(obs_ph, list(hidden_sizes) + [1], tf.tanh, None),
            axis=1
        )
    
    # ── 2. Restore from checkpoint ────────────────────────────────────────────
    saver = tf.train.Saver()
    sess = tf.Session()
    
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
    
    print(f"Restoring from: {ckpt}")
    saver.restore(sess, ckpt)
    
    return sess, obs_ph, v, vc


def predict_cost_critic(sess, obs_ph, vc, observations):
    """
    Run inference on the cost critic.
    
    Args:
        observations: np.array of shape (N, obs_dim)
    Returns:
        cost values: np.array of shape (N,)
    """
    if observations.ndim == 1:
        observations = observations[np.newaxis, :]  # Add batch dim
    
    return sess.run(vc, feed_dict={obs_ph: observations})


# ── Main usage ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    
    CHECKPOINT_DIR = '/home/ed21b059/ddp/data_new/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s42/critic_checkpoints'
    
    # PointGoal1 observation dim — check your env or infer from checkpoint
    import gym, safety_gym
    env = gym.make('Safexp-PointGoal1-v0')
    OBS_DIM = env.observation_space.shape[0]
    print(f"Observation dim: {OBS_DIM}")
    env.close()
    
    # Load
    sess, obs_ph, v, vc = load_critic(CHECKPOINT_DIR, hidden_sizes=(256, 256), obs_dim=OBS_DIM)
    
    # Test with a random observation
    # dummy_obs = np.random.randn(1, OBS_DIM).astype(np.float32)
    
    # value      = sess.run(v,  feed_dict={obs_ph: dummy_obs})
    # cost_value = sess.run(vc, feed_dict={obs_ph: dummy_obs})
    
    # print(f"Value (v):      {value}")
    # print(f"Cost value (vc): {cost_value}")
    obs = env.reset()
    done = False
    values, cost_values = [], []

    while not done:
        o = obs[np.newaxis, :].astype(np.float32)
        v_val  = sess.run(v,  feed_dict={obs_ph: o})[0]
        vc_val = sess.run(vc, feed_dict={obs_ph: o})[0]
        values.append(v_val)
        cost_values.append(vc_val)
        
        obs, reward, done, info = env.step(env.action_space.sample())

    print(f"Mean value:       {np.mean(values):.4f}")
    print(f"Mean cost value:  {np.mean(cost_values):.4f}")
    print(f"Rollout length:   {len(values)} steps")
    
    sess.close()