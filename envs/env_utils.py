import gym


def param_to_str(params):
    if not hasattr(params, "__len__"):
        return "{:.4f}".format(params)
    s = []
    for i in range(len(params)):
        s.append("{:.4f}".format(params[i]))
    return '_'.join(s)


def params_to_env_name(env, params):
    if params is None:
        return ["{}-v0".format(env)]
    return ["{}_{}-v0".format(env, param_to_str(p)) for p in params]


def get_env_names(env, train_params, test_params):
    train_envs = params_to_env_name(env, train_params)
    test_envs = params_to_env_name(env, test_params)

    return train_envs, test_envs


def get_env_id(envs):
    if type(envs) is list:
        return [env.unwrapped.spec.id for env in envs]
    else:
        return envs.unwrapped.spec.id


def initialize_envs(env_names, seed):
    envs = []
    for name in env_names:
        genv = gym.make(name)
        genv.seed(seed)
        genv.action_space.np_random.seed(seed)
        envs.append(genv)
    return envs


def close_envs(envs):
    return [env.close() for env in envs]


def get_envs(env_type, train_env_params, test_env_params):
    train_envs, test_envs = get_env_names(env_type, train_env_params, test_env_params)
    return train_envs, test_envs
