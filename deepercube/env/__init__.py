from gymnasium.envs.registration import register

# Initialization:
#
# gym.make() will make the RubiksCubeEnv
# gym.make_vec(vectorization_mode=mode)
#
# Mode:
#   - None | "deepercube.env.cube_env:RubiksCubeEnvVec" will make RubiksCubeEnvVec
#   - "sync" | "async" will make a gym wrapper of RubiksCubeEnv

register(
    id="deepercube/RubiksCube-v0",
    entry_point="deepercube.env.cube_env:RubiksCubeEnv",
    vector_entry_point="deepercube.env.cube_env:RubiksCubeEnvVec",
    disable_env_checker=False,
)
