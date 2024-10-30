from gymnasium.envs.registration import register

register(
    id="deepercube/RubiksCube-v0",
    entry_point="deepercube.env.cube_env:RubiksCubeEnv",
    vector_entry_point="deepercube.env.cube_env:RubiksCubeEnvVec",
    disable_env_checker=False
)
