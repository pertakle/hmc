from gymnasium.envs.registration import register

# Initialization:
#
# gym.make() will make the RubiksCubeEnv
# gym.make_vec(vectorization_mode=mode)
#
# Mode:
#   - None | "hmc.env.cube_env:RubiksCubeEnvVec" will make RubiksCubeEnvVec
#   - "sync" | "async" will make a gym wrapper of RubiksCubeEnv

register(
    id="hmc/LightsOut-v0",
    entry_point="hmc.env.lights_out:LightsOut",
    vector_entry_point="hmc.env.lights_out:LightsOutVec",
    disable_env_checker=False,
)

register(
    id="hmc/RubiksCube-v0",
    entry_point="hmc.env.cube_env:RubiksCubeEnv",
    vector_entry_point="hmc.env.cube_env:RubiksCubeEnvVec",
    disable_env_checker=False,
)
register(
    id="hmc/TorchRubiksCube-v0",
    entry_point="hmc.env.torch_cube_env:TorchRubiksCubeEnv",
    vector_entry_point="hmc.env.torch_cube_env:TorchRubiksCubeEnvVec",
    disable_env_checker=False,
)
