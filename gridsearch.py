import main
from argparse import Namespace
from copy import deepcopy



def grid_search(args: Namespace, grid: dict[str, list]) -> None:
    total_runs = main.np.prod([len(v) for k, v in grid.items()]).item()
    print(total_runs, "total runs")

    for arg_name, values in grid.items():
        if len(values) == 0:
            return
        setattr(args, arg_name, values[-1])
    main.train_agent(args)

    for arg_name, values in grid.items():
        popped_value = values.pop(-1)
        grid_search(args, grid)
        values.append(popped_value)

def line_search(args: Namespace, grid: dict[str, list]) -> None:
    total_runs = sum([len(v) for k, v in grid.items()]) + 1
    print(total_runs, "total runs")
    print(f"[1/{total_runs}] defaults")
    main.train_agent(args)

    run = 1
    for arg_name, arg_values in grid.items():
        default_value = getattr(args, arg_name)
        print()
        for value in arg_values:
            run += 1
            setattr(args, arg_name, value)
            print(arg_name, value)
            print(f"[{run}/{total_runs}] {arg_name} {value}")
            main.train_agent(args)
        setattr(args, arg_name, default_value)


if __name__ == "__main__":

    grid = {
        "learning_rate": [3e-6, 3e-5, 3e-4, 3e-3, 3e-2],
        "batch_size": [32, 64, 128, 256, 512],
        "min_train_size": [512, 1024, 2048, 4096],
        "lambd": [0.9, 0.95, 1.],
        "clip_epsilon": [0.1, 0.15, 0.2, 0.25, 0.3],
        "entropy_reg": [0.0001, 0.001, 0.01, 0.1],
        "last_layer_init_scale": [0.001, 0.01, 0.1, 1.],
    }

    args = main.parser.parse_args()
    setattr(args, "eval_each", args.max_steps)
    line_search(args, grid)
    # grid_search(args, grid)


