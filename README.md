# Miniproject for BIOENG-456: Controlling behavior in animals and robots

Welcome to the Miniproject for BIOENG-456!

## Setup
`pynput` is required for the KeyBoardController in `explore_levels.py`. To install it, run:
```bash
conda activate flygym
pip install pynput
```

## Usage
To explore the levels interactively, run the `explore_levels.py` script:
```bash
conda activate flygym
python explore_levels.py --level <level> --seed <seed>
```
Replace `<level>` with the desired level number (0 to 4 for the 5 levels, -1 for just a flat terrain) and `<seed>` with the random seed for reproducibility.

The `run_simulation.py` script contains code that will be used to evaluate the controller.

To run the simulation, use:
```bash
conda activate flygym
python run_simulation.py <submission_folder> --level <level> --seed <seed> --progress --max-steps 10000
```

Replace `<submission_folder>` with the path to your submission folder. An example submission folder is provided in the `submission` directory. It contains a controller that walks forward regardless of the observations it receives. You may use that as a starting point for your own controller.

A `test_controller.ipynb` notebook is provided to help you test your controller. You may for example use it to get the list of observations over time and plot them when designing your controller.

## Submission
The submission folder should contain the following files:
- controller.py: The controller code. It should contain a class named `Controller` that subclasses `BaseController` and implements the `get_action` method and `done_level` method.
- __init__.py: A file to mark the directory as a package.
- any other files you need for your controller.

You may run the `check_submission.py` script to check if your submission folder is valid and zip it for submission:
```bash
conda activate flygym
python check_submission.py <submission_folder>
```