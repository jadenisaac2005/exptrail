# Digits MLP example

A 64→64→10 NumPy MLP (no framework) trained on sklearn's `load_digits`, with three
optimiser configs. Everything in the table below is generated from `runs/` and is
checked in CI with `exptrail verify examples/digits/README.md`.

```bash
pip install exptrail numpy scikit-learn
python examples/digits/train.py                      # trains 3 configs, rewrites the table
exptrail --root examples/digits/runs ls
exptrail --root examples/digits/runs compare sgd-lr0.1 momentum-lr0.01 momentum-lr0.05
exptrail --root examples/digits/runs plot sgd-lr0.1 momentum-lr0.01 momentum-lr0.05 --metric val_acc
exptrail verify examples/digits/README.md
```

## Results

