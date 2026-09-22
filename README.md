# narmax-matlab-neural-network-toolbox

This repository demonstrates how to train NARX and NARMAX for multi-step prediction

* Acronym list is shown below.

    | Acronyms: |
    | ------------ |
    | NARX   - Nonlinear Autoregressive eXogenous. |
    | NARMAX - Nonlinear Autoregressive Moving Average eXogenous. |

**NOTE: This repository is a work in progress. Also the training examples will be cleaned up in the future.**

## Software Requirement:
MATLAB R2018b and Deep Learning Toolbox Version 12.0.

## Project Layout:
* **Training Files:**
Contains examples for training NARX and NARMAX in MATLAB for time-series modeling on real-life data.

    | Example Files: | 
    | ------------- |
    | Training NARX on Mag-Lev Data Set: [**train_narx_maglev.m**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/train_narx_maglev.m) |
    | Training NARX on Servo-Motor Data Set: [**train_narx_servomotor.m**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/train_narx_servomotor.m) |
    | Training NARMAX on Mag-Lev Data Set: [**train_narmax_maglev.m**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/train_narmax_maglev.m) |
    | Training NARMAX on Servo-Motor Data Set: [**train_narmax_servomotor**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/train_narmax_servomotor.m) |

* **Models:**
Contains MATLAB implementation of NARX and NARMAX models for time-series modeling. Link to the models are shown below.

    | Models: |
    | ------------ |
    | NARX Model: [**NARXModel.m**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/NARXmodel.m) |
    | NARMAX Model: [**NARMAXModel.m**](https://github.com/joekelley120/narmax-matlab-neural-network-toolbox/blob/master/NARMAXmodel.m) |

## Training Sequences:
`NARXmodel` cuts the record into training sequences of `Horizon_Step + delay`
samples: the first `delay` samples are the warm-up that fills the tapped delay
lines, and the rest are the multi-step targets.

**The origins of those sequences are drawn at random** — the "randomly
overlapping" scheme of Kelley (2024). NARX and ARX both use it and there is no
switch to turn it off. As many origins are drawn as a non-overlapping tiling
would give, but placed uniformly rather than on a grid, which avoids both the
redundancy of a one-sample stride and the risk that a tiling lands every origin
at the same point of a periodic cycle. `sequenceSeed` (default 0) selects the
draw. See `prepare_data` in `NARXmodel.m`.

`NARMAXmodel` is a separate class and still tiles; it has not been changed.

