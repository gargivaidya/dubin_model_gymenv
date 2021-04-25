# OpenAI Gym Compatible Dubin's Car Model

## Virtual env dependencies</br>
(You will have to install some packages using pip3)</br>
pip3 install numpy matplotlib gym tensorboard</br>
python -m pip install --upgrade setuptools</br>
pip3 install <check compatible version of torch> </br>

## Dependencies </br>
gym - 0.18.0 </br>
numpy - 1.19.5 </br>
matplotlib - 3.3.4 </br>
pytorch - 1.8.1 </br>

### For evaluation of trained models ###
```python dubins_gym_evaluation.py```

### For implementation of navigating from random initialisation point on unit circle to origin
```python dubins_randomized_AtoB.py```

### For implementation using custom algorithms for tracking given waypoints in a trajectory
```python dubin_gymenv.py```

### For implementation using Stable Baselines
```python dubins_gym.py```

Dubin's Visualization Model source - https://github.com/AtsushiSakai/PythonRobotics

