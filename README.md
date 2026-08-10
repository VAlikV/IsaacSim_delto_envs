# IsaacSim_delto_envs
IsaacLab envs with tesolo delto for RL and control experiments

## Интеграция

1. Переместить файлы из `robots/` в `<директория IsaacLab>/robots/dg5f_right/`

2. Переместить файлы из `envs/` в `<директория IsaacLab>/scripts/my_examples/` (что-то типо)

3. Переместить файлы из `tasks/` в `<директория IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct`

4. Распаковать архив с весами из `weights/` в `<директория IsaacLab>/logs/rl_games/`

## Запуск среды

```bash
./isaaclab.sh -p scripts/my_examples/tesolo_delto_UR_env/setup_env.py 
```

## Запуск задачи

```bash
# Train
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task <TASK_NAME> --headless --num_envs 4096 --wandb-entity rubitek_dextrous --wandb-project-name RubetekDextorousDirectEnv --wandb-name UR10-Tessolo-v21-ForceRew --track

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task <TASK_NAME> --num_envs 4096 --headless

# Val
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task <TASK_NAME> --num_envs 16

```

Флаги:
- `--checkpoint /ABS/PATH/TO/nn/last.pth`
- `--video --video_length 1000 --video_fps 30`

## Описание

1. `envs/` - среды isaaclab

2. `robots/` - usd файлы роботов

3. `scripts` - скрипты для управленяи

This project uses USD models from:

https://github.com/tesollodelto/delto_m_ros2

Licensed under BSD-3-Clause.
