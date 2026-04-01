26.3.17

训练路径常碰壁，与reward的设计有关。

修改：

1. env中奖励函数中不涉及边界碰撞产生的惩罚
2. 将AUV起始状态，target坐标，障碍物坐标均变为固定的

Result:

1. GPU3060，训练2000轮耗时16h，采用obstacles0，max_iteration=2000

26.3.18

训练后期无法正确转向，会沿直线前进直到碰壁。

修改：

1. 增加模型保存的路径，保存actor和critic
2. actor的激活函数由softsign改为了tanh
3. 动作采样时的高斯噪声explore_noise_range从0.1增大为了0.2

Result:

1. GPU3060，训练2000轮，采用obstacles1，max_iteration=20000，success rate = 71%，初始状态（0，0，0，0，0）
2. GPU3060，训练2000轮，采用obstacles0，max_iteration=20000，success rate = 14%，初始状态（0，0，0，0，0）耗时4.5h

26.3.20

AUV的操纵能力不行，总是在任务后期出现不必要的抬头，导致AUV的朝向出现问题，就容易冲过目标点，只要冲过目标点后，AUV就无法调整回来了。

修改：

1. 扩大env的边界范围，[500,500,500]，并将AUV的到达判定范围规定为15，碰撞半径改为了5，危险阈值改为了15，障碍物半径变为了[10,16]
2. 软更新系数从0.002增大为0.005
3. 障碍物的位置和target的位置随之变化





















训练指令：

##### 从头开始训练（使用 config.py 中的默认 episode 数）
python main_ai.py

##### 从头训练，但自定义 episode 数
python main_ai.py --train_episodes 1000

##### 加载最新模型权重，继续训练
python main_ai.py --load

##### 加载指定 episode 的 checkpoint，继续训练

python main_ai.py --load --load_episode 200

##### 加载最新模型，评估 50 个 episode（默认数量）
python main_ai.py --evaluate --load

##### 加载最新模型，自定义评估 episode 数
python main_ai.py --evaluate --load --eval_episodes 100

##### 加载指定 episode 的 checkpoint 进行评估
python main_ai.py --evaluate --load --load_episode 500

##### ❌ 错误用法：评估模式必须带 --load，否则会报错退出
python main_ai.py --evaluate
