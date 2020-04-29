EPISODES=1000
MOD=50

ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD


time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic


ST=many_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD


time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD


time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
