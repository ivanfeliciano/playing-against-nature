EPISODES=1000
ST=one_to_one
MOD=50

python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD
python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD
python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD


python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
