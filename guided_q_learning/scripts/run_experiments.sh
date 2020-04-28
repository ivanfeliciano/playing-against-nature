EPISODES=500
ST=one_to_one

python app.py --num 5 --structure $ST --episodes $EPISODES
python app.py --num 5 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 7 --structure $ST --episodes $EPISODES
python app.py --num 7 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 9 --structure $ST --episodes $EPISODES
python app.py --num 9 --structure $ST --stochastic --episodes $EPISODES

ST=one_to_many


python app.py --num 5 --structure $ST --episodes $EPISODES
python app.py --num 5 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 7 --structure $ST --episodes $EPISODES
python app.py --num 7 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 9 --structure $ST --episodes $EPISODES
python app.py --num 9 --structure $ST --stochastic --episodes $EPISODES


ST=many_to_one

python app.py --num 5 --structure $ST --episodes $EPISODES
python app.py --num 5 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 7 --structure $ST --episodes $EPISODES
python app.py --num 7 --structure $ST --stochastic --episodes $EPISODES

python app.py --num 9 --structure $ST --episodes $EPISODES
python app.py --num 9 --structure $ST --stochastic --episodes $EPISODES
