python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
screen -S flask_app
flask run --host=0.0.0.0 --port=5000
Ctrl + A 然后按 D


screen -r flask_app

sudo netstat -tulnp | grep 5000
sudo kill -9 +进程号
