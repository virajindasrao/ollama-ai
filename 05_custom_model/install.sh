yum install python3-pip -y
pip3 install --upgrade pip
pip uninstall torch torchvision -y
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
