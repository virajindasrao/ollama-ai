sudo yum install python3-pip -y
sudo yum install pip -y
pip install --upgrade pip
pip uninstall torch torchvision -y
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python3 05_custom_model/fine_tune_model.py --model_name mistralai/Mistral-7B-v0.1 --data_path 05_custom_model/train-data.json --output_dir 05_custom_model/fine_tuned_model --epochs 3 --batch_size 8 --learning_rate 5e-5
pip install transformers==4.49.0
pip install psutil
pip install numpy==1.24.3
pip install pybind11>=2.12