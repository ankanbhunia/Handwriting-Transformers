Download Dataset files and models from https://drive.google.com/file/d/16g9zgysQnWk7-353_tMig92KsZsrcM6k/view?usp=sharing and unzip inside ```files``` folder. In short, run following lines in a bash terminal. 

```bash
git clone https://github.com/ankanbhunia/Handwriting-Transformers
cd Handwriting-Transformers
pip install --upgrade --no-cache-dir gdown
gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k && unzip files.zip && rm files.zip
```

To start training the model: run

```
python train.py
```

You can change different parameters in the ```params.py``` file.

