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

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file (ln:4). 

You can change different parameters in the ```params.py``` file.

You can train the model in any custom dataset other than IAM and CVL. The process involves creating a ```dataset_name.pickle``` file and placing it inside ```files``` folder. The structure of ```dataset_name.pickle``` is a simple python dictionary. 

```python
{
'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
}
```


