<h2>🛠️ Installation Steps:</h2>

<p>1. Dependencies</p>

```
Navigate to the /ui folder. Execute 'npm install' command from a terminal.
```

<p>2. Data Preparation</p>

```
Dataset is already ready for training. If one needs to change the dataset then to extract sequences and their labels from the new dataset run "modify_dataset.py" (Assuming the dataset structure is as follows: Dataset_folder -> Genre1 Genre2 ... GenreX -> Song1 Song2 Song3)
```

<p>3. Training</p>

```
Tranied model already exists under /models folder. If one needs to train a new model run "train.py" after configuring the dataset
```

<p>4. Extracting random seeds</p>

```
If dataset is changed at any point run "random_row_selector" to generate new seed sequences according to the new dataset
```

<p>5. Inference/Starting the app</p>

```
Run 'main.py'. After that navigate to /ui folder and execute 'npm start' from that folder.
```
