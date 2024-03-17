import os
import pandas as pd
import shutil
import logging
from tqdm import tqdm

def get_genres(path):
    """
    This function reads the genre labels and puts it into a pandas DataFrame.
    
    @input path: The path to the genre label file.
    @type path: String
    
    @return: A pandas dataframe containing the genres and midi IDs.
    @rtype: pandas.DataFrame
    """
    ids = []
    genres = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                [x, y, *_] = line.strip().split("\t")
                ids.append(x)
                genres.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
    return genre_df


logging.basicConfig(filename='../logs/refactor_dataset.log', level=logging.DEBUG)

dataset_path = "../raw_data/New_Data/lmd_matched/lmd_matched"
genres_path = "../sources/msd_tagtraum_cd1.cls"
raw_dataset_folder_path = "../genre_classified_dataset"


genre_df = get_genres(genres_path)

raw_track_count = 0
original_data_track_count = 0

                    
if not os.path.exists(raw_dataset_folder_path):

    os.makedirs(raw_dataset_folder_path)


for first_initial in tqdm(os.listdir(dataset_path)):

    for second_initial in os.listdir(f"{dataset_path}/{first_initial}"):

        for third_initial in os.listdir(f"{dataset_path}/{first_initial}/{second_initial}"):

            for track_id in os.listdir(f"{dataset_path}/{first_initial}/{second_initial}/{third_initial}"):

                original_data_track_count += len(f"{dataset_path}/{first_initial}/{second_initial}/{third_initial}/{track_id}")

                track_genres = genre_df.loc[genre_df['TrackID'] == track_id, 'Genre'].values

                if(len(track_genres) != 0):

                    for track in os.listdir(f"{dataset_path}/{first_initial}/{second_initial}/{third_initial}/{track_id}"):
                            
                        raw_track_count += 1

                        for genre in track_genres:

                            if not (os.path.exists(f"{raw_dataset_folder_path}/{genre}")):
                                os.makedirs(f"{raw_dataset_folder_path}/{genre}")

                            source_path = f"{dataset_path}/{first_initial}/{second_initial}/{third_initial}/{track_id}/{track}"
                            destination_path = f"{raw_dataset_folder_path}/{genre}/"

                            try:

                                shutil.copy(source_path, destination_path)

                            except PermissionError:
                                
                                if os.path.exists(destination_path):
                                    try:
                                        os.chmod(destination_path, 0o777)                                        
                                        shutil.copy(source_path, destination_path)

                                    except Exception as e:
                                        logging.error(f"Failed to change permissions or copy file: {e}")
                                else:
                                    logging.error("Permission denied and the file does not exist to change permissions.")       

logging.info(f"Refactor completed with {raw_track_count} tracks")
logging.info(f"Original track amount: {original_data_track_count}")