# written by Jannek Ulm at 06.03.2025

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import os
from tqdm import tqdm
import re
"""
This script contains the functions to load the datasets. 


We first start with a modified version of the BabyLM dataset, which is a collection of 100M stories. 
We replace the [Childes, BNC, Switchboard] part with the [TinyStories] part. 
We hence need 38M words of TinyStories to replace the 38M words of BabyLM. 

Hence our dataset then consists of 100M-38M=62M stories from BabyLM and 38M stories from TinyStories. 

1. Gutenberg: 26M 
2. SimpleWiki: 14.3M
3. OpenSubtitles: 16.8M
4. TinyStories: 38M

We name this dataset TinyBabyLM. 

"""

def count_words_in_dataset(dataset):
    num_rows = dataset.num_rows
    print(f"Number of rows: {num_rows}")
    count = 0
    for i in tqdm(range(num_rows)):
        line = dataset[i]["text"]
        count += len(line.split(' '))
    return count


def load_babylm_dataset_parts(data_path, split: str, source: str, apply_line_grouping: bool = False, grouping_delimiter_strategy: str = "= = =",remove_delimiters: bool = False):
    # the original babylm data comes in the format of large text files, it is depending on the source, how the lines, are structured and how their correlation is wrt to each other, 
    # we hence need to regroup some of the lines to form a story/document/paragraph ... 
    # the corpus should be a collection of these stories/documents/paragraphs ... instead of a collection of lines

    # possible strategies for grouping the lines: (only applied if apply_line_grouping is True)
    # 1. "= = =" : One common thing is that articles are seperated by a line: = = = ... = = =
    # 2. "empty line": A empty line could be used as a delimiter
    assert apply_line_grouping in [True, False], "apply_line_grouping must be True or False"
    assert grouping_delimiter_strategy in ["= = =", "empty line"], "grouping_delimiter_strategy must be '= = =' or 'empty line'"

    # source is either "bnc", "wiki", ....
    data_path += "osf_download24/ad7qg/osfstorage/text_data"

    source_names = ["bnc_spoken","childes","gutenberg","open_subtitles","simple_wiki","switchboard"]
    assert source in source_names, "Source not found"

    # folder name which contains all the the files, with their respective names and the encdings ()
    split_ending_mapping = {"dev": "dev", "test": "test", "train_10M": "train", "train_100M": "train"}
    
    assert split in split_ending_mapping, "Split not found"

    dataset_path = os.path.join(data_path, split, source+"."+split_ending_mapping[split])
    # load the dataset
    data_set_name = "babylm_"+source+"_"+split

    if apply_line_grouping:
        
        grouped_dataset = []

        # start with loading the large txt file which contains all the lines at the dataset path
        with open(dataset_path, "r") as file:
            lines = file.readlines()

            # apply the line grouping
    
            # check if the line starts with = = = and ends with = = =
            def check_if_line_is_delimiter(line: str, delimiter_strategy: str):
                if delimiter_strategy == "= = =":
                    valid_delimiter = line.startswith("= = =") and line.endswith("= = =\n")
                    return valid_delimiter
                elif delimiter_strategy == "empty line":
                    return line == "\n"
                else:
                    raise ValueError(f"Invalid delimiter strategy: {delimiter_strategy}")
            
            current_story = ""
            for line in lines:
                if check_if_line_is_delimiter(line, grouping_delimiter_strategy):
                    if not (current_story == "" or current_story == "\n"):
                        grouped_dataset.append(current_story)
                        current_story = ""
                # in any case add the line to the current story
                    if not remove_delimiters:
                        current_story += line
                else:
                    current_story += line
            # dont forget to add the last story to the grouped dataset
            if not (current_story == "" or current_story == "\n"):
                grouped_dataset.append(current_story)
            # Convert list to Dataset
            dataset = Dataset.from_dict({"text": grouped_dataset})
            return dataset
        
        
    else:   
        dataset = load_dataset(name=data_set_name,path="text", data_files=dataset_path)["train"]
    return dataset


def load_tinystories_dataset_parts(data_path, split: str):
    data_set_name = "tinystories_"+split
    
    split_name_mapping = {
        "train_38": "train_38m.json", 
        "train_85": "train_85m.json", 
        "train_100": "train_100m.json",
        "train_full": "train_full.json",
        "val": "validation_full.json",
        }
    
    assert split in split_name_mapping, "Split not found"

    data_path += "/tinystories/"+split_name_mapping[split]
    print("dataset_path:",data_path)
    dataset = load_dataset(name=data_set_name,path="json", data_files=data_path)["train"]
    return dataset


def expand_seed_dataset(seed_dataset, min_words=100):
    """
    Expand the seed dataset into multiple starting points for generations.
    
    Args:
        seed_dataset: The dataset to expand.
        min_words: The minimum number of words for each expanded seed.

    Returns:
        The expanded seed dataset with each entry having at least min_words words.
    """
    # Count how many entries are below/above the minimum word threshold
    short_entries = []
    long_entries = []
    
    for i, entry in enumerate(seed_dataset):
        text = entry["text"]
        word_count = len(text.split())
        
        if word_count < min_words:
            short_entries.append(i)
        else:
            long_entries.append(i)
    
    print(f"Initial analysis of seed dataset ({len(seed_dataset)} entries):")
    print(f"- Entries below {min_words} words: {len(short_entries)}")
    print(f"- Entries with {min_words}+ words: {len(long_entries)}")
    
    # Process the dataset to expand longer entries
    expanded_entries = []
    
    # Keep entries that are already above the minimum
    for i in short_entries:
        expanded_entries.append(seed_dataset[i])
    
    # Split longer entries
    for i in long_entries:
        text = seed_dataset[i]["text"]
        
        # Try to split by paragraph breaks first
        chunks = re.split(r'[.!?]\n', text)
        #chunks = text.split(".")

        current_chunk = ""
        current_word_count = 0
        
        for chunk in chunks:
            chunk_word_count = len(chunk.split())
            
            # If adding this chunk would exceed min_words, save current chunk and start a new one
            if current_word_count >= min_words and current_word_count + chunk_word_count > min_words * 2:
                expanded_entries.append({"text": current_chunk})
                current_chunk = chunk
                current_word_count = chunk_word_count
            else:
                # Add delimiter if not the first chunk in current group
                if current_chunk:
                    current_chunk += ".\n"
                current_chunk += chunk
                current_word_count += chunk_word_count
        
        # Don't forget the last chunk
        if current_chunk and current_word_count >= min_words:
            expanded_entries.append({"text": current_chunk})
    
    # Create a new dataset from the expanded entries
    expanded_dataset = Dataset.from_list(expanded_entries)
    
    print(f"Expanded seed dataset: {len(expanded_dataset)} entries")
    
    # Verify the expansion worked as expected
    below_min = 0
    for entry in expanded_dataset:
        if len(entry["text"].split()) < min_words:
            below_min += 1
    
    if below_min > 0:
        print(f"Warning: {below_min} entries still below minimum word count ({min_words})")
    
    if True:
        # Create a histogram of word counts per row
        word_counts = [len(entry["text"].split()) for entry in expanded_dataset]
        import matplotlib.pyplot as plt
        plt.hist(word_counts, bins=100)
        plt.xlabel('Number of words per entry')
        plt.ylabel('Count (log scale)')
        plt.yscale('log')
        plt.title('Distribution of word counts in expanded seed dataset')
        plt.savefig('word_counts_histogram.png')
        plt.close()




    return expanded_dataset




"""
This script contains the functions to load the datasets. 


We first start with a modified version of the BabyLM dataset, which is a collection of 100M stories. 
We replace the [Childes, BNC] part with the [TinyStories] part. 
We hence need 38M words of TinyStories to replace the 38M words of BabyLM. 

Hence our dataset then consists of 100M-38M=62M stories from BabyLM and 38M stories from TinyStories. 

1. Gutenberg: 26M 
2. SimpleWiki: 14.3M
3. OpenSubtitles: 16.8M
4. TinyStories: 38M

We name this dataset TinyBabyLM. 

"""
def construct_tiny_babylm_dataset(path):

    test_gutenberg = load_babylm_dataset_parts(path, split="test", source= "gutenberg", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    all_100m_gutenberg = load_babylm_dataset_parts(path, split="train_100M", source= "gutenberg", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    # split the train into train and seed
    gutenberg_dict = all_100m_gutenberg.train_test_split(test_size=0.05, seed=42)
    train_100m_gutenberg = gutenberg_dict["train"]
    seed_100m_gutenberg = gutenberg_dict["test"]

    test_wiki = load_babylm_dataset_parts(path, split="test", source= "simple_wiki", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    all_100m_wiki = load_babylm_dataset_parts(path, split="train_100M", source= "simple_wiki", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    # split the train into train and seed
    wiki_dict = all_100m_wiki.train_test_split(test_size=0.05, seed=42)
    train_100m_wiki = wiki_dict["train"]
    seed_100m_wiki = wiki_dict["test"]


    test_open_subtitles = load_babylm_dataset_parts(path, split="test", source= "open_subtitles", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    all_100m_open_subtitles = load_babylm_dataset_parts(path, split="train_100M", source= "open_subtitles", apply_line_grouping=True, grouping_delimiter_strategy="empty line",remove_delimiters=True)
    # split the train into train and seed
    open_subtitles_dict = all_100m_open_subtitles.train_test_split(test_size=0.05, seed=42)
    train_100m_open_subtitles = open_subtitles_dict["train"]
    seed_100m_open_subtitles = open_subtitles_dict["test"]


    test_tiny = load_tinystories_dataset_parts(path, split="val")
    all_38m_tiny = load_tinystories_dataset_parts(path, split="train_38")
    # split the train into train and seed
    tiny_dict = all_38m_tiny.train_test_split(test_size=0.05, seed=42)
    train_38m_tiny = tiny_dict["train"]
    seed_38m_tiny = tiny_dict["test"]


    # combine the datasets
    to_combine_train = [train_100m_gutenberg, train_100m_wiki, train_100m_open_subtitles, train_38m_tiny]
    to_combine_test = [test_gutenberg, test_wiki, test_open_subtitles, test_tiny]
    to_combine_seed = [seed_100m_gutenberg, seed_100m_wiki, seed_100m_open_subtitles, seed_38m_tiny]


    # now one can inspect the datasets
    # first go over the datasets and print out the first x rows and the number of rows
    if False:
        print("Test datasets:")
        for i, dataset in enumerate(to_combine_test):
            print(f"Dataset {i}:", dataset)
            print("-"*50)
            # print one sample 
            print("First sample has", len(dataset[0]["text"].split(" ")), "words. Fist 100 chars", dataset[0]["text"][:500])
            print("-"*50)
            print(f"Number of rows: {len(dataset)}")
            print("-"*50)
        print("-"*50)
    
    if True:
        print("Check the ratios of the datasets are roughly correct and balanced accross splits/train/test")
        # print the number of words in the datasets
        print("Train datasets:")
        for i, dataset in enumerate(to_combine_train):
            print(f"Number of words in dataset {i}: {count_words_in_dataset(dataset)}")
        print("-"*50)
        print("Test datasets:")
        # print the number of words in the datasets
        for i, dataset in enumerate(to_combine_test):
            print(f"Number of words in dataset {i}: {count_words_in_dataset(dataset)}")
        print("-"*50)
        print("Seed datasets:")
        for i, dataset in enumerate(to_combine_seed):
            print(f"Number of words in dataset {i}: {count_words_in_dataset(dataset)}")
        print("-"*50)

    # combine the datasets
    train_concat = concatenate_datasets(to_combine_train)
    test_concat = concatenate_datasets(to_combine_test)
    seed_concat = concatenate_datasets(to_combine_seed)

    # now the seed should be split into many at most ? token long good possible starting points for generations...
    print("Seed dataset before expansion:", seed_concat)
    seed_concat = expand_seed_dataset(seed_concat)
    print("Seed dataset after expansion:", seed_concat)

    import random
    if False:
        # print a few samples of the seed dataset
        print("-"*50)
        print("-"*50)
        print("First 5 samples of the seed dataset:")
        for i in range(20):
            # get random index in 0, len(seed_concat)
            random_idx = random.randint(0, len(seed_concat)-1)
            print(seed_concat[random_idx]["text"])
            print("-"*50)
        print("-"*50)
        print("-"*50)


    # create a dataset dict
    tinybabylm_dataset = DatasetDict({
        "train": train_concat,
        "test": test_concat,
        "seed": seed_concat
    })

    return tinybabylm_dataset



def main():

    path = "../data/"

    tinybabylm_dataset = construct_tiny_babylm_dataset(path)
    
    print(tinybabylm_dataset)

    print("Number of words in train:",count_words_in_dataset(tinybabylm_dataset["train"]))
    print("Number of words in test:",count_words_in_dataset(tinybabylm_dataset["test"]))
    print("Number of words in seed:",count_words_in_dataset(tinybabylm_dataset["seed"]))

    # now save the dataset
    #path = "../data/tinybabylm_dataset"
    #tinybabylm_dataset.save_to_disk(path)


if __name__ == "__main__":
    main()




