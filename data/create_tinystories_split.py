from datasets import load_dataset
from tqdm import tqdm
import os
import json
# Set a fixed random seed for reproducibility
RANDOM_SEED = 42


def count_words_in_split(dataset):
    num_rows = dataset.num_rows
    print(f"Number of rows: {num_rows}")
    count = 0
    for i in tqdm(range(num_rows)):
        line = dataset[i]["text"]
        count += len(line.split(' '))
    return count

# Function to create a subset of the dataset with a specific word count limit
def create_word_limited_split(dataset, word_limit, output_file, shuffle=True):
    # Shuffle the dataset if requested
    if shuffle:
        print(f"Shuffling dataset with random seed {RANDOM_SEED}")
        dataset = dataset.shuffle(seed=RANDOM_SEED)
    
    num_items = len(dataset)
    word_count = 0
    stories = []
    
    print(f"Creating split with word limit: {word_limit}")
    
    for i in tqdm(range(num_items)):
        story = dataset[i]["text"]
        story_word_count = len(story.split(' '))
        
        if word_count + story_word_count <= word_limit:
            stories.append({"text": story})
            word_count += story_word_count
        else:
            break
    
    print(f"Created split with {len(stories)} stories and {word_count} words")
    
    # Save the split to a JSON file
    with open(output_file, 'w') as f:
        json.dump(stories, f)
    
    print(f"Saved split to {output_file}")
    return stories, word_count






def main():
    
    # Define the data path directory
    data_path = "../data/tinystories"
    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Load the dataset
    ds = load_dataset("roneneldan/TinyStories")
    print(ds)


    # Save the full dataset
    train_file = os.path.join(data_path, "train_full.json")
    val_file = os.path.join(data_path, "validation_full.json")

    # Storing the full dataset and validation set

    with open(train_file, 'w') as f:
        json.dump([{"text": item["text"]} for item in ds["train"]], f)

    with open(val_file, 'w') as f:
        json.dump([{"text": item["text"]} for item in ds["validation"]], f)

    print(f"Full dataset saved to {data_path}")

    
    # Create a 100M word split of the training data
    split_100m_file = os.path.join(data_path, "train_100m.json")
    stories_100m, word_count_100m = create_word_limited_split(ds["train"], 100_000_000, split_100m_file)

    split_85m_file = os.path.join(data_path, "train_85m.json")
    stories_85m, word_count_85m = create_word_limited_split(ds["train"], 85_000_000, split_85m_file)

    # generate one with 38M words
    split_38m_file = os.path.join(data_path, "train_38m.json")
    stories_38m, word_count_38m = create_word_limited_split(ds["train"], 38_000_000, split_38m_file)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Full train set: {count_words_in_split(ds['train'])} words")
    print(f"Full validation set: {count_words_in_split(ds['validation'])} words")
    print(f"100M split: {word_count_100m} words ({len(stories_100m)} stories)")
    print(f"85M split: {word_count_85m} words ({len(stories_85m)} stories)")
    print(f"38M split: {word_count_38m} words ({len(stories_38m)} stories)")
    

if __name__ == "__main__":
    main()