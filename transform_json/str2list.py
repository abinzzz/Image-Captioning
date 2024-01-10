import json

"""这里面list的元素是每句话"""
# Load the JSON file
#file_path = '/Users/chenyubin/Desktop/no_emo/github/ViT1/captions/generated_captions1.json'

file_path = '/data/df/test_captions.json'  # Replace with your file path

with open(file_path, 'r') as file:
    data = json.load(file)

# Splitting the values into sentences and updating the dictionary
for key, value in data.items():
    # Splitting the value string into a list of sentences
    sentences = value.split('. ')
    # Making sure that each sentence ends with a period
    sentences = [sentence if sentence.endswith('.') else sentence + '.' for sentence in sentences if sentence]
    # Updating the dictionary with the list of sentences
    data[key] = sentences

# Saving the transformed data to a new JSON file
transformed_file_path = '/metrics/data/RES.json'  # Replace with your desired output file path
with open(transformed_file_path, 'w') as file:
    json.dump(data, file, indent=4)
