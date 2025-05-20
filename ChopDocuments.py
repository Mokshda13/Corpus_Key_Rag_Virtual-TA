# ChopDocuments.py
# This code takes all pdfs in documents, scans them, then generates paragraph embeddings on a sliding scale of 200 tokens (roughly 150 words)
# make sure you change all mapping



import os
from PyPDF2 import PdfReader 
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import docx
import re

"""
next time you need to run things, change project folders

project_folders=["input", "chunks", "embeddings", "bundle", "output"]



"""

# load user settings and api key
def read_settings(file_name):
    settings = {}
    with open(file_name, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            settings[key] = value
    return settings


nltk.download('punkt_tab')
nltk.data.path.append(r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing\documents")
def preprocess_text(text):
    # Remove any non-alphanumeric characters (keeping spaces)
    #cleaned_text = ''.join(e if e.isprintable() or e.isspace() else ' ' for e in text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Replace any character that is not alphanumeric or whitespace with ''

    # Tokenize the text and convert to lowercase
    tokens = word_tokenize(cleaned_text.lower())
    
    # Join the tokens back into a cleaned text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def chop_documents(filedirectory, output_folder, chunk_size, overlap_size):
    # Loop through all pdf, txt, tex in the "input" folder
    supported_filetypes = ('.pdf', '.doc', '.docx', '.txt', '.tex') # picked from cfg
    for filename in os.listdir(filedirectory):
        if filename.endswith(supported_filetypes):
            # Create an empty DataFrame to store the text and title of each document
            df = pd.DataFrame(columns=["Title", "Text"])
            print("Loading " + filename)
            if filename.endswith(".pdf"):
                # Open the PDF file in read-binary mode
                filepath = os.path.join(filedirectory, filename)
                reader = PdfReader(filepath)

                # Extract the text from each page of the PDF
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                # Add the text and title to the DataFrame
                title = os.path.splitext(filename)[0]  # Remove the file extension from the filename
                new_row = pd.DataFrame({"Title": [title], "Text": [text]})
                df = pd.concat([df, new_row], ignore_index=True)

            elif filename.endswith(".doc") or filename.endswith(".docx"):
                # Open the DOC/DOCX file in binary mode and read the raw data
                filepath = os.path.join(filedirectory, filename)
                doc = docx.Document(filepath)

                # Convert the file to UTF-8 and extract the text
                text = ''
                for paragraph in doc.paragraphs:
                    text += paragraph.text

                # Add the text and title to the DataFrame
                title = os.path.splitext(filename)[0]  # Remove the file extension from the filename
                new_row = pd.DataFrame({"Title": [title], "Text": [text]})
                df = pd.concat([df, new_row], ignore_index=True)

            elif filename.endswith(".txt"):
                # Open the text file and read its contents
                filepath = os.path.join(filedirectory, filename)
                with open(filepath, "r", encoding="latin-1") as file:
                    text = file.read()

                # Add the text and title to the DataFrame
                title = os.path.splitext(filename)[0]  # Remove the file extension from the filename
                new_row = pd.DataFrame({"Title": [title], "Text": [text]})
                df = pd.concat([df, new_row], ignore_index=True)
                
            elif filename.endswith(".tex"):
                # Use regular expressions to extract regular text from the LaTeX file
                filepath = os.path.join(filedirectory, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    text = file.read()
                
                # Add the text and title to the DataFrame
                title = os.path.splitext(filename)[0] # Remove the file extension from the filename
                new_row = pd.DataFrame({"Title": [title], "Text": [text]})
                df = pd.concat([df, new_row], ignore_index=True)

            # Loop through the rows and create overlapping chunks for each text
            df['Text'] = df['Text'].apply(preprocess_text)
            
            chunks = []
            for i, row in df.iterrows():
                # Tokenize the text for the current row
                tokens = nltk.word_tokenize(row['Text'])

                # Loop through the tokens and create overlapping chunks
                for j in range(0, len(tokens), chunk_size - overlap_size):
                    # Get the start and end indices of the current chunk
                    start = j
                    end = j + chunk_size

                    # Create the current chunk by joining the tokens within the start and end indices
                    chunk = ' '.join(tokens[start:end])

                    # Add the article title to the beginning of the chunk
                    # chunk_with_title = "This text comes from the document " + row['Title'] + ". " + chunk

                    # this step used to read chunk_with_title versus chunk
                    chunks.append([row['Title'], chunk])

            # Convert the list of chunks to a dataframe
            df_chunks = pd.DataFrame(chunks, columns=['Title', 'Text'])

            # Truncate the filename if it's too long, e.g., limit to 250 characters
            max_filename_length = 250
            if len(filename) > max_filename_length:
                filename = filename[:max_filename_length]

            # Remove the file extension from the filename
            filename_without_extension = os.path.splitext(filename)[0]

            # Save the df_chunks to the output_folder subfolder with the new file name
            output_file = os.path.join(output_folder, filename_without_extension + "-originaltext.csv")
            df_chunks.to_csv(output_file, encoding='utf-8', escapechar='\\', index=False)

            print("Saving " + filename)

def make_chunks_folder(corpa_directory):
    output_folder = os.path.join(corpa_directory, "chunks")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

if __name__ == "__main__":
    print("This is module.py being run directly.")
    os.chdir(r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing")
    current_dir = os.getcwd()

    corpa_directory = os.path.join(current_dir)
    
    settings_file_path = os.path.join(corpa_directory, "settings.txt")
    
    settings = read_settings(settings_file_path)
    filedirectory = os.path.join(corpa_directory, "documents_test")
    output_folder = make_chunks_folder(corpa_directory)
    chunk_size = 200
    overlap_size = 100
    chop_documents(filedirectory, output_folder, chunk_size, overlap_size)




