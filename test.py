import pandas as pd
import chardet

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]

def read_csv_with_encoding(file_path):
    encodings_to_try = ["utf-8", "ISO-8859-1", "latin1", "utf-16"]
    detected_encoding = detect_encoding(file_path)
    print(f"Detected Encoding: {detected_encoding}")
    
    for encoding in [detected_encoding] + encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            print("File loaded successfully!")
            print(df.head())
            return df  # Return DataFrame if successful
        except Exception as e:
            print(f"Failed with encoding {encoding}: {e}")
    
    print("All encoding attempts failed.")
    return None

if __name__ == "__main__":
    file_name = "b4.csv"  # Change to your actual file name
    df = read_csv_with_encoding(file_name)
    if df is not None:
        print("CSV loaded successfully!")
    else:
        print("Failed to load CSV.")