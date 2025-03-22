import pandas as pd

# Load the Excel file
excel_file_path = 'questions_with_answers_and_context.xlsx'  # Replace with your Excel file path
df = pd.read_excel(excel_file_path)

# Debug: Print column names and first few rows
print("Column names in the DataFrame:", df.columns.tolist())
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Filter out rows where the "Answer" column contains "unsure" or error message about rate limits
df = df[~df['Answer'].astype(str).str.contains("unsure", case=False)]
df = df[~df['Answer'].astype(str).str.contains("Error: Error response 429", case=False)]

# Save the cleaned data to a new Excel file
cleaned_excel_file_path = 'cleaned_file_with_context_excel.xlsx'  # Replace with your desired output file path
df.to_excel(cleaned_excel_file_path, index=False)

print(f"Cleaned data saved to {cleaned_excel_file_path}")