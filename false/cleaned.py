import pandas as pd
from odf import opendocument, table
from odf import text  # Import the text module

# Load the ODS file
ods_file_path = 'refined_questions_with_context.ods'  # Replace with your file path
doc = opendocument.load(ods_file_path)

# Assuming the data is in the first sheet
sheet = doc.spreadsheet.getElementsByType(table.Table)[0]

# Extract data from the sheet (preserve line breaks)
data = []
for row in sheet.getElementsByType(table.TableRow):
    row_data = []
    for cell in row.getElementsByType(table.TableCell):
        # Get cell content with line breaks preserved
        paragraphs = cell.getElementsByType(text.P)
        cell_value = '\n'.join([str(p) for p in paragraphs])
        row_data.append(cell_value)
    data.append(row_data)

# Create a DataFrame
df = pd.DataFrame(data[1:], columns=data[0])  # First row is headers

# Debug: Print column names and first few rows
print("Column names in the DataFrame:", df.columns.tolist())
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Filter out rows where the "Answer" column starts with "I'm unsure"
df = df[~df['Answer'].astype(str).str.startswith("I'm unsure")]

# Save the cleaned data back to the ODS file (preserving line breaks)
cleaned_ods_file_path = 'cleaned_file_with_context.ods'  # Replace with your desired output file path

# Create a new ODS document
new_doc = opendocument.OpenDocumentSpreadsheet()

# Create a new sheet
new_sheet = table.Table(name="Sheet1")
new_doc.spreadsheet.addElement(new_sheet)

# Add headers
header_row = table.TableRow()
for col_name in df.columns:
    cell = table.TableCell(valuetype="string")
    # Split header by newlines (if any) and add as separate paragraphs
    for line in str(col_name).split('\n'):
        p = text.P(text=line)
        cell.addElement(p)
    header_row.addElement(cell)
new_sheet.addElement(header_row)

# Add data
for _, row in df.iterrows():
    data_row = table.TableRow()
    for value in row:
        cell = table.TableCell(valuetype="string")
        # Split value by newlines and add each as a separate paragraph
        for line in str(value).split('\n'):
            p = text.P(text=line)
            cell.addElement(p)
        data_row.addElement(cell)
    new_sheet.addElement(data_row)

# Save the document
new_doc.save(cleaned_ods_file_path)

print(f"Cleaned data saved to {cleaned_ods_file_path}")