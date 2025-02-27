import PyPDF2


def pdf_to_text(pdf_path, output_txt_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += f"\n--- Page {page_num + 1} ---\n" + text

            # Write the extracted text to a file
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(full_text)

        print(f"Successfully converted '{pdf_path}' to '{output_txt_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_pdf = "Coursera.pdf"    # Replace with your PDF file path
    output_text = "output.txt"   # Replace with your desired output file path
    pdf_to_text(input_pdf, output_text)
