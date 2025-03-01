import requests
from io import BytesIO
from PyPDF2 import PdfReader


def read_pdf_from_url(url):
    try:
        # Send GET request to the URL
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:
            # Create a BytesIO object from the response content
            pdf_file = BytesIO(response.content)

            # Create PDF reader object
            pdf_reader = PdfReader(pdf_file)

            # Get number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Number of pages: {num_pages}")

            # Extract text from all pages
            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"

            # Print or return the text
            print("PDF Content:")
            print(full_text)

            return full_text

        else:
            print(f"Failed to access PDF. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None
    except Exception as e:
        print(f"PDF processing error occurred: {e}")
        return None


# URL provided
url = "https://docsysmanage.s3.ap-south-1.amazonaws.com/2021_2_English.pdf"

# Call the function
text = read_pdf_from_url(url)
print(text)
