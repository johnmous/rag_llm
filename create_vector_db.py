import pypdf

# Open the PDF file in read-binary mode
with open('/home/imoustakas/llm_playground/RAG/Sohil et al. - 2022 - An introduction to statistical learning with appli.pdf', 'rb') as file:
    # Create a PDF reader object
    reader = pypdf.PdfReader(file)

    # Get the total number of pages in the PDF
    num_pages = len(reader.pages)
    print(num_pages)

    print(reader.pages[12].extract_text())


    # # Loop through each page and extract the text
    # counter = 0
    # for page in reader.pages:
    #     text = page.extract_text()
    #     # Do something with the extracted text
    #     print(f"Page: {counter}")
    #     print(text)
    #     if counter > 10:
    #         break
    #     counter += 1
