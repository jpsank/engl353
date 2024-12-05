import pypdf
from word2num import word2num

reader = pypdf.PdfReader('Claude_Closky_1000.pdf')

for page in reader.pages:
    text = page.extract_text()
    nums = text.split(",")
    for num in nums:
        print(num.strip().lower())
        print(word2num(num.strip().lower()))
