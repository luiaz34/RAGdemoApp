#Create python virtual environment
python -m venv venv
.\venv\Scripts\activate


#Install necessary packages
pip install -r requirments.txt

#start the app
uvicorn main:app --reload

#upload the pdf file you want

#ask any question related to that pdf
