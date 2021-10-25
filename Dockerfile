# Use an official Python 3 runtime as a parent image
FROM huggingface/transformers-pytorch-gpu:4

# Set the /app directory as the work directory
WORKDIR /app

# Copy the requirements file to the workdir 
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install torchvision==0.10.0
RUN pip install -r requirements.txt
#RUN python -m spacy download en_core_web_sm
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz

#Copy the remaining files to the working directory
COPY ./src /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches

ENTRYPOINT [ "python3" ]

CMD ["main.py"]
