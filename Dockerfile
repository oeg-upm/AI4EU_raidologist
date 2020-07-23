# Use an official Python 3 runtime as a parent image
FROM python:3

# Set the /app directory as the work directory
WORKDIR /app

# Copy the requirements file to the workdir 
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

#Copy the remaining files to the working directory
COPY ./src /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["python", "main.py"]
