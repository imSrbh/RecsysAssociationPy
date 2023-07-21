# Use the official Python image as the base image
FROM python:3.7

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install the required packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app.py and OnlineRetail.csv files to the working directory in the container
COPY app.py /app/
COPY OnlineRetail.csv /app/

# Expose port 5000 to access the Flask app
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]
