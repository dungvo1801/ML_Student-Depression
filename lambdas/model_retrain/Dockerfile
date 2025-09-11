# Use AWS official Python 3.9 Lambda base image
FROM public.ecr.aws/lambda/python:3.10
 
# Copy your function code to /var/task (the Lambda working directory)
COPY . ${LAMBDA_TASK_ROOT}/
 
# Install dependencies - you can list all needed in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}
 
# (Optional) If you have large dependencies, consider cleaning cache:
RUN rm -rf ~/.cache/pip
 
# Set the CMD to your handler (file.function)
CMD ["lambda_function.lambda_handler"]