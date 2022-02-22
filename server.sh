docker run -p 8500:8500 \
    -p 8501:8501 \
    --mount type=bind,source="$(pwd)"/saved_models,target=/models/covid \
    -e MODEL_NAME=covid \
    -t tensorflow/serving