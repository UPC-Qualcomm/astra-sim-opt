docker run -it\
    --name xavi_astra_new \
    -v ~/astra-sim:/app/astra-sim \
    -w /app/astra-sim \
    astra-sim:dev \
    "$@"