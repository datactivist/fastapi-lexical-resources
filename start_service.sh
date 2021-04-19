. api-config.config     # Read config file

Help()
{
   # Display Help
   echo "Start embeddins API service using api-config.config configuration."
   echo
   echo "Syntax: start_service.sh [-d|-s|-r|-h]"
   echo "options:"
   echo "d     Run the docker image on a container with the same name"
   echo "s     Run the docker image in silent mode (when using -d option)"
   echo "r     Run the API in reload mode (development)"
   echo "h     Show this help panel"
   echo
}

flag_start_docker=false
flag_silence_docker=false
flag_reload=false

while [ -n "$1" ]; do # while loop starts

	case "$1" in

    -d) flag_start_docker=true ;;

	-s) flag_silence_docker=true ;;

    -r) flag_reload=true ;;

    -h) Help 
        exit;;

	*) echo "Option $1 not recognized, use option -h to see available options" 
       exit;; # In case you typed a different option other than a,b,c

	esac

	shift

done


# Updating embeddings activation config
echo "Updating embeddings configuration"
python3 app/preprocess/update_embeddingsconfig.py $embeddings_config
echo "Done"
echo ""

# Downloading missing embeddings and converting them
echo "Downloading missing embeddings, this can take a while..."
python3 app/preprocess/download_embeddings.py
echo "Downloading done!"
echo ""


start_docker()
{
    if $flag_start_docker; then
        if $flag_silence_docker; then
            sudo docker run -d --name $docker_name -h $API_host_name -p $API_port:80 $docker_name:$docker_version
        else
            sudo docker run --name $docker_name -h $API_host_name -p $API_port:80 $docker_name:$docker_version
        fi
    else
        echo "Not starting docker"
    fi
    echo ""
}

start_local_mode()
{
    cd app
    bash ./prestart.sh -l
    
    if $flag_reload; then
        uvicorn main:app --host $API_host_name --port $API_port --reload
    else
        uvicorn main:app --host $API_host_name --port $API_port
    fi
    
}

if [ $deployment_method == "local" ]; then
    start_local_mode
elif [ $deployment_method == "docker" ]; then
    sudo docker build -t $docker_name:$docker_version .
    start_docker
else
    echo "'$deployment_method' is not a valid value for deployment_method in config.config"
fi
