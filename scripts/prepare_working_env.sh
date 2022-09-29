#########################
# ENVIRONMENT VARIABLES #
#########################

export UNXSCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export UNXAPPLI="$( dirname $UNXSCRIPT )"
export UNXCONF="$UNXAPPLI/conf"
export UNXDATA="$UNXAPPLI/data"
export UNXNOTEBOOK="$UNXAPPLI/notebook"
echo "Loading prepare_working_env.sh from this repository: $UNXAPPLI"

# This folder contains the source code of the application
export UNXPACKAGE="src/image_segmentation"

if [ ! -e $UNXPACKAGE ]
then
    echo "ERROR! Python package '$UNXPACKAGE' not found in this repository"
    return
fi

#########################
# GENERAL CONFIGURATION #
#########################

export TF_CPP_MIN_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1

###############
# VIRTUAL ENV #
###############
if [ -e $UNXAPPLI/.venv/bin/activate ]
then
    echo "Virtual Env has been found in this repository."
    source $UNXAPPLI/.venv/bin/activate
else
    echo "No virtual env find in '$UNXAPPLI'. Start creating one."
    python3 -m venv $UNXAPPLI/.venv
    source $UNXAPPLI/.venv/bin/activate
    pip install -r $UNXAPPLI/requirements.txt
    pip install -e $UNXAPPLI
fi
